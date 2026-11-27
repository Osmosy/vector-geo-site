"""Море в кольце: не один признак, а согласие нескольких (S2, E-038).

Раньше море определялось только береговой линией OSM: `sea_in_ring = coast_m <= 2000`.
Это отказывало Суве (62.9 % воды по снимку) и Сиднею (залив Порт-Джексон нарисован полигонами),
потому что `natural=coastline` в кольце не попадал в выборку. Для фэншуй гавань и океан — это
море, главная вода перед участком, поэтому «нет береговой линии» нельзя выдавать за «моря нет».

Признаки (каждый попадает в `evidence`):

1. `coastline` — береговая линия OSM в круге 8 км (расстояние по геометрии отрезков).
2. `coastline_wide` — береговая линия в расширенном круге до 15 км: море есть, но берег дальше
   ближнего кольца.
3. `polygon` — площадь воды с признаком моря: `natural=bay`, `water=bay|lagoon|harbour`,
   `place=sea|ocean`, `seamark:*` (в пакете такие площади уже идут с `kind: sea`).
4. `tidal` — приливная вода, соединённая с береговой линией.
5. `satellite` — снимок: доля воды WorldCover ≥ 30 % или постоянная вода GSW ≥ 30 %.
   Это подтверждение, а не единственный признак.

Правила согласия:

- есть любой из признаков 1, 3, 4 → `present: true`;
- OSM молчит, но снимок показывает ≥ 30 % постоянной воды → `present: null`,
  `confidence: low`, предупреждение `sea_conflict`: тихо писать `false` запрещено;
- нет ничего и снимок ниже порога → `present: false`.

Старое `sea_in_ring` остаётся производным (`present and nearest_m <= 2000`) и помечено
устаревшим: у Сувы и Сиднея оно давало `false` при очевидном море.
"""

from __future__ import annotations

ALGO_VERSION = "1.0.0"

NEAR_KM = 8.0              # кольцо участка
WIDE_KM = 15.0             # расширенный круг для береговой линии
INLAND_EXPLAINS_RATIO = 0.7   # снимок объяснён внутренними водами
SATELLITE_MIN = 0.30       # доля воды по снимку, ниже которой снимок не спорит с OSM
DEPRECATED_NEAR_M = 2000.0

SEA_KINDS = ("ocean", "bay", "harbour", "lagoon", "estuary")

# Порядок старшинства вида: залив и гавань конкретнее океана по умолчанию
KIND_PRIORITY = ["bay", "harbour", "lagoon", "estuary", "ocean"]

POLYGON_KIND_RU = {"ocean": "океан", "bay": "залив", "harbour": "гавань",
                   "lagoon": "лагуна", "estuary": "устье"}


def _num(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _polygon_kind(item: dict) -> str | None:
    """Вид моря по тегам площади: залив, гавань, лагуна или океан."""
    tags = item.get("tags") or {}
    names = " ".join(str(tags.get(k) or "") for k in ("water", "natural", "place", "seamark:type"))
    name = str(item.get("name") or "").lower()
    # «Port» в имени — не гавань: Port Jackson и Port Phillip это заливы, поэтому вид берём
    # из тегов и из слов harbour/harbor, а не из общего port
    if "harbour" in names or "harbor" in names or "harbour" in name or "harbor" in name:
        return "harbour"
    if "lagoon" in names or "lagoon" in name:
        return "lagoon"
    if "bay" in names or "bay" in name or "gulf" in name:
        return "bay"
    if "sea" in names or "ocean" in names or " strait" in name:
        return "ocean"
    return None


def detect(*, water: list[dict], radius_km: float = NEAR_KM, coast_m: float | None,
           coast_wide_m: float | None = None, wc_share: float | None = None,
           gsw_share: float | None = None, sea_polygons: list[dict] | None = None,
           osm_water_share: float | None = None) -> dict:
    """Свести признаки моря в одно поле `sea` с доказательствами и уверенностью."""
    evidence: list[dict] = []
    near_m = radius_km * 1000.0

    if coast_m is not None and coast_m <= near_m:
        evidence.append({"kind": "coastline", "distance_m": round(coast_m, 1),
                         "ring_km": radius_km,
                         "source": "OSM natural=coastline"})
    if coast_wide_m is not None and coast_wide_m <= WIDE_KM * 1000.0 and (
            coast_m is None or coast_m > near_m):
        evidence.append({"kind": "coastline_wide", "distance_m": round(coast_wide_m, 1),
                         "ring_km": WIDE_KM, "source": "OSM natural=coastline"})

    polygons: list[dict] = []
    tidal: list[dict] = []
    for item in water or []:
        distance = _num(item.get("nearest_m"))
        if item.get("kind") == "sea" and (distance is None or distance <= near_m):
            kind = _polygon_kind(item)
            polygons.append({"kind": kind or "ocean", "name": item.get("name"),
                             "distance_m": None if distance is None else round(distance, 1),
                             "area_m2": _num(item.get("area_m2"))})
            continue
        # приливная вода — только если она связана с берегом: иначе приливная река в глубине
        # материка давала бы «море» без моря
        near_coast = coast_m is not None and coast_m <= WIDE_KM * 1000.0
        if item.get("tidal") is True and near_coast and (distance is None or distance <= near_m):
            tidal.append({"name": item.get("name"), "kind": item.get("kind"),
                          "distance_m": None if distance is None else round(distance, 1)})

    for item in sea_polygons or []:
        polygons.append({"kind": item.get("kind") or "ocean", "name": item.get("name"),
                         "distance_m": item.get("distance_m"), "area_m2": None})
    # Гавань без берега рядом — это портовый бассейн, а не море: у Амстердама 13 таких
    # площадей в Западном порту, и они выдавали город за приморский. Залив (natural=bay)
    # и море (place=sea) остаются признаком сами по себе.
    # Гавань и залив без берега рядом — это портовый бассейн или залив внутреннего водоёма,
    # а не море: у Амстердама 13 гаваней Западного порта и IJburgbaai на Эймере выдавали город
    # за приморский. Самостоятельным признаком остаются только место-море (place=sea|ocean)
    # и пролив (water=strait).
    # Залив и гавань — признак, но не приговор: ТЗ требует подтверждения вторым признаком
    # (берег в 15 км, прилив или снимок). Иначе IJburgbaai на Эймере делал Амстердам приморским
    # городом, а Ханчжоу с Чунцином — тоже: их «вода» это озеро и русла рек.
    coast_is_near = coast_m is not None and coast_m <= WIDE_KM * 1000.0
    sat_share = max([v for v in (wc_share, gsw_share) if v is not None], default=None)
    corroborated = coast_is_near or bool(tidal) or (sat_share is not None
                                                    and sat_share >= SATELLITE_MIN)
    dropped: list[str] = []
    if not corroborated:
        kept = []
        for item in polygons:
            if item["kind"] in {"harbour", "bay", "lagoon"}:
                dropped.append(item.get("name") or item["kind"])
                continue
            kept.append(item)
        polygons = kept
    ordered = sorted(polygons, key=lambda p: p.get("distance_m") if p.get("distance_m") is not None
                     else 1e12)
    for item in ordered[:4]:
        evidence.append({"kind": "polygon", "sea_kind": item["kind"], "name": item["name"],
                         "distance_m": item["distance_m"], "area_m2": item["area_m2"],
                         "source": "OSM natural=bay / water=bay|lagoon|harbour / place=sea"})
    if len(ordered) > 4:
        evidence.append({"kind": "polygon", "count": len(ordered) - 4,
                         "note": "остальные площади моря: имена в water.geojson"})
    if tidal:
        # один признак на все приливные объекты: иначе у Сувы в evidence попадало 60 записей
        # «tidal», и доказательство тонуло в перечислении
        nearest_tidal = min((t for t in tidal if t.get("distance_m") is not None),
                            key=lambda t: t["distance_m"], default=None)
        evidence.append({"kind": "tidal", "count": len(tidal),
                         "name": (nearest_tidal or {}).get("name"),
                         "distance_m": (nearest_tidal or {}).get("distance_m"),
                         "source": "OSM tidal=yes + береговая линия в 15 км"})

    satellite_share = max([v for v in (wc_share, gsw_share) if v is not None], default=None)
    if satellite_share is not None and satellite_share >= SATELLITE_MIN:
        evidence.append({"kind": "satellite", "share": round(float(satellite_share), 4),
                         "threshold": SATELLITE_MIN,
                         "source": "ESA WorldCover класс 80 / JRC GSW permanent"})

    osm_evidence = [e for e in evidence if e["kind"] in {"coastline", "coastline_wide",
                                                         "polygon", "tidal"}]
    distances = [e["distance_m"] for e in osm_evidence if e.get("distance_m") is not None]
    nearest = min(distances) if distances else None

    if dropped:
        evidence.append({"kind": "polygon_dropped", "count": len(dropped),
                         "names": dropped[:4],
                         "note": "гавани и заливы без береговой линии в 15 км — это не море"})

    if polygons:
        kinds = {p["kind"] for p in polygons}
        kind = next((k for k in KIND_PRIORITY if k in kinds), "ocean")
    elif osm_evidence:
        kind = "ocean"
    else:
        kind = None

    if osm_evidence:
        present: bool | None = True
        confidence = "high" if nearest is not None and nearest <= DEPRECATED_NEAR_M else "medium"
        if any(e["kind"] == "coastline_wide" for e in osm_evidence) and not any(
                e["kind"] in {"coastline", "polygon", "tidal"} for e in osm_evidence):
            confidence = "medium"
        note = None
    elif satellite_share is not None and satellite_share >= SATELLITE_MIN and (
            osm_water_share is not None
            and osm_water_share >= satellite_share * INLAND_EXPLAINS_RATIO):
        # Ханчжоу и Чунцин: снимок видит 40 % и 35 % воды, но это Западное озеро и русла
        # Янцзы с Цзялином, а не море. Когда OSM-вода объясняет почти всю воду снимка,
        # морем это не считается — и в поле пишется, чем именно объясняется.
        present = False
        confidence = "medium"
        note = ("снимок показывает %.1f %% воды, но это внутренние воды (OSM %.1f %%): "
                "море не подтверждено" % (satellite_share * 100, osm_water_share * 100))
    elif satellite_share is not None and satellite_share >= SATELLITE_MIN:
        present = None
        confidence = "low"
        note = ("OSM моря не видит, а снимок показывает %.1f %% воды в кольце — признаки "
                "противоречат, ответ «не знаю»" % (satellite_share * 100))
    elif satellite_share is None and not osm_evidence and not water:
        # Ни OSM, ни снимка: у арктической точки 84.3° с. ш. нет ни береговой линии, ни
        # покрытия WorldCover. «Нет данных» — это не «моря нет»: тот участок стоит в океане.
        present = None
        confidence = "low"
        note = "нет ни OSM-признаков, ни снимка — море не подтверждено и не опровергнуто"
    else:
        present = False
        confidence = "medium"
        note = None

    sectors = sorted({s for s in (item.get("nearest_mountain") for item in (water or [])
                                  if item.get("kind") == "sea") if s})
    return {
        "present": present,
        "kind": kind,
        "kind_ru": POLYGON_KIND_RU.get(kind) if kind else None,
        "nearest_m": None if nearest is None else round(nearest, 1),
        "sectors": sectors,
        "confidence": confidence,
        "evidence": evidence,
        "note": note,
        "radius_km": radius_km,
        # устаревшее поле: оставлено, чтобы старые пакеты и навыки не сломались
        "deprecated_sea_in_ring": bool(present is True and nearest is not None
                                       and nearest <= DEPRECATED_NEAR_M),
    }


def _query(lat: float, lon: float, radius_m: int) -> str:
    """Признаки моря отдельным запросом: заливы, гавани, лагуны, море по имени.

    Слой воды (`natural=water`) их не приносит: Порт-Джексон в OSM помечен `natural=bay`,
    а море — `place=sea`. `place=sea|ocean` берём как `out tags center`: полигон океана
    огромен, и его геометрия в ответе не нужна — для признака хватает центра и тега.
    """
    r = int(radius_m)
    return (
        "[out:json][timeout:90];"
        f'(way(around:{r},{lat},{lon})["natural"="bay"];'
        f'relation(around:{r},{lat},{lon})["natural"="bay"];'
        # лагуну не спрашиваем: Laguna de los Coipos в Буэнос-Айресе — пруд в заповеднике,
        # а не море; прибрежные лагуны видны по береговой линии и без этого тега
        f'way(around:{r},{lat},{lon})["water"~"^(bay|harbour|strait)$"];'
        f'relation(around:{r},{lat},{lon})["water"~"^(bay|harbour|strait)$"];'
        f'way(around:{r},{lat},{lon})["seamark:type"~"^(harbour|bay|sea)$"];'
        f'node(around:{r},{lat},{lon})["place"~"^(sea|ocean)$"];'
        f'way(around:{r},{lat},{lon})["place"~"^(sea|ocean)$"];'
        ");out tags center;"
    )


def fetch_payload(lat: float, lon: float, radius_m: int = 8000, *, allow_network: bool = True,
                  retries: int = 2) -> dict | None:
    """Ответ Overpass с признаками моря; `None` — не смотрели (офлайн без кеша)."""
    import hashlib
    import json
    import os
    from pathlib import Path

    cache_dir = Path(os.environ.get("GEOSITE_CACHE", Path.home() / ".cache" / "vector-geo-site"))
    cache_dir = cache_dir / "osm"
    key = hashlib.sha256(f"sea,{lat:.4f},{lon:.4f},{radius_m}".encode()).hexdigest()[:20]
    path = cache_dir / f"sea-{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    if not allow_network or os.environ.get("GEOSITE_OFFLINE") == "1":
        return None
    from . import water
    try:
        payload = water._ask(_query(lat, lon, radius_m), retries=retries)
    except Exception:
        return None
    if payload.get("elements"):
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return payload


def payload_polygons(payload: dict | None, lat: float, lon: float,
                     radius_km: float = 8.0) -> list[dict]:
    """Полигоны моря из отдельного запроса: вид по тегам, расстояние по центру."""
    import math

    out: list[dict] = []
    for el in (payload or {}).get("elements") or []:
        tags = el.get("tags") or {}
        # марины и причалы — не море: `seamark:type=harbour` висит на каждой стоянке, и в Сиднее
        # их 40 штук. Нужны заливы, лагуны, гавани и море по тегу, а не лодочные стоянки.
        if tags.get("leisure") in {"marina", "slipway"}:
            continue
        # лагуна в старых фикстурах и кеше — это чаще пруд в заповеднике, чем море
        if tags.get("water") == "lagoon" and tags.get("natural") != "bay":
            continue
        if tags.get("seamark:type") == "harbour" and not (
                tags.get("natural") == "bay" or tags.get("water") in {"bay", "harbour", "strait"}
                or tags.get("place") in {"sea", "ocean"}):
            continue
        center = el.get("center") or {}
        distance = None
        if center.get("lat") is not None and center.get("lon") is not None:
            dlat = (float(center["lat"]) - lat) * 111.32
            dlon = (float(center["lon"]) - lon) * 111.32 * max(math.cos(math.radians(lat)), 1e-6)
            distance = round(math.hypot(dlat, dlon) * 1000.0, 1)
        if distance is not None and distance > radius_km * 1000.0:
            continue
        item = {"kind": "sea", "name": tags.get("name"), "nearest_m": distance,
                "area_m2": None, "tags": tags}
        kind = _polygon_kind(item)
        out.append({"kind": kind or "ocean", "name": tags.get("name"),
                    "distance_m": distance,
                    "tags": tags})
    seen: set[str] = set()
    unique: list[dict] = []
    for item in sorted(out, key=lambda i: i.get("distance_m") or 1e12):
        key = (item.get("name") or "") + "|" + (item.get("kind") or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def conflict_warning(sea: dict) -> str | None:
    """Текст предупреждения `sea_conflict` для quality.warnings."""
    if sea.get("present") is not None:
        return None
    satellite = next((e for e in sea.get("evidence") or [] if e["kind"] == "satellite"), None)
    if not satellite:
        return None
    return ("sea_conflict: OSM не находит моря в кольце, снимок даёт %.1f %% воды "
            "(порог %.0f %%) — море записано как «не знаю», а не «нет»"
            % (satellite["share"] * 100, satellite["threshold"] * 100))


__all__ = ["ALGO_VERSION", "DEPRECATED_NEAR_M", "NEAR_KM", "POLYGON_KIND_RU", "SATELLITE_MIN",
           "SEA_KINDS", "WIDE_KM", "conflict_warning", "detect", "fetch_payload",
           "payload_polygons"]
