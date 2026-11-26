"""Вода: реки, каналы, водоёмы из OSM, изгибы, слияния, водные врата, видимость.

Источник — OpenStreetMap через Overpass (ODbL). Модуль измеряет, но не оценивает:
он не решает, «нефритовый пояс» это или «обратный лук», а сообщает, с какой стороны
от русла лежит участок и каков радиус кривизны.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geo, hydronym, tides, worldnames

CACHE_DIR = Path(os.environ.get("GEOSITE_CACHE", Path.home() / ".cache" / "vector-geo-site")) / "osm"
ALGO_VERSION = "8.5.0"

USER_AGENT = "vector-geo-site/0.1 (fengshui landform measurement; https://github.com/Osmosy/vector-geo-site)"

# Зеркала Overpass: первое доступное. Порядок не случайный: mail.ru держит полную планету и
# отвечает, когда overpass-api.de в очереди; overpass.osm.ch — только Швейцария, поэтому на
# запросы о Москве он честно возвращает пусто, и пустой ответ нельзя принимать за ответ.
MIRRORS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

WATERWAY_KINDS = {"river", "stream", "canal", "drain", "ditch"}
LOOP_MIN_RATIO = 3.0        # W2: петля — отношение длины русла к разрыву не меньше этого
LOOP_MIN_TURN_DEG = 180.0   # R4: и русло должно завернуться не меньше чем на пол-оборота
LOOP_MIN_AREA_K = 0.05      # R4: площадь кольца в долях gap²


def loop_confidence(ratio: float, turn_deg: float, area_k: float) -> tuple[dict, str]:
    """Насколько петля превышает каждый порог и какова её уверенность (E-021).

    `margin` — превышение в долях собственного порога: 0.0 значит «ровно на пороге».
    Уверенность берётся по худшему из трёх условий: меньше 10 % — low, до 30 % — medium,
    дальше high. Петля Темзы в Сити (`ratio` 3.05 при пороге 3) иначе выглядела так же
    твёрдо, как настоящая излучина с запасом вдвое.
    """
    margins = {
        "ratio": ratio / LOOP_MIN_RATIO - 1.0,
        "turn": turn_deg / LOOP_MIN_TURN_DEG - 1.0,
        "area": area_k / LOOP_MIN_AREA_K - 1.0,
    }
    worst = min(margins.values())
    confidence = "low" if worst < 0.10 else ("medium" if worst < 0.30 else "high")
    return {key: float(round(value, 3)) for key, value in margins.items()}, confidence


@dataclass
class Feature:
    osm_id: int
    osm_type: str
    name: str | None
    kind: str                      # river / stream / canal / lake / sea / dock / culvert / road
    tags: dict = field(default_factory=dict)
    coords: list[tuple[float, float]] = field(default_factory=list)   # (lon, lat)
    closed: bool = False
    # линия русла или контур водоёма: контуры берегов нельзя считать водотоками, иначе
    # пересечение двух берегов засчитывается как впадение (E-017)
    geom: str = "line"             # line / area
    # дыры мультиполигона: остров внутри озера, суша внутри речного разлива (E-022).
    # Площадь считается за вычетом дыр, иначе «озеро с островом» — это озеро плюс остров.
    holes: list[list[tuple[float, float]]] = field(default_factory=list)
    source: str = "osm_way"
    # G3: имена и сезонность переживают склейку — иначе у реки остаётся только `name`,
    # а `name:zh` и `int_name` теряются вместе с кусками, из которых она собрана
    names: list[str] = field(default_factory=list)
    seasonal: bool | None = None        # osm_way / osm_relation / hydrorivers
    # G3: остальные имена из тегов (name:en, name:zh, int_name, alt_name) — навыку нужны оба
    names: list[str] = field(default_factory=list)
    # G3: временная вода (intermittent / seasonal / wadi) — это факт о воде, а не оценка
    seasonal: bool | None = None
    # доля скрытого участка: у одиночного куска её считает склейка, у прямого вызова — None
    hidden_share: float | None = None


def _query(lat: float, lon: float, radius_m: int) -> str:
    r = int(radius_m)
    return (
        "[out:json][timeout:90];"
        f'(way(around:{r},{lat},{lon})["waterway"~"^(river|stream|canal|drain|ditch|riverbank)$"];'
        f'way(around:{r},{lat},{lon})["natural"="water"];'
        # Площадь воды: отношения-мультиполигоны (E-022). `type=multipolygon` не в фильтре —
        # у отношения вода описывается членами, и любой `natural=water` собирается целиком;
        # фильтр отбросил бы берега, помеченные как `type=waterway`.
        f'relation(around:{r},{lat},{lon})["natural"="water"];'
        f'relation(around:{r},{lat},{lon})["waterway"="riverbank"];'
        f'relation(around:{r},{lat},{lon})["landuse"~"^(reservoir|basin)$"];'
        f'way(around:{r},{lat},{lon})["waterway"="dam"];'
        # `out body geom`, а не `out geom`: у отношений `out geom` отдаёт члены без геометрии,
        # и мультиполигон приходил пустым (E-022)
        ");out body geom;"
    )


def _cache_path(lat: float, lon: float, radius_m: int, *, kind: str = "water") -> Path:
    key = hashlib.sha256(f"{lat:.4f},{lon:.4f},{radius_m}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"{kind}-{key}.json"


def _ask(query: str, *, retries: int = 3, timeout: int = 240) -> dict:
    """Один запрос по всем зеркалам с повторами.

    Пустой ответ принимается только после того, как его дали все зеркала: у osm.ch в базе
    только Швейцария, и «ноль элементов» про Москву — это пустая база, а не пустая вода.
    """
    last_error: Exception | None = None
    empty: dict | None = None
    for attempt in range(retries):
        for mirror in MIRRORS:
            try:
                req = urllib.request.Request(
                    mirror, data=urllib.parse.urlencode({"data": query}).encode(),
                    headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode())
                if "elements" not in payload:
                    last_error = RuntimeError(f"неожиданный ответ: {str(payload)[:120]}")
                    continue
                if payload["elements"]:
                    return payload
                empty = payload
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                    json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(1 + attempt * 2)
    if empty is not None:
        return empty
    raise RuntimeError(f"Overpass недоступен: {last_error}")


def fetch_raw(lat: float, lon: float, radius_m: int = 8000, *, allow_network: bool = True,
              retries: int = 3) -> dict:
    """Сырой ответ Overpass с кешем. Сеть трогается только если кеша нет.

    К воде вокруг участка добавляются куски рек из отношений OSM (см. `relation_ways`):
    одиночные ways в выгрузке обрываются, и устье притока вместе с ними теряется.
    """
    path = _cache_path(lat, lon, radius_m)
    if path.exists():
        base = json.loads(path.read_text())
    else:
        if not allow_network or os.environ.get("GEOSITE_OFFLINE") == "1":
            raise FileNotFoundError(f"нет кеша воды: {path}")
        base = _ask(_query(lat, lon, radius_m), retries=retries)
        # пустой ответ не кешируем: «ноль элементов» у зеркала с частичным покрытием не факт
        # о местности, а факт о зеркале — иначе участок навсегда остаётся без воды
        if base.get("elements"):
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(base))

    extra = relation_ways(lat, lon, radius_m, allow_network=allow_network, retries=retries)
    if not extra:
        return base
    seen = {(el.get("type"), el.get("id")) for el in base.get("elements", [])}
    elements = list(base.get("elements", []))
    for el in extra:
        key = (el.get("type"), el.get("id"))
        if key in seen:
            continue
        seen.add(key)
        elements.append(el)
    return {"version": base.get("version", 0.6), "generator": base.get("generator", ""),
            "osm3s": base.get("osm3s", {}), "elements": elements}


COAST_MAX_M = 1000.0        # ближе километра к берегу вода приливная, дальше — не решаем
TILE_SIDE = 2               # плотный город: запрос плитками 2×2


def _query_bbox(south: float, west: float, north: float, east: float) -> str:
    """Тот же набор слоёв, но в прямоугольнике: так режется плотный город."""
    box = f"{south:.5f},{west:.5f},{north:.5f},{east:.5f}"
    return (
        "[out:json][timeout:180];"
        f'(way["waterway"~"^(river|stream|canal|drain|ditch|riverbank)$"]({box});'
        f'way["natural"="water"]({box});'
        f'relation["natural"="water"]({box});'
        f'relation["waterway"="riverbank"]({box});'
        f'relation["landuse"~"^(reservoir|basin)$"]({box});'
        f'way["waterway"="dam"]({box});'
        ");out body geom;"
    )


def _query_coast(lat: float, lon: float, radius_m: int) -> str:
    """Береговая линия в круге: нужна только как признак приливной воды (G3)."""
    r = int(radius_m)
    return ("[out:json][timeout:120];"
            f'way(around:{r},{lat},{lon})["natural"="coastline"];out body geom;')


def truncated(payload: dict | None) -> bool:
    """Ответ Overpass обрезан: сервер пишет об этом в `remark` (таймаут, память, лимит)."""
    if not payload:
        return False
    remark = str(payload.get("remark") or "").lower()
    return bool(remark)


def merge_payloads(parts: list[dict]) -> dict:
    """Склейка плиток: элементы по (type, id), без повторов."""
    seen = set()
    elements = []
    for part in parts:
        for el in part.get("elements") or []:
            key = (el.get("type"), el.get("id"))
            if key in seen:
                continue
            seen.add(key)
            elements.append(el)
    out = {"version": 0.6, "generator": "merge", "elements": elements}
    if any(truncated(part) for part in parts):
        out["remark"] = "часть плиток пришла обрезанной"
    return out


def _tile_boxes(lat: float, lon: float, radius_m: int) -> list[tuple[float, float, float, float]]:
    """Круг делится на плитки по bbox: у плотного города один `around` упирается в лимит."""
    from .dem import bbox_for_radius

    lat_min, lat_max, lon_min, lon_max = bbox_for_radius(lat, lon, radius_m / 1000.0)
    lat_step = (lat_max - lat_min) / TILE_SIDE
    lon_step = (lon_max - lon_min) / TILE_SIDE
    boxes = []
    for i in range(TILE_SIDE):
        for j in range(TILE_SIDE):
            boxes.append((lat_min + i * lat_step, lon_min + j * lon_step,
                          lat_min + (i + 1) * lat_step, lon_min + (j + 1) * lon_step))
    return boxes


def fetch_payload(lat: float, lon: float, radius_m: int = 8000, *, allow_network: bool = True,
                  retries: int = 3, dense: bool | None = None) -> dict:
    """Вода вокруг участка: один запрос, а у плотного города — плитками (G3.5).

    `dense=None` — решает сервер: если ответ пришёл с `remark` (таймаут, память), запрос
    повторяется плитками 2×2, и в пакете поднимается `partial`.
    """
    base = fetch_raw(lat, lon, radius_m, allow_network=allow_network, retries=retries)
    if dense is None and not truncated(base):
        return base
    if dense is False:
        return base
    parts = []
    for south, west, north, east in _tile_boxes(lat, lon, radius_m):
        query = _query_bbox(south, west, north, east)
        try:
            parts.append(_ask(query, retries=retries))
        except RuntimeError:
            parts.append({"elements": [], "remark": "плитка недоступна"})
    merged = merge_payloads(parts)
    merged["tiled"] = True
    # в плиточной склейке остаются и отношения — но их члены берём отдельно, как и раньше
    extra = relation_ways(lat, lon, radius_m, allow_network=allow_network, retries=retries)
    if extra:
        seen = {(el.get("type"), el.get("id")) for el in merged["elements"]}
        for el in extra:
            key = (el.get("type"), el.get("id"))
            if key not in seen:
                seen.add(key)
                merged["elements"].append(el)
    return merged


def fetch_coast(lat: float, lon: float, radius_m: int = 8000, *, allow_network: bool = True,
                retries: int = 2) -> dict | None:
    """Береговая линия с кешем; `None` — не смотрели (офлайн без кеша)."""
    path = _cache_path(lat, lon, radius_m, kind="coast")
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    if not allow_network or os.environ.get("GEOSITE_OFFLINE") == "1":
        return None
    try:
        payload = _ask(_query_coast(lat, lon, radius_m), retries=retries)
    except RuntimeError:
        return None
    if payload.get("elements"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return payload


def coast_lines(payload: dict | None) -> list[list[tuple[float, float]]]:
    """Линии берега в градусах (lon, lat)."""
    out: list[list[tuple[float, float]]] = []
    for el in (payload or {}).get("elements") or []:
        geom = el.get("geometry") or []
        line = [(float(p["lon"]), float(p["lat"])) for p in geom if p.get("lon") is not None]
        if len(line) >= 2:
            out.append(line)
    return out


def nearest_coast_m(payload: dict | None, lat: float, lon: float) -> float | None:
    """Ближайшее расстояние до береговой линии в метрах; `None` — берега не видели.

    Нужно не для воды, а для пакета: у прибрежного участка море задано в OSM береговой линией,
    а снимок считает его водой (E-028). Без этого числа 42 % моря в Рио попадают в расхождение
    «устаревший OSM» и в замену OSM снимком.
    """
    coast = coast_lines(payload)
    if not coast:
        return None
    return _coast_distance_m(coast, [(lon, lat)])


def _coast_distance_m(coast: list[list[tuple[float, float]]],
                      points: list[tuple[float, float]]) -> float | None:
    """Ближайшее расстояние от воды до берега: точки в градусах (lon, lat)."""
    best = None
    for line in coast:
        for lon, lat in points:
            for clon, clat in line[:: max(1, len(line) // 40)]:
                dist = geo.haversine_m(lat, lon, clat, clon)
                if best is None or dist < best:
                    best = dist
    return best


RELATION_KINDS = "^(river|stream|canal|riverbank)$"
# E-027: у площади воды члены отношения помечены `outer` / `inner`; без них мультиполигон
# (Нил, озеро, залив) не собирался вовсе, даже когда геометрия приходила
RELATION_ROLES = {"", "main_stream", "side_stream", "waterway", "outer", "inner", "part"}


def _relation_query(lat: float, lon: float, radius_m: int) -> str:
    """Отношения с водой в круге — по площади, а не по имени (E-027).

    Раньше здесь были только `waterway=river|stream|canal` **с именем**. В мире это неверно:
    Нил в Каире — анонимное отношение `natural=water` + `water=river`, залив Виктории в Гонконге
    и бухта Гуанабара в Рио заданы береговой линией и частью полигонами без имён. Проверка живьём
    дала долю воды по OSM 0.0 % в Каире при 11 % по снимку — и причина была не в разметке,
    а в запросе.

    Теперь три вида отношений (площадь воды, старый тег русла, водохранилище) без требования имени.
    """
    r = int(radius_m)
    around = f"around:{r},{lat},{lon}"
    # E-027: три условия объединяются скобками. Без объединения Overpass выполняет их
    # последовательно, и `out` отдаёт только последний набор: у Каира последнее условие
    # (водохранилища) пусто, и ответ приходил пустым при десяти отношениях воды вокруг.
    return ("[out:json][timeout:180];("
            f'rel({around})["natural"="water"];'
            f'rel({around})["waterway"~"{RELATION_KINDS}"];'
            f'rel({around})["landuse"~"^(reservoir|basin)$"];'
            ");out body;")


def _ways_query(ids: list[int]) -> str:
    return "[out:json][timeout:180];way(id:" + ",".join(str(i) for i in ids) + ");out tags geom;"


def _relations_ways_query(rel_ids: list[int]) -> str:
    """Члены нескольких отношений одним запросом: восемь отношений — это восемь запросов,
    а зеркало в очереди отвечает минуты; суммарный объём от этого не меняется."""
    ids = ",".join(str(int(i)) for i in rel_ids)
    return f"[out:json][timeout:240];rel(id:{ids})->.r;way(r.r);out tags geom;"


def _near_site(coords: list[dict], lat: float, lon: float, limit_m: float) -> bool:
    """Есть ли у куска geometry точка ближе limit_m к участку."""
    for g in coords:
        if g and geo.haversine_m(lat, lon, g["lat"], g["lon"]) <= limit_m:
            return True
    return False


def relation_ways(lat: float, lon: float, radius_m: int = 8000, *, allow_network: bool = True,
                  retries: int = 3, max_relations: int = 24, max_ways: int = 600,
                  batch: int = 40, keep_factor: float = 2.5) -> list[dict]:
    """Геометрия отношений OSM, собранная обратно в отношения.

    Отношение описывает объект целиком: река (`waterway=river`) стыкуется узлами, площадь воды
    (`natural=water`) состоит из `outer` и `inner`. `out body` отдаёт членов без геометрии,
    поэтому геометрия тянется пачками по ways (`out geom` по отношению целиком — 0.5–3 МБ
    и 502 от перегруженного зеркала).

    E-027: куски собираются обратно в **отношение** с ролями `outer`/`inner`, а не отдаются
    как отдельные ways. Иначе мультиполигон терял дыры, а озеро из десяти незамкнутых кусков
    не давало площади вовсе. Члены дальше `keep_factor * radius` выбрасываются.
    """
    # ключ кеша с версией: запрос и форма ответа изменились (E-027), старый кеш отвечал бы
    # нулевой водой по Нилу, Гонконгу и Рио ещё месяц
    path = _cache_path(lat, lon, radius_m, kind="relations-v3")
    if path.exists():
        return json.loads(path.read_text()).get("elements", [])
    if not allow_network or os.environ.get("GEOSITE_OFFLINE") == "1":
        return []

    try:
        rel = _ask(_relation_query(lat, lon, radius_m), retries=retries)
        relations: list[tuple[int, list[int]]] = []        # (rel_id, его ways)
        rel_tags: dict[int, dict] = {}
        rel_role: dict[int, str] = {}                      # way -> outer / inner / ""
        rel_of: dict[int, int] = {}
        for el in rel.get("elements", [])[:max_relations]:
            # E-027: имя не обязательно. Нил в Каире — анонимное отношение `natural=water`;
            # требование имени и было причиной нулевой доли воды при 11 % по снимку.
            tags = dict(el.get("tags") or {})
            rel_id = int(el.get("id") or 0)
            ways: list[int] = []
            for member in el.get("members", []):
                if member.get("type") != "way" or (member.get("role") or "") not in RELATION_ROLES:
                    continue
                ref = int(member.get("ref") or 0)
                if not ref or ref in rel_of:
                    continue
                rel_of[ref] = rel_id
                rel_role[ref] = member.get("role") or ""
                ways.append(ref)
                if len(ways) >= max_ways:
                    break
            if ways:
                rel_tags[rel_id] = tags
                relations.append((rel_id, ways))
        if not relations:
            return []

        keep_m = keep_factor * float(radius_m)
        geom_of: dict[int, list[dict]] = {}
        try:
            payload = _ask(_relations_ways_query([rel_id for rel_id, _ in relations]), retries=retries)
            for el in payload.get("elements", []):
                geom_of[int(el.get("id") or 0)] = el.get("geometry") or []
        except Exception:
            for rel_id, ways in relations:
                for start in range(0, len(ways), batch):
                    for el in _ask(_ways_query(ways[start:start + batch]),
                                   retries=retries).get("elements", []):
                        geom_of[int(el.get("id") or 0)] = el.get("geometry") or []

        out: list[dict] = []
        for rel_id, ways in relations:
            members: list[dict] = []
            for ref in ways:
                geometry = geom_of.get(ref) or []
                if len(geometry) < 2 or not _near_site(geometry, lat, lon, keep_m):
                    continue
                members.append({"type": "way", "ref": ref, "role": rel_role.get(ref, ""),
                                "geometry": geometry})
            if members:
                out.append({"type": "relation", "id": rel_id, "tags": rel_tags.get(rel_id, {}),
                            "members": members})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"elements": out}))
        return out
    except Exception:
        # без отношений вода всё равно измеряется, просто площадь и устье могут не найтись
        return []


def is_hidden(tags: dict) -> bool:
    """Вода под землёй: culvert, covered, underground — для «скрытой воды» канона.

    Единственный источник правила — `hidden_tags` в `data/name_aliases.json`: два
    независимых списка расходились на `covered=culvert` и `tunnel=yes`, и одна и та же
    линия считалась скрытой для вида, но не для доли (E-013).
    """
    return hydronym.is_hidden(tags)


def _kind_of(tags: dict) -> str:
    natural = tags.get("natural")
    waterway = tags.get("waterway")
    if (natural == "water" or waterway == "riverbank"
            or (tags.get("landuse") or "") in AREA_LANDUSE):
        # Вид описывает поверхность воды, а не русло: у контура это lake / dock / sea (E-018).
        # До 7.0.0 `water=river|canal` у контура давал `river`, и потребитель пакета принимал
        # берег за реку — с входом и выходом воды, изгибом и петлёй.
        water = (tags.get("water") or "").lower()
        landuse = (tags.get("landuse") or "").lower()
        text = " ".join(str(tags.get(k) or "") for k in ("name", "name:en", "alt_name")).lower()
        if water == "sea" or tags.get("place") == "sea":
            return "sea"
        if water in {"basin", "dock"} or landuse in {"basin", "dock"} or "basin" in text or "dock" in text:
            return "dock"
        return "lake"
    if waterway == "river":
        return "river"
    if waterway == "canal":
        return "canal"
    if waterway in {"stream", "drain", "ditch"}:
        return "stream"
    if waterway == "dam":
        return "dam"
    return "other"


AREA_LANDUSE = {"reservoir", "basin"}


def _geom_of(tags: dict, closed: bool) -> str:
    """Контур водоёма или линия русла.

    Площадью считается: замкнутый `natural=water`, `waterway=riverbank`, `landuse=reservoir|basin`,
    а также любой замкнутый `waterway=*` без явного `area=no` — в OSM так иногда рисуют берега.
    Всё остальное — линия. Разделение нужно, потому что пересечение двух берегов не слияние:
    у Стратфорда контур канала и контур реки «впадали» друг в друга (E-017).
    """
    explicit = (tags.get("area") or "").lower()
    if explicit == "no":
        return "line"
    if tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
        return "area"
    if (tags.get("landuse") or "") in AREA_LANDUSE:
        return "area"
    if not closed:
        return "line"
    return "area" if tags.get("waterway") else "line"


def _name_of(tags: dict) -> str | None:
    """Имя водотока: основное, иначе английское/китайское/русское, иначе альтернативное.

    G3: имя любой письменности. Нормализация NFKC и разбор списка — в `worldnames`.
    """
    return worldnames.primary_name(tags)


def _names_of(tags: dict) -> list[str]:
    return worldnames.names_of(tags)


def _kind_with_name(tags: dict, name: str | None) -> str:
    """Вид воды по тегам, а если тег молчит — по родовому слову имени (G3)."""
    kind = _kind_of(tags)
    if kind != "other":
        return kind
    guess = worldnames.kind_from_name(name)
    if not guess:
        return "other"
    # в пакете свой короткий список видов: пруд и болото — стоячая вода, бухта и порт — море
    return {"pond": "lake", "wetland": "lake", "harbour": "sea", "bay": "sea"}.get(
        guess["kind"], guess["kind"])


CLIP_MARGIN = 1.05          # обрезаем по кругу с запасом: клин кольца начинается внутри


def _clip_circle(lat: float, lon: float, radius_m: float, *, steps: int = 120):
    """Круг выборки в градусах — для обрезки мультиполигона до пересечений.

    Точность обрезки на 8 км — сантиметры по дуге и единицы метров по радиусу: круг нужен
    только чтобы не тащить в расчёт берег, уходящий на десятки километров. Площади секторов
    считаются потом в UTM, поэтому запас `CLIP_MARGIN` важнее точности самого круга.
    """
    from shapely.geometry import Polygon

    m_per_deg_lat = 111132.0
    m_per_deg_lon = m_per_deg_lat * max(0.05, math.cos(math.radians(lat)))
    points = []
    for index in range(steps):
        azimuth = 2.0 * math.pi * index / steps
        points.append((lon + radius_m * math.sin(azimuth) / m_per_deg_lon,
                       lat + radius_m * math.cos(azimuth) / m_per_deg_lat))
    return Polygon(points)


def _member_segments(el: dict) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """Члены отношения по ролям: внешние кольца и дыры, отдельными кусками."""
    outer: list[list[tuple[float, float]]] = []
    inner: list[list[tuple[float, float]]] = []
    for member in el.get("members") or []:
        if member.get("type") != "way":
            continue
        coords = [(g["lon"], g["lat"]) for g in member.get("geometry") or [] if g]
        if len(coords) < 2:
            continue
        if (member.get("role") or "").lower() == "inner":
            inner.append(coords)
        else:
            outer.append(coords)
    return outer, inner


def _assemble_relation(el: dict, *, lat: float, lon: float, radius_m: float,
                       limit: int = 8) -> list[Feature]:
    """Мультиполигон OSM в площади: внешние кольца склеиваются, дыры вычитаются (E-022).

    Отношение — это не набор линий: `outer` идёт цепочкой кусков и замыкается только вместе.
    До 8.0.0 каждый член становился отдельным объектом с тегами отношения, поэтому берег
    Темзы приходил десятком незамкнутых обрывков, а поверхность реки в долю воды не попадала.

    Склейка через `polygonize`: он собирает кольца по узлам, а не по порядку членов, поэтому
    куски, отданные в другом порядке или в обратном направлении, всё равно замыкаются.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union

    outer, inner = _member_segments(el)
    if not outer:
        return []
    try:
        faces = [poly for poly in polygonize(unary_union([LineString(s) for s in outer]))
                 if poly.area > 0.0]
        if not faces:
            return []
        holes = ([poly for poly in polygonize(unary_union([LineString(s) for s in inner]))
                  if poly.area > 0.0] if inner else [])
        holes_union = unary_union(holes) if holes else None
        circle = _clip_circle(lat, lon, radius_m * CLIP_MARGIN)
    except Exception:
        return []
    tags = el.get("tags") or {}
    name = _name_of(tags)
    out: list[Feature] = []
    for face in faces:
        piece = face if holes_union is None else face.difference(holes_union)
        if not piece.is_valid:
            piece = piece.buffer(0)
        if piece.is_empty:
            continue
        try:
            piece = piece.intersection(circle)     # обрезаем до пересечений с кольцами
        except Exception:
            continue
        parts = list(piece.geoms) if piece.geom_type == "MultiPolygon" else [piece]
        for poly in parts:
            if poly.is_empty or poly.geom_type != "Polygon" or poly.area <= 0.0:
                continue
            ring = [(float(x), float(y)) for x, y in poly.exterior.coords]
            if len(ring) < 4:
                continue
            out.append(Feature(int(el.get("id") or 0) * 1000 + len(out), "relation", name,
                               _kind_with_name(tags, name), tags, ring, True, "area",
                               names=_names_of(tags),
                               seasonal=worldnames.is_seasonal(tags),
                               holes=[[(float(x), float(y)) for x, y in hole.coords]
                                      for hole in poly.interiors],
                               source="osm_relation"))
            if len(out) >= limit:
                return out
    return out


def parse_features(payload: dict, *, lat: float | None = None, lon: float | None = None,
                   radius_m: float = 8000.0) -> list[Feature]:
    out: list[Feature] = []
    for el in payload.get("elements", []):
        tags = el.get("tags", {}) or {}
        if el.get("type") == "way" and el.get("geometry"):
            coords = [(g["lon"], g["lat"]) for g in el["geometry"] if g]
            kind = _kind_of(tags)
            if not coords or kind == "other":
                continue
            closed = len(coords) > 3 and abs(coords[0][0] - coords[-1][0]) < 1e-9 and abs(coords[0][1] - coords[-1][1]) < 1e-9
            out.append(Feature(el.get("id", 0), "way", _name_of(tags), kind, tags, coords, closed,
                               _geom_of(tags, closed), names=_names_of(tags),
                               seasonal=worldnames.is_seasonal(tags)))
        elif el.get("type") == "relation" and el.get("members"):
            # E-022: площадь отношения собирается целиком (outer + inner), а не по членам:
            # берег Темзы приходит десятком незамкнутых кусков и без сборки не даёт площади.
            # Обрезка по кругу — здесь, до пересечений с секторами.
            if lat is not None and lon is not None:
                out.extend(_assemble_relation(el, lat=lat, lon=lon, radius_m=radius_m))
                continue
            # без участка (старые вызовы, тесты) — прежнее поведение по членам
            for idx, member in enumerate(el["members"]):
                geom = [(g["lon"], g["lat"]) for g in member.get("geometry") or [] if g]
                if len(geom) >= 2:
                    closed = (len(geom) > 3
                              and abs(geom[0][0] - geom[-1][0]) < 1e-9
                              and abs(geom[0][1] - geom[-1][1]) < 1e-9)
                    # член отношения с `natural=water` — часть контура водоёма, площадью он
                    # становится по тому же правилу, что и отдельный замкнутый way
                    name_r = _name_of(tags)
                    out.append(Feature(el.get("id", 0) * 1000 + idx, "relation", name_r,
                                       _kind_with_name(tags, name_r), tags, geom, closed,
                                       _geom_of(tags, closed), names=_names_of(tags),
                                       seasonal=worldnames.is_seasonal(tags)))
    return out


# ---------- геометрия ----------

def to_utm(coords: list[tuple[float, float]], crs: int | str) -> np.ndarray:
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(crs), always_xy=True)
    xs, ys = tr.transform([c[0] for c in coords], [c[1] for c in coords])
    return np.column_stack([xs, ys])


def _segment_distances(pts: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Расстояния от точки p до каждого отрезка ломаной."""
    a, b = pts[:-1], pts[1:]
    ab = b - a
    ap = p - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom == 0] = 1e-9
    t = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return np.hypot(proj[:, 0] - p[0], proj[:, 1] - p[1])


def _bend_center(pts: np.ndarray, i: int, *, half_window_m: float = 30.0) -> np.ndarray | None:
    """Центр кривизны в точке `i`: берём точки на равном удалении по дуге в обе стороны.

    Симметричное окно делает результат независимым от направления обхода линии: три точки
    те же, только переставлены, а центр описанной окружности от порядка не зависит.
    """
    if len(pts) < 3:
        return None
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s0 = cum[min(i, len(cum) - 1)]
    total = cum[-1]
    hw = min(half_window_m, total / 3.0)   # короткая линия: окно сужается вместе с ней
    if hw < 1.0:
        return _circumcenter(pts[i - 1], pts[i], pts[i + 1])
    left_s, right_s = max(0.0, s0 - hw), min(total, s0 + hw)
    if right_s - left_s < 1.0:
        return None
    left = np.array([np.interp(left_s, cum, pts[:, 0]), np.interp(left_s, cum, pts[:, 1])])
    right = np.array([np.interp(right_s, cum, pts[:, 0]), np.interp(right_s, cum, pts[:, 1])])
    return _circumcenter(left, pts[i], right)


def _circumcenter(a, b, c) -> np.ndarray | None:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    return np.array([ux, uy])


def _crossings_with_circle(pts: np.ndarray, radius: float) -> list[np.ndarray]:
    """Точки пересечения ломаной с окружностью радиуса radius вокруг начала координат."""
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        da, db = np.hypot(*a), np.hypot(*b)
        if (da - radius) * (db - radius) > 0:
            continue
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            m = a + (b - a) * mid
            if (np.hypot(*m) - radius) * (da - radius) > 0:
                lo = mid
            else:
                hi = mid
        out.append(a + (b - a) * ((lo + hi) / 2))
    return out


def clip_runs_near(pts: np.ndarray, limit_m: float) -> list[tuple[int, int]]:
    """Участки ломаной внутри круга как диапазоны индексов исходных точек.

    Возвращает пары `(i0, i1)` включительно, включая по одной точке снаружи с каждой стороны:
    так кусок доходит до границы окна и не рвётся раньше срока. Сами точки не пересчитываются —
    узлы остаются теми же, что пришли из OSM.
    """
    if len(pts) < 2:
        return []
    d = np.hypot(pts[:, 0], pts[:, 1])
    if float(d.max()) <= limit_m:
        return [(0, len(pts) - 1)]
    inside = d <= limit_m
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(inside):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 2:
                runs.append((max(0, start - 1), min(len(pts) - 1, i)))
            start = None
    if start is not None and len(pts) - start >= 2:
        runs.append((max(0, start - 1), len(pts) - 1))
    return runs


def clip_near(pts: np.ndarray, limit_m: float) -> list[np.ndarray]:
    """Куски ломаной (в метрах от участка), попадающие в круг limit_m.

    Точки вне круга выбрасываются, но граница сохраняется точкой пересечения: без неё линия
    обрывается на последнем узле внутри круга, и «где вода пересекает радиус» уезжает на
    длину шага. Метрики участка живут в 8 км, поэтому окно в 12 км их не меняет, а склейка
    Волги на 3000 км перестаёт стоить секунд в переборе слияний.
    """
    if len(pts) < 2:
        return []
    d = np.hypot(pts[:, 0], pts[:, 1])
    if float(d.max()) <= limit_m:
        return [pts]
    pieces: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        da, db = float(d[i]), float(d[i + 1])
        inside_a, inside_b = da <= limit_m, db <= limit_m
        if inside_a and not cur:
            cur.append(a)
        if inside_a != inside_b:
            t = (limit_m - da) / (db - da)
            edge = a + (b - a) * t
            if inside_a:
                cur.append(edge)
                if len(cur) >= 2:
                    pieces.append(np.vstack(cur))
                cur = []
            else:
                cur = [edge]
        if inside_b:
            cur.append(b)
    if len(cur) >= 2:
        pieces.append(np.vstack(cur))
    return pieces


@dataclass
class MergedLine:
    key: str
    name: str | None
    kind: str
    coords: list[tuple[float, float]]
    ways: int
    hidden: bool
    hidden_share: float | None = None
    uid: str = ""
    # теги, одинаковые у всех кусков: по ним решается приливность (тег `tidal`) и порядок
    # отрисовки. У склейки своих тегов не было, и приливная Темза получала flow_rule 5 —
    # «рельеф молчит», хотя тег `tidal=yes` есть у всех её кусков.
    tags: dict = field(default_factory=dict)
    geom: str = "line"             # line / area
    # дыры мультиполигона (остров в озере) и источник: у площади из отношения площадь
    # считается за вычетом дыр, и в пакете видно `source: osm_relation` (E-022)
    holes: list[list[tuple[float, float]]] = field(default_factory=list)
    source: str = "osm_way"
    # G3: имена и сезонность переживают склейку — иначе у реки остаётся только `name`,
    # а `name:zh` и `int_name` теряются вместе с кусками, из которых она собрана
    names: list[str] = field(default_factory=list)
    seasonal: bool | None = None

    def __post_init__(self) -> None:
        if not self.uid:
            self.uid = self.key


# одна река бывает размечена и river, и stream: тип куска берётся по старшему, а не по самому
# длинному way — иначе один старый кусок waterway=stream переименовывает Волгу в ручей
KIND_RANK = {"river": 5, "canal": 4, "stream": 3, "lake": 2, "dock": 1, "sea": 0}


def _union_names(items: list[Feature]) -> list[str]:
    """Имена всех кусков без повторов: `name:zh` бывает только у одного из десяти ways."""
    out: list[str] = []
    for item in items:
        for value in getattr(item, "names", []) or []:
            if value not in out:
                out.append(value)
    return out


def _any_seasonal(items: list[Feature]) -> bool | None:
    """Сезонность — свойство реки: если хоть один кусок помечен временным, вода временная."""
    flags = [getattr(item, "seasonal", None) for item in items]
    if any(flag is True for flag in flags):
        return True
    return False if any(flag is False for flag in flags) else None


def merge_by_name(features: list[Feature], *, max_gap_m: float = 150.0,
                  near: tuple[float, float] | None = None) -> list[MergedLine]:
    """Склеить отрезки одной реки в одну линию: OSM хранит реку десятками ways.

    Подряд идущие куски сшиваются по ближайшим концам, если разрыв не больше max_gap_m;
    иначе линия остаётся отдельной. Анонимные куски (без имени) не склеиваются вовсе.

    `near` — координаты участка: теги берутся у куска, ближайшего к нему (приливность и
    порядок отрисовки — свойство места, а не всей реки). Без `near` остаются только теги,
    одинаковые у всех кусков.
    """
    groups: dict[tuple[str, str], list[Feature]] = {}
    for f in features:
        if f.kind not in {"river", "stream", "canal", "lake", "sea", "dock"}:
            continue
        # ключ склейки — нормализованное имя: «River Fleet» и «The Fleet» — один водоток
        # тип в ключ не входит: одна река в OSM бывает и river, и stream на разных участках
        # контур водоёма и русло с тем же именем — разные объекты: склеенные в одну линию,
        # они дают скачки геометрии и ложные слияния берегов (E-017)
        # у площади из отношения свой osm_id на каждый кусок берега: `rel…` в ключе
        # отличает его от way-контура с тем же числом (E-022)
        fallback = (f"rel{f.osm_id}" if getattr(f, "source", "") == "osm_relation"
                    else f"#{f.osm_id}")
        key = (hydronym.normalize_name(f.name) or fallback, f.geom)
        groups.setdefault(key, []).append(f)

    merged: list[MergedLine] = []
    for (name, geom), items in groups.items():
        longest = max(items, key=lambda f: len(f.coords))
        # `#<id>` — ключ для безымянных, а не название: иначе объект зовётся «#28934005»
        # и привязка берега к реке по имени не находит ничего (E-022)
        display = longest.name
        kind = max((f.kind for f in items), key=lambda k: KIND_RANK.get(k, 0))
        pieces = [f.coords for f in items]
        hidden = all(is_hidden(f.tags) for f in items)
        # теги склейки: у склейки своих тегов не было, и приливная Темза получала rule 5 —
        # «рельеф молчит», хотя тег `tidal=yes` есть у кусков у самого участка. Берём теги
        # ближайшего к участку куска; без участка — только то, что совпадает у всех кусков
        # (иначе один кусок в трубе объявил бы трубой всю реку).
        if near is not None:
            lat0, lon0 = near
            nearest_item = min(
                items,
                key=lambda f: min(geo.haversine_m(lat0, lon0, c[1], c[0]) for c in f.coords))
            common = dict(nearest_item.tags)
        else:
            common = {k: v for k, v in items[0].tags.items()
                      if all(f.tags.get(k) == v for f in items[1:])}
        total_len = sum(_line_len(f.coords) for f in items)
        hidden_len = sum(_line_len(f.coords) for f in items if hydronym.is_hidden(f.tags))
        hidden_share = None if total_len <= 0 else round(hidden_len / total_len, 2)
        # Склейка по связности, а не «тянем хвост»: река в OSM растёт в обе стороны от самого
        # длинного куска, и односторонняя жадность рвала её на 13 линий (E-014). Сначала
        # компоненты связности по концам (|разрыв| ≤ max_gap_m), затем внутри компоненты —
        # пришиваем куски то к началу, то к концу, пока есть что пришивать.
        if geom == "area":
            # Контуры не сшиваются: склейка колец даёт хорды между ними, и площадь
            # «озёр» одного имени считалась бы вместе с сушей между ними (E-022).
            for item in items:
                item_hidden = is_hidden(item.tags)
                length = _line_len(item.coords)
                merged.append(MergedLine(f"osm-{name}-{kind}", item.name or display, kind,
                                         list(item.coords), 1, item_hidden,
                                         (1.0 if item_hidden else 0.0) if length > 0 else None,
                                         tags=dict(item.tags), geom="area",
                                         holes=[list(h) for h in item.holes],
                                         source=item.source,
                                         names=list(item.names or []),
                                         seasonal=item.seasonal))
            continue
        for comp in _components(pieces, max_gap_m):
            items_count = len(comp)
            for chain, forward, backward in _chain_component(pieces, max_gap_m, idx=comp,
                                                             with_orientation=True):
                # Направление линии — как в OSM у большинства её метров: иначе склейка могла
                # развернуть реку, и «вход/выход» с приливным правилом смотрели в море наоборот.
                if backward > forward:
                    chain = chain[::-1]
                merged.append(MergedLine(f"osm-{name}-{kind}", display, kind, chain,
                                         items_count, hidden, hidden_share,
                                         tags=dict(common), geom=items[0].geom,
                                         names=_union_names(items),
                                         seasonal=_any_seasonal(items)))
    # ключ куска неуникален: у одной реки их несколько, и словарь слияний молча терял все,
    # кроме последнего. uid различает куски, key остаётся именем реки для потребителя.
    for idx, line in enumerate(merged):
        line.uid = f"{line.key}#{idx}"
    return merged


def _components(pieces: list[list[tuple[float, float]]], max_gap_m: float) -> list[list[int]]:
    """Группы кусков, связанных концами: транзитивно, через союз-файнд.

    Возвращает индексы кусков. Порог тот же, что у склейки: два куска в одной группе, если
    хоть одна пара их концов ближе `max_gap_m`.
    """
    n = len(pieces)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            d = min(_haversine_deg(a, b) for a in (pieces[i][0], pieces[i][-1])
                    for b in (pieces[j][0], pieces[j][-1]))
            if d <= max_gap_m:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _chain_component(pieces: list[list[tuple[float, float]]], max_gap_m: float,
                     idx: list[int] | None = None,
                     with_orientation: bool = False):
    """Сшить куски компоненты в линии: главный ствол — самый длинный путь по графу кусков.

    Река в OSM — не ломаная, а граф: у островов два русла, притоки впадают в середину, а в
    центре Лондона Темза нарезана десятками коротких ways, концы которых лежат в 10–100 м друг
    от друга. Жадная пришивка «к концу цепочки» рвала её на 13 линий (E-014).

    Кусок — ребро, его концы — узлы. Узел один, если координаты совпадают (допуск 1 м): общий
    узел OSM даёт ровно 0, а соседние, но разные вершины плотной нарезки сливать нельзя —
    иначе весь центр города становится одним узлом, и путь через него проходится один раз.
    Шаг из узла возможен в любой неиспользованный кусок, конец которого ближе `max_gap_m`:
    так сшиваются разрывы до порога. Главный ствол — самый длинный путь без повтора узлов;
    остальные куски (второе русло острова, старица) отдаются отдельными линиями.
    """
    idx = list(range(len(pieces))) if idx is None else list(idx)
    if len(idx) == 1:
        only = [list(pieces[idx[0]])]
        return [(only[0], _line_len(pieces[idx[0]]), 0.0)] if with_orientation else only

    ends: list[tuple[float, float]] = []
    for i in idx:
        ends.extend((pieces[i][0], pieces[i][-1]))
    parent = list(range(len(ends)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            if find(a) != find(b) and _haversine_deg(ends[a], ends[b]) <= 1.0:
                parent[find(a)] = find(b)
    nid = [find(i) for i in range(len(ends))]
    xy: dict[int, tuple[float, float]] = {}
    for pos, i in enumerate(idx):
        xy.setdefault(nid[2 * pos], ends[2 * pos])
        xy.setdefault(nid[2 * pos + 1], ends[2 * pos + 1])
    na = {i: nid[2 * pos] for pos, i in enumerate(idx)}
    nb = {i: nid[2 * pos + 1] for pos, i in enumerate(idx)}
    length = {i: _line_len(pieces[i]) for i in idx}

    def options(cur: int, used: set[int]) -> list[tuple[float, int, int, int]]:
        """(длина куска, кусок, узел входа, узел выхода) — шаги из текущего узла."""
        cx, cy = xy[cur]
        out = []
        for i in idx:
            if i in used:
                continue
            a, b = na[i], nb[i]
            da = 0.0 if a == cur else _haversine_deg((cx, cy), xy[a])
            db = 0.0 if b == cur else _haversine_deg((cx, cy), xy[b])
            if da <= db and da <= max_gap_m:
                out.append((length[i], i, a, b))
            elif db <= max_gap_m:
                out.append((length[i], i, b, a))
        return out

    used: set[int] = set()
    lines: list[list[tuple[float, float]]] = []
    while len(used) < len(idx):
        best_total, best_path = -1.0, []
        for start in list(xy):
            path: list[tuple[int, int]] = []          # (кусок, узел входа)
            here: set[int] = set()
            seen = {start}
            cur, total = start, 0.0
            while True:
                step = None
                for opt in sorted(options(cur, used | here), key=lambda o: -o[0]):
                    if opt[3] not in seen:
                        step = opt
                        break
                if step is None:
                    break
                _, piece, entry, other = step
                path.append((piece, entry))
                here.add(piece)
                total += length[piece]
                seen.add(other)
                cur = other
            if total > best_total:
                best_total, best_path = total, path
        if not best_path:
            piece = next(i for i in idx if i not in used)
            single = list(pieces[piece])
            lines.append((single, length[piece], 0.0) if with_orientation else single)
            used.add(piece)
            continue
        chain: list[tuple[float, float]] = []
        forward, backward = 0.0, 0.0      # сколько метров уложено по OSM и против
        for piece, entry in best_path:
            coords = pieces[piece]
            if na[piece] != nb[piece]:
                if _haversine_deg(xy[entry], coords[-1]) < _haversine_deg(xy[entry], coords[0]):
                    coords = coords[::-1]
                    backward += length[piece]
                else:
                    forward += length[piece]
            # Совпадающий узел не дублируем (допуск 1 м). Больший допуск съел бы до 150 м
            # геометрии на каждой сшивке — этого хватало, чтобы уехала ширина устья.
            if chain and _haversine_deg(chain[-1], coords[0]) <= 1.0:
                chain.extend(coords[1:])
            else:
                chain.extend(coords)
            used.add(piece)
        lines.append((chain, forward, backward) if with_orientation else chain)
    return lines


def _line_len(coords: list[tuple[float, float]]) -> float:
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        total += geo.haversine_m(lat1, lon1, lat2, lon2)
    return total


def _haversine_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Расстояние между двумя точками (lon, lat) в метрах."""
    return geo.haversine_m(a[1], a[0], b[1], b[0])


@dataclass
class WaterMetrics:
    id: str
    name: str | None
    kind: str
    order: int | None
    nearest_m: float
    in_mountain: str | None
    out_mountain: str | None
    bend: str | None
    bend_kind: str | None
    curvature_radius_m: float | None
    straight_at_site: bool | None
    confluences: list[dict]
    water_mouth: dict | None
    outflow_visible: bool | None
    flow_dem_dir: str | None
    flow_osm_dir: str | None
    flow_agree: bool | None
    flow_chosen: str | None
    flow_rule: int | None
    flow_confidence: str | None
    dem_slope_m_per_km: float | None
    dem_slope_snr: float | None
    dem_reason: str | None
    tidal: bool | None
    tidal_source: str | None
    source: str
    tidal_reason: str | None = None
    loop: dict | None = None
    site_position: str | None = None
    hidden_share: float | None = None
    visible: bool | None = None
    pieces: int = 1
    between_branches: bool | None = None
    between_pair_deg: list[int] | None = None
    crossings_2km: int = 0
    crossing_angle_deg: float | None = None
    # длина русла внутри круга 2 км и причина, если вход и выход попали в одну гору (E-020)
    length_2km_m: float | None = None
    same_mountain_reason: str | None = None
    flow_ways_share: float | None = None
    discharge_m3s: float | None = None
    hydro_distance_m: float | None = None
    hydro_agree: float | None = None
    coords: list[tuple[float, float]] = field(default_factory=list)
    # места, где русло входит в водоём: не слияние, поэтому отдельный список (E-017)
    waterbody_links: list[dict] = field(default_factory=list)
    geom: str = "line"             # line / area: контур водоёма не русло
    # поля водоёма: у контура нет входа/выхода, изгиба и петли, но есть площадь и сторона (E-018)
    nearest_azimuth_deg: float | None = None
    nearest_mountain: str | None = None
    area_m2: float | None = None
    site_inside: bool | None = None
    riverbank_of: str | None = None
    # линия канала внутри дока: отдельной рекой не выводится (E-020 п.1)
    dock_channel_of: str | None = None
    # G3: имена любой письменности и признак временной воды
    names: list[str] = field(default_factory=list)
    seasonal: bool | None = None


def is_watercourse(obj) -> bool:
    """Русло реки: линия, а не контур водоёма.

    Единственное место, где решается «считать ли речные поля». Принимает `Feature`,
    `MergedLine` и `WaterMetrics` — у всех есть `geom` и `kind`. Площадной контур получает
    только поля водоёма: у него нет входа и выхода воды, изгиба, петли и направления (E-018).
    """
    return getattr(obj, "geom", "line") == "line" and getattr(obj, "kind", None) in {
        "river", "stream", "canal", "culvert"}


def _measure_waterbody(feature, *, pts: np.ndarray, nearest_seg: int, nearest_m: float,
                       hole_pts: list[np.ndarray] | None = None,
                       offset_deg: float = 0.0) -> "WaterMetrics":
    """Контур водоёма: площадь, сторона и ближайшая точка, без речных полей.

    `in_mountain`, `out_mountain`, `bend`, `loop`, `flow_*`, `water_mouth` и `tidal_*` не
    заполняются, потому что их нет по смыслу: у бассейна нет течения и устья. Речных нулей
    тоже не пишем — потребитель должен отличать «не считали» от «посчитали и вышло 0».
    """
    x, y = float(pts[nearest_seg][0]), float(pts[nearest_seg][1])
    bearing = geo.azimuth_xy(x, y, offset_deg)
    return WaterMetrics(
        names=list(getattr(feature, "names", []) or []),
        seasonal=getattr(feature, "seasonal", None),
        id=getattr(feature, "key", None) or f"osm-{feature.osm_type}-{feature.osm_id}",
        name=feature.name, kind=feature.kind, order=None,
        nearest_m=round(nearest_m, 1),
        in_mountain=None, out_mountain=None, bend=None, bend_kind=None,
        curvature_radius_m=None, straight_at_site=None, confluences=[],
        water_mouth=None, outflow_visible=None, flow_dem_dir=None, flow_osm_dir=None,
        flow_agree=None, flow_chosen=None, flow_rule=None, flow_confidence=None,
        dem_slope_m_per_km=None, dem_slope_snr=None, dem_reason=None,
        tidal=None, tidal_source=None,
        geom="area",
        nearest_azimuth_deg=round(bearing, 1),
        nearest_mountain=geo.mountain_name(bearing),
        # дыры мультиполигона вычитаются: остров внутри озера — суша (E-022)
        area_m2=(round(max(0.0, abs(_polyline_area(pts))
                          - sum(abs(_polyline_area(h)) for h in hole_pts or [])), 1)
                 if len(pts) >= 4 else None),
        site_inside=(bool(_point_in_poly(0.0, 0.0, pts)) if len(pts) >= 4 else None),
        hidden_share=getattr(feature, "hidden_share", None),
        visible=(getattr(feature, "hidden_share", None) or 0.0) < 0.8,
        coords=list(feature.coords),
        source=getattr(feature, "source", "openstreetmap"),
    )


def measure_feature(feature, *, site_xy: np.ndarray, lat: float, lon: float, crs: int | str,
                    grid, in_radius_m: float = 2000.0, meso_radius_m: float = 8000.0) -> WaterMetrics | None:
    if len(feature.coords) < 2:
        return None
    pts = to_utm(feature.coords, crs) - site_xy          # координаты относительно участка
    dists = _segment_distances(pts, np.zeros(2)) if len(pts) > 1 else np.array([np.hypot(*pts[0])])
    nearest_seg = int(np.argmin(dists))
    nearest_m = float(dists[nearest_seg])
    if nearest_m > meso_radius_m:
        return None

    if getattr(feature, "geom", "line") == "area":
        # Контур водоёма: площадь, сторона и ближайшая точка. Речных полей у него нет по
        # смыслу — бассейн не втекает и не вытекает, у него нет ни устья, ни петли (E-018).
        holes = [to_utm(hole, crs) - site_xy for hole in (getattr(feature, "holes", None) or [])]
        return _measure_waterbody(feature, pts=pts, nearest_seg=nearest_seg, nearest_m=nearest_m,
                                  offset_deg=geo.convergence_deg(lat, lon, crs),
                                  hole_pts=holes)

    tags = getattr(feature, "tags", {}) or {}
    # Труба — свойство всей линии, а не ближайшего куска: с 6.1.1 склейка несёт теги
    # ближайшего к участку куска, и поверхностная река с коротким засыпанным участком
    # (River Moselle, hidden_share 0.10) получала kind: culvert. Теги нужны для приливности
    # и правила течения, для трубы — нет.
    #
    # Вид определяется здесь, до правил течения: иначе правило зависело от того, каким путём
    # объект стал трубой (тег куска давал rule 0, доля скрытого — правило 3/5), и две
    # одинаково подземные реки получали разный ответ (E-012).
    hidden = getattr(feature, "hidden", False)
    share = getattr(feature, "hidden_share", None)
    if share is not None and share >= 0.8:
        hidden = True
    kind = "culvert" if hidden else feature.kind

    # --- приливность (W5): тег OSM, иначе таблица известных границ ---
    from pyproj import Transformer
    inv = Transformer.from_crs(geo.crs_str(crs), "EPSG:4326", always_xy=True)
    nearest_abs = pts[nearest_seg] + site_xy
    # Сторона участка — по ближайшей точке русла: она нужна и тогда, когда вход с выходом
    # не измерить (вне круга 2 км), чтобы в отчёте остались расстояние и сторона.
    offset = geo.convergence_deg(lat, lon, crs)
    side_bearing = geo.azimuth_xy(float(pts[nearest_seg][0]), float(pts[nearest_seg][1]), offset)
    n_lon, n_lat = inv.transform(nearest_abs[0], nearest_abs[1])
    tide = tides.status(tags, feature.name, n_lon, n_lat)

    # --- уклон по рельефу: регрессия высот по длине, а не две точки (W4) ---
    dem_slope = dem_snr = None
    dem_reason = "no_dem"
    flow_dem = None
    if grid is not None and len(pts) > 4:
        seg = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        near_center = cum[nearest_seg]
        window = (cum >= near_center - 2000.0) & (cum <= near_center + 2000.0)
        idx = np.where(window)[0]
        if len(idx) >= 6:
            xs = cum[idx] / 1000.0                       # километры вдоль русла
            zs = np.array([grid.point_value(*(pts[k] + site_xy)) for k in idx])
            ok = ~np.isnan(zs)
            if ok.sum() >= 6:
                xs, zs = xs[ok], zs[ok]
                slope, intercept = np.polyfit(xs, zs, 1)     # м на км
                resid = zs - (slope * xs + intercept)
                noise = float(np.nanstd(resid)) or 0.5
                dem_slope = float(slope)
                span_km = float(xs.max() - xs.min()) or 1.0
                dem_snr = float(abs(slope) * span_km / max(noise, 0.5))
                dem_reason = None
                if abs(dem_slope) > 1e-9:
                    flow_dem = "downstream_toward_end" if dem_slope < 0 else "downstream_toward_start"
            else:
                dem_reason = "no_dem_values"
    elif grid is None:
        dem_reason = "no_dem"

    # --- выбор направления по правилам W4 (П4: пишем оба и причину) ---
    flow_osm = "downstream_toward_end" if len(pts) > 1 else None
    explicit_dir = (tags.get("direction") or "").lower()
    flow_rule, flow_conf, flow_chosen = 5, "medium", None
    if kind in {"canal", "culvert", "lake", "sea", "dock"}:
        flow_rule, flow_conf = 0, "high"
        flow_chosen = None                     # у канала и трубы течения нет
    elif explicit_dir in {"up", "down", "forward", "backward"} or tags.get("oneway"):
        flow_rule, flow_conf, flow_chosen = 1, "high", flow_osm
    elif tide["tidal"] is True:
        flow_rule, flow_conf, flow_chosen = 2, "medium", flow_osm
    elif dem_snr is not None and dem_snr >= 3.0 and flow_dem == flow_osm:
        flow_rule, flow_conf, flow_chosen = 3, "high", flow_dem
    elif dem_snr is not None and dem_snr >= 3.0 and flow_dem != flow_osm:
        flow_rule, flow_conf, flow_chosen = 4, "low", flow_osm
    else:
        flow_rule, flow_conf, flow_chosen = 5, "medium", flow_osm
        if dem_reason is None:
            dem_reason = "flat"
        flow_dem = None

    # --- 來水 / 去水 по выбранному направлению ---
    cross = _crossings_with_circle(pts, in_radius_m)
    if len(cross) >= 2:
        a, b = cross[0], cross[-1]
    else:
        a, b = pts[0], pts[-1]
    upstream, downstream = (a, b) if flow_chosen != "downstream_toward_start" else (b, a)
    in_bearing = geo.azimuth_xy(upstream[0], upstream[1], offset)
    out_bearing = geo.azimuth_xy(downstream[0], downstream[1], offset)
    in_m = geo.mountain_name(in_bearing)
    out_m = geo.mountain_name(out_bearing)
    crossing_angle = geo.angle_distance(in_bearing, out_bearing)

    # изгиб: центр кривизны в точке nearest; участок внутри кривой = обнимает.
    # Окно симметричное по длине дуги, а не по индексам: при склейке линия может идти в любую
    # сторону, и «сосед по индексу» менялся вместе с направлением — bend переворачивался на
    # ровном месте (E-014).
    i = min(max(nearest_seg, 1), len(pts) - 2)
    center = _bend_center(pts, i, half_window_m=30.0)
    if center is None:                      # ровный участок — берём окно шире
        center = _bend_center(pts, i, half_window_m=150.0)
    bend, radius_m = None, None
    if center is not None:
        radius_m = float(np.hypot(*(center - pts[i])))
        to_center = center - pts[i]
        to_site = -pts[i]
        if np.hypot(*to_site) > 1.0 and np.hypot(*to_center) > 1.0:
            cos = float(np.dot(to_center, to_site) / (np.hypot(*to_center) * np.hypot(*to_site)))
            bend = "inner" if cos > 0 else "outer"
    straight = radius_m is None or radius_m > 5000.0

    # водные врата: где вода покидает круг 8 км — по выбранному направлению
    mouth = None
    outer = _crossings_with_circle(pts, meso_radius_m)
    if outer and grid is not None:
        exit_pt = outer[-1] if flow_chosen != "downstream_toward_start" else outer[0]
        if len(outer) == 1:
            exit_pt = outer[0]
        mouth = _water_mouth(grid, exit_pt, pts[i], site_xy, offset_deg=offset)

    visible = None
    if mouth is not None and grid is not None:
        visible = _line_of_sight(grid, np.zeros(2), np.array([mouth.pop("_x"), mouth.pop("_y")]))

    loop_raw = _detect_loop(pts, offset_deg=offset) if len(pts) > 20 else None
    loop = None
    site_position = None
    bend_kind = "straight" if (radius_m is None or radius_m > 5000.0) else "bend"
    if loop_raw:
        # R4: петля — свойство русла (ratio, поворот ≥ 180°, площадь кольца), а не участка.
        # Положение участка пишется отдельно, чтобы интерпретатор знал и про 反弓.
        site_position = loop_raw.get("site_position")
        if loop_raw.get("is_loop"):
            loop = {k: v for k, v in loop_raw.items() if k != "is_loop"}
            bend_kind = "loop"
        elif loop_raw.get("ratio", 0.0) >= 1.5:
            bend_kind = "bend"
    between = site_between_branches(pts) if len(pts) > 2 else {"between_branches": False}

    # I-07b (E-020): одноимённые горы входа и выхода у видимой воды объясняются причиной.
    # Проверка I-07 молчала, если ручей мало пересекал круг 2 км, и короткий ручей проходил
    # её насквозь. Теперь нарушение отделено от предупреждения: нарушение — только длинный
    # кусок с пологим входом, остальное получает причину и пониженную уверенность.
    length_2km = _coverage_m(feature.coords, site_xy, crs, in_radius_m)
    same_reason = None
    if in_m and in_m == out_m and (feature.hidden_share or 0.0) < 0.8:
        if loop:
            same_reason = "петля"
        elif length_2km <= 0.0:
            same_reason = "вне круга выборки"
        elif length_2km < 500.0:
            same_reason = "короткий кусок"
        elif crossing_angle < 30.0:
            same_reason = "касательная"
        if same_reason:
            # E-020 п.2: причина «вход и выход в одной горе» понижает уверенность у любого
            # видимого водотока. Раньше это касалось только river/stream, а у каналов стояло
            # `high` по правилу 0 («течения нет»), и док-каналы с 乙 → 乙 выглядели надёжно.
            flow_conf = "low"
        if same_reason == "вне круга выборки":
            # Внутри круга 2 км от русла нет ни метра: пара 來水/去水 не измерена, поэтому горы,
            # направление и уверенность не пишем вовсе, а не выдаём «вход 辰, выход 辰».
            # `flow_chosen` убирается вместе с уверенностью — иначе I-10 видит поле без пары.
            in_m = out_m = None
            flow_conf = None
            flow_chosen = None

    return WaterMetrics(
        id=getattr(feature, "key", None) or f"osm-{feature.osm_type}-{feature.osm_id}",
        name=feature.name,
        names=list(getattr(feature, "names", []) or []),
        seasonal=getattr(feature, "seasonal", None),
        kind=kind,
        order=None,
        nearest_m=round(nearest_m, 1),
        in_mountain=in_m,
        out_mountain=out_m,
        bend=bend,
        bend_kind=bend_kind,
        curvature_radius_m=None if radius_m is None else round(radius_m, 1),
        straight_at_site=bool(straight and nearest_m < in_radius_m),
        confluences=[],
        geom=getattr(feature, "geom", "line"),
        water_mouth=mouth,
        hidden_share=feature.hidden_share,
        visible=(feature.hidden_share or 0.0) < 0.8,
        outflow_visible=visible,
        flow_osm_dir=flow_osm,
        flow_dem_dir=flow_dem,
        flow_chosen=flow_chosen,
        flow_rule=flow_rule,
        flow_confidence=flow_conf,
        flow_agree=(flow_dem == flow_osm) if flow_dem and flow_osm else None,
        dem_slope_m_per_km=None if dem_slope is None else round(dem_slope, 3),
        dem_slope_snr=None if dem_snr is None else round(dem_snr, 2),
        dem_reason=dem_reason,
        tidal=tide["tidal"],
        tidal_source=tide["tidal_source"],
        source=getattr(feature, "source", "osm_way"),
        loop=loop,
        site_position=site_position,
        between_branches=between.get("between_branches"),
        between_pair_deg=between.get("pair_deg"),
        crossings_2km=len(cross),
        crossing_angle_deg=round(crossing_angle, 1),
        length_2km_m=round(length_2km, 1) if length_2km is not None else None,
        same_mountain_reason=same_reason,
        nearest_azimuth_deg=round(side_bearing, 1),
        nearest_mountain=geo.mountain_name(side_bearing),
        coords=list(feature.coords),
    )


def _water_mouth(grid, exit_pt: np.ndarray, flow_pt: np.ndarray, site_xy: np.ndarray,
                 offset_deg: float = 0.0,
                 *, max_steps: int = 60, step_m: float = 50.0) -> dict | None:
    """Водные врата: ширина долины, высоты бортов и замкнутость (W1).

    Ширина — до подъёма борта на 20 м. Если борт не поднялся за 3 км, сторона считается
    «упёршейся в предел» (width_capped), а не нулевой. Замкнутость — не доля от 4 км,
    а отношение высоты бортов к полуширине долины: enclosure = mean(hills) / (width/2),
    затем монотонно в [0,1): closure_index = enclosure / (1 + enclosure).
    Обе стороны упёрлись → closure_index null, причина flat_valley (П1, П2).
    """
    flow_dir = exit_pt - flow_pt
    norm = np.hypot(*flow_dir)
    if norm < 1e-6:
        return None
    flow_dir = flow_dir / norm
    perp = np.array([-flow_dir[1], flow_dir[0]])
    x_abs, y_abs = exit_pt + site_xy
    z0 = grid.point_value(x_abs, y_abs)
    level_source = "mouth"
    if math.isnan(z0):
        # в устье DEM часто пуст: вода не даёт высоты, и Copernicus оставляет nodata.
        # Это не повод отказываться от врат: уровень воды ищем в ближайшей клетке поперёк
        # русла (в пределах 500 м) и говорим об этом полем `level_source` (E-036).
        found = None
        for d in range(1, 11):
            for sign in (1, -1):
                z = grid.point_value(*(site_xy + exit_pt + perp * sign * d * step_m))
                if not math.isnan(z):
                    found = float(z)
                    break
            if found is not None:
                break
        if found is None:
            return None
        z0 = found
        level_source = "nearest_cell"

    half_widths, hills, capped, capped_sides = [], [], [], 0
    for sign in (1, -1):
        half_width, top, stop_reason = 0.0, 0.0, "capped"
        for step in range(1, max_steps + 1):
            d = step * step_m
            z = grid.point_value(*(site_xy + exit_pt + perp * sign * d))
            if math.isnan(z):
                stop_reason = "grid_edge"
                break
            top = max(top, float(z) - z0)
            if z - z0 > 20.0:
                half_width = d
                # дальше по профилю берём максимум до конца замера — борт бывает выше порога
                for step2 in range(step + 1, max_steps + 1):
                    z2 = grid.point_value(*(site_xy + exit_pt + perp * sign * step2 * step_m))
                    if not math.isnan(z2):
                        top = max(top, float(z2) - z0)
                stop_reason = "slope"
                break
            half_width = d
        # I-12: ноль при упёршейся стороне — не измеренная высота, а «борта не нашли».
        # Писать 0 значит утверждать, что борт ровно нулевой; честнее не писать ничего.
        hit_limit = stop_reason != "slope"
        if hit_limit:
            capped_sides += 1
        half_widths.append(half_width)
        capped.append(hit_limit)
        # Борт ниже полуметра — не борт: это парапет или неровность, и после округления он
        # превращается в 0.0, то есть в утверждение «высота нулевая» (I-12).
        hills.append(None if hit_limit and top < 0.5 else top)

    width = half_widths[0] + half_widths[1]
    hill_left, hill_right = hills[0], hills[1]
    known = [h for h in hills if h is not None]
    mean_hill = sum(known) / len(known) if known else None
    half = width / 2.0
    enclosure = None
    closure_index = None
    reason = None
    if capped_sides >= 2 or mean_hill is None:
        reason = "flat_valley"
    elif half <= 1.0:
        enclosure = 99.0
        closure_index = 1.0
    else:
        enclosure = mean_hill / half
        closure_index = round(enclosure / (1.0 + enclosure), 3)

    bearing = geo.azimuth_xy(exit_pt[0], exit_pt[1], offset_deg)
    return {
        "mountain": geo.mountain_name(bearing),
        "width_m": round(width, 1),
        "left_hill_m": None if hill_left is None else round(hill_left, 1),
        "right_hill_m": None if hill_right is None else round(hill_right, 1),
        "width_capped": capped_sides > 0,
        "capped_sides": capped_sides,
        "enclosure": None if enclosure is None else round(enclosure, 3),
        "closure_index": closure_index,
        "closure_reason": reason,
        # П7: в застройке DSM даёт крыши вместо бортов; очень малая замкнутость означает,
        # что настоящих бортов долины мы не нашли, — это low, а не «средне»
        "closure_confidence": ("low" if (capped_sides > 0 or (enclosure is not None and enclosure < 0.05))
                               else ("high" if (enclosure or 0.0) >= 0.2 else "medium")),
        # уровень воды в устье: измерен в самой точке или взят в ближайшей клетке
        "level_source": level_source,
        "_x": float(x_abs),
        "_y": float(y_abs),
    }


def _line_of_sight(grid, start: np.ndarray, end: np.ndarray, *, eye_m: float = 2.0,
                   step_m: float = 50.0) -> bool | None:
    """Видна ли точка выхода воды из участка (прямая поверх рельефа)."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1.0:
        return True
    z0 = grid.point_value(start[0], start[1])
    z1 = grid.point_value(end[0], end[1])
    if math.isnan(z0) or math.isnan(z1):
        return None
    h0, h1 = z0 + eye_m, z1 + 0.5
    n = max(2, int(length / step_m))
    for k in range(1, n):
        t = k / n
        z = grid.point_value(start[0] + dx * t, start[1] + dy * t)
        if math.isnan(z):
            continue
        if z > h0 + (h1 - h0) * t + 1.0:
            return False
    return True


def _point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def _polyline_area(ring: np.ndarray) -> float:
    """Площадь замкнутой ломаной (формула шнурков), м²."""
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])) / 2.0)


def _turn_deg(pts: np.ndarray) -> float:
    """Суммарный поворот русла по азимуту, градусы (накопленный модуль со знаком)."""
    if len(pts) < 3:
        return 0.0
    d = np.diff(pts, axis=0)
    angles = np.arctan2(d[:, 0], d[:, 1])
    steps = np.diff(angles)
    steps = (steps + np.pi) % (2 * np.pi) - np.pi          # нормализуем в (-π, π]
    return float(abs(np.degrees(steps).sum()))


def _site_position(ring: np.ndarray, mid: np.ndarray, far: np.ndarray) -> str:
    """Где стоит участок относительно петли: внутри, снаружи выпукло или вогнуто.

    反弓 (вода выгибается от участка) — это `outside_convex`: участок снаружи, на стороне
    выпуклости. Если участок снаружи, но с противоположной стороны — он «обнимает» изгиб
    снаружи (`outside_concave`).
    """
    if _point_in_poly(0.0, 0.0, ring):
        return "inside"
    bulge = far - mid
    if float(np.dot(-mid, bulge)) > 0:      # участок в начале координат, относительно хорды
        return "outside_convex"
    return "outside_concave"


def _detect_loop(pts: np.ndarray, *, offset_deg: float = 0.0,
                 near_m: float = 2000.0, apart_m: float = 1500.0,
                 min_ratio: float = 3.0, min_turn_deg: float = 180.0,
                 area_k: float = 0.05, near_site_m: float = 8000.0) -> dict | None:
    """Петля русла: возврат к себе, поворот не меньше 180° и площадь кольца.

    Петля — свойство реки, а не участка (R4). Положение участка пишется отдельно
    (`site_position`: inside / outside_convex / outside_concave). Пороги: ratio = along/gap,
    накопленный поворот русла на участке i…j и площадь кольца в долях gap².
    Возвращается `None`, если даже изгиба нет (ratio < 1.5).

    Кандидаты ограничены точками ближе `near_site_m`: у реки в 3000 км самая «петлистая»
    пара лежит за сотни километров, и пакет участка показывал бы чужой изгиб.
    """
    if len(pts) < 20:
        return None
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    stride = max(1, len(pts) // 600)
    near = [i for i in range(0, len(pts), stride)
            if float(np.hypot(pts[i][0], pts[i][1])) <= near_site_m]
    candidates = []
    for pos, i in enumerate(near):
        for j in near[pos + 1:]:
            if j <= i + 2:
                continue
            along = cum[j] - cum[i]
            if along < apart_m:
                continue
            gap = float(np.hypot(*(pts[i] - pts[j])))
            if gap > near_m or gap < 1.0:
                continue
            candidates.append((along / gap, gap, along, i, j))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    if candidates[0][0] < 1.5:
        return None

    # петля — не максимальное отношение: из лучших пар берём ту, где русло реально
    # завернулось (поворот ≥ 180°) и кольцо имеет площадь; иначе остаётся изгиб
    chosen = None
    for cand in candidates[:120]:
        ratio_c, gap_c, along_c, i_c, j_c = cand
        turn_c = _turn_deg(pts[i_c:j_c + 1])
        ring_c = np.vstack([pts[i_c:j_c + 1], pts[i_c:i_c + 1]])
        area_c = _polyline_area(ring_c)
        if turn_c >= min_turn_deg and area_c >= area_k * gap_c * gap_c:
            chosen = (ratio_c, gap_c, along_c, i_c, j_c, turn_c, area_c, ring_c)
            break
    if chosen is None:
        ratio_f, gap_f, along_f, i_f, j_f = candidates[0]
        turn_f = _turn_deg(pts[i_f:j_f + 1])
        ring_f = np.vstack([pts[i_f:j_f + 1], pts[i_f:i_f + 1]])
        chosen = (ratio_f, gap_f, along_f, i_f, j_f, turn_f, _polyline_area(ring_f), ring_f)

    ratio, gap, along, i, j, turn, area, ring = chosen
    mid = (pts[i] + pts[j]) / 2.0
    chord_line = pts[j] - pts[i]
    norm = np.hypot(*chord_line) or 1.0
    normal = np.array([-chord_line[1], chord_line[0]]) / norm
    offsets = np.abs((ring[:-1] - pts[i]) @ normal)
    far = ring[int(np.argmax(offsets))]
    bearing = geo.azimuth_xy(mid[0], mid[1], offset_deg)

    is_loop = (ratio >= min_ratio and turn >= min_turn_deg and area >= area_k * gap * gap)
    margin, confidence = loop_confidence(ratio, turn, area / (gap * gap))
    return {
        "gap_m": round(gap, 1), "along_m": round(along, 1), "ratio": round(ratio, 2),
        "turn_deg": round(turn, 1), "area_m2": round(area), "area_k": round(area / (gap * gap), 3),
        "mountain": geo.mountain_name(bearing),
        "is_loop": bool(is_loop),
        "site_position": _site_position(ring, mid, far),
        "margin": margin, "confidence": confidence,
    }


def site_between_branches(pts: np.ndarray, *, near_m: float = 1500.0, tol_deg: float = 30.0) -> dict:
    """Стоит ли участок между двумя ветвями русла (внутри излучины)."""
    if len(pts) < 3:
        return {"between_branches": False, "min_by_sector_deg": {}}
    dist = np.hypot(pts[:, 0], pts[:, 1])
    bearing = np.degrees(np.arctan2(pts[:, 0], pts[:, 1])) % 360.0
    best: dict[int, float] = {}
    for d, b in zip(dist, bearing):
        idx = int(((b + 7.5) % 360) // 15)
        if d < best.get(idx, 1e18):
            best[idx] = float(d)
    sectors = sorted(best)
    between, pair = False, None
    for i in sectors:
        for delta in (10, 11, 12, 13, 14):
            j = (i + delta) % 24
            if j in best and best[i] <= near_m and best[j] <= near_m:
                between = True
                pair = (i * 15, ((i + delta) % 24) * 15, round(best[i]), round(best[j]))
                break
        if between:
            break
    return {"between_branches": between, "pair_deg": None if not pair else [pair[0], pair[1]],
            "half_widths_m": None if not pair else [pair[2], pair[3]],
            "nearest_by_sector_m": {str(k * 15): round(v) for k, v in sorted(best.items())}}


def _seg_intersect(p, q, r, s_):
    r_ = q - p
    s2 = s_ - r
    denom = r_[0] * s2[1] - r_[1] * s2[0]
    if abs(denom) < 1e-12:
        return None
    t = ((r[0] - p[0]) * s2[1] - (r[1] - p[1]) * s2[0]) / denom
    u = ((r[0] - p[0]) * r_[1] - (r[1] - p[1]) * r_[0]) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p + r_ * t
    return None


def _bbox(pts: np.ndarray) -> tuple[float, float, float, float]:
    return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())


def _bbox_near(a, b, pad: float) -> bool:
    return not (a[1] + pad < b[0] or b[1] + pad < a[0] or a[3] + pad < b[2] or b[3] + pad < a[2])


def _confluence_kind(a_kind: str, b_kind: str) -> str:
    kinds = {a_kind, b_kind}
    if "culvert" in kinds:
        return "culvert_outfall"
    if "canal" in kinds:
        return "canal_junction"
    return "confluence"


def find_waterbody_links(features, *, site_xy: np.ndarray, crs: int | str,
                         radius_m: float = 8000.0) -> dict[str, list[dict]]:
    """Где линия русла входит в водоём: конец линии лежит внутри контура площади.

    Это не слияние: река не «впадает» в озеро в смысле двух русел, она в него втекает.
    В `confluences` такие места не пишутся — раньше контуры берегов считались водотоками,
    и пересечение двух берегов у Стратфорда выглядело как впадение (E-017).

    Ключ — uid линии, значение — список `{"water": имя или uid водоёма, "dist_m", "kind"}`.
    """
    from shapely.geometry import Point, Polygon

    def label(f, idx: int) -> str:
        """Имя записи: uid склейки, иначе osm-тип-id, иначе порядковый — у MergedLine нет osm_id."""
        return (getattr(f, "uid", None) or getattr(f, "key", None)
                or (f"osm-{f.osm_type}-{f.osm_id}-{idx}" if hasattr(f, "osm_id") else f"line-{idx}"))

    areas: list[tuple[str, object]] = []
    for idx, f in enumerate(features):
        if f.geom != "area" or len(f.coords) < 4:
            continue
        ring = to_utm(f.coords, crs) - site_xy
        if float(np.hypot(*ring.T).min()) > radius_m:
            continue
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 2000.0:
            # меньше 2000 м² — не водоём, а щель в обводке контура
            continue
        # ключ контура: имя, иначе uid склейки — у MergedLine нет osm_type/osm_id (E-022)
        areas.append((f.name or label(f, idx), poly))

    out: dict[str, list[dict]] = {}
    for idx, f in enumerate(features):
        if f.geom != "line" or f.kind not in {"river", "stream", "canal"} or len(f.coords) < 2:
            continue
        pts = to_utm(f.coords, crs) - site_xy
        uid = label(f, idx)
        title = f.name or uid
        for area_label, poly in areas:
            if area_label == title:
                continue
            # конец линии внутри контура — вход в водоём; у русла, нарисованного от устья,
            # вход в озеро — это начало, поэтому проверяются оба конца
            for end in (pts[-1], pts[0]):
                if not poly.contains(Point(float(end[0]), float(end[1]))):
                    continue
                dist = float(np.hypot(*end))
                if dist <= radius_m:
                    out.setdefault(uid, []).append({"water": area_label,
                                                    "dist_m": round(dist, 1),
                                                    "kind": "enters_waterbody"})
                break
    return out


def find_confluences(features, *, site_xy: np.ndarray, crs: int | str,
                     offset_deg: float = 0.0,
                     radius_m: float = 8000.0, merge_m: float = 50.0,
                     max_per_line: int = 5, clip_m: float = 5000.0) -> dict[str, list[dict]]:
    """Слияния водотоков: точки пересечения, общие узлы, без дублей (W3).

    Три вещи, которых не хватало:
    1) точки одной пары ближе `merge_m` схлопываются в одну запись;
    2) совпадающие концы линий (общий узел OSM, до 0.5 м) тоже считаются слиянием;
    3) пары отбираются по пересечению bbox и грубой сетке-хешу, а не полным перебором.

    Ключ линии — `uid` куска, а не имя реки: у одной реки кусков несколько, и словарь по
    имени молча терял все, кроме последнего (слияние Оки и Волги так и не находилось).
    Линии обрезаются окном `clip_m`: перебор пар отрезков квадратичен, а слияние дальше
    радиуса участка в пакет всё равно не попадает.

    На каждый водоток пишется не больше `max_per_line` ближайших слияний: интерпретатору
    нужны ближние, остальное живёт в water.geojson (W7).
    """
    lines: dict[str, np.ndarray] = {}
    kinds: dict[str, str] = {}
    labels: dict[str, str] = {}
    base: dict[str, str] = {}          # ключ куска в круге -> ключ реки
    for idx, f in enumerate(features):
        # только русла: контур берега — не водоток, и пересечение двух берегов не впадение (E-017)
        if f.kind in {"river", "stream", "canal"} and len(f.coords) > 1 and f.geom == "line":
            uid = (getattr(f, "uid", None) or getattr(f, "key", None)
                   or f"osm-{f.osm_type}-{f.osm_id}-{idx}")
            full = to_utm(f.coords, crs) - site_xy
            pieces = clip_near(full, clip_m) or []
            if not pieces:
                continue
            # Все куски линии внутри круга, а не только самый длинный: река подковой проходит
            # мимо участка дважды, и приток впадает в тот рукав, где вершин меньше — прежний
            # выбор «кто длиннее» терял слияние у обеих рек (E-015).
            for k, piece in enumerate(pieces):
                key = uid if len(pieces) == 1 else f"{uid}#{k}"
                lines[key] = piece
                base[key] = uid
                kinds[key] = f.kind
                labels[key] = getattr(f, "key", None) or uid

    boxes = {key: _bbox(pts) for key, pts in lines.items()}
    result: dict[str, list[dict]] = {uid: [] for uid in dict.fromkeys(base.values())}

    keys = list(lines)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key, b_key = keys[i], keys[j]
            # Куски одной реки сходятся у острова или старицы — это не слияние, а её же русло:
            # без этой проверки Темза «впадала в себя» дважды (E-014).
            if labels[a_key] and labels[a_key] == labels[b_key]:
                continue
            # грубый пространственный отбор: без пересечения bbox пары не бывает слияния
            if not _bbox_near(boxes[a_key], boxes[b_key], pad=1.0):
                continue
            a, b = lines[a_key], lines[b_key]
            hits: list[np.ndarray] = []
            for p, q in zip(a[:-1], a[1:]):
                for r, s_ in zip(b[:-1], b[1:]):
                    hit = _seg_intersect(p, q, r, s_)
                    if hit is not None:
                        hits.append(hit)
            # общий узел: концы линий совпадают до 0.5 м, пересечения при этом может не быть
            for pa in (a[0], a[-1]):
                for pb in (b[0], b[-1]):
                    if float(np.hypot(*(pa - pb))) <= 0.5:
                        hits.append((pa + pb) / 2.0)
            if not hits:
                continue
            # схлопнуть близкие точки (одна и та же точка слияния, найденная дважды)
            clusters: list[list[np.ndarray]] = []
            for hit in hits:
                for cluster in clusters:
                    if float(np.hypot(*(cluster[0] - hit))) <= merge_m:
                        cluster.append(hit)
                        break
                else:
                    clusters.append([hit])
            for cluster in clusters:
                point = np.mean(np.array(cluster), axis=0)
                dist = float(np.hypot(*point))
                if dist > radius_m:
                    continue
                bearing = geo.azimuth_xy(point[0], point[1], offset_deg)
                entry = {"dist_m": round(dist, 1), "mountain": geo.mountain_name(bearing),
                         "kind": _confluence_kind(kinds[a_key], kinds[b_key])}
                result[base[a_key]].append({"with": labels[b_key], **entry})
                result[base[b_key]].append({"with": labels[a_key], **entry})

    for key, items in result.items():
        items.sort(key=lambda item: item["dist_m"])
        result[key] = items[:max_per_line]
    return result


def flow_convention_share(features: list[Feature], grid, crs: int | str, *,
                          min_drop_m: float = 1.0) -> float | None:
    """Доля кусков реки, где порядок отрисовки OSM совпадает с падением по рельефу.

    В OSM реку рисуют по течению — это соглашение, а не тег. Проверяем его рельефом:
    если конец куска ниже начала хотя бы на min_drop_m, порядок совпал. Возвращает долю
    совпавших кусков (None, если проверять нечего).
    """
    if grid is None:
        return None
    agree = total = 0
    for f in features:
        if f.kind not in {"river", "stream"} or not is_watercourse(f) or len(f.coords) < 2:
            continue
        pts = to_utm([f.coords[0], f.coords[-1]], crs)
        z0 = grid.point_value(*pts[0])
        z1 = grid.point_value(*pts[1])
        if math.isnan(z0) or math.isnan(z1):
            continue
        total += 1
        if z1 < z0 - min_drop_m:
            agree += 1
    return None if total == 0 else round(agree / total, 2)


def hydrorivers_metrics(reaches: list[dict] | None, *, site_xy: np.ndarray, lat: float,
                        lon: float, crs: int | str, grid=None,
                        max_km: float = 2.0) -> list["WaterMetrics"]:
    """Реки HydroRIVERS как объекты воды, когда OSM разрежен (G3.2).

    HydroRIVERS не знает имён, зато знает порядок и расход: это второй источник, который
    отвечает на вопрос «какая это река по величине», а не «как она называется». Каждый участок
    проходит те же измерения, что и линия OSM, и помечен `source: hydrorivers` — подмены
    источника в пакете не бывает незаметной.
    """
    out: list[WaterMetrics] = []
    for reach in reaches or []:
        points = [(float(a), float(b)) for a, b in (reach.get("_points") or [])]
        if len(points) < 2:
            continue
        feature = Feature(int(reach.get("HYRIV_ID") or 0), "hydrorivers", None, "river", {},
                          points, False, "line", source="hydrorivers")
        metrics = measure_feature(feature, site_xy=site_xy, lat=lat, lon=lon, crs=crs, grid=grid,
                                  meso_radius_m=int(max_km * 1000))
        if metrics is None:
            continue
        metrics.order = reach.get("ORD_STRA")
        metrics.discharge_m3s = reach.get("DIS_AV_CMS")
        metrics.hydro_distance_m = reach.get("distance_m")
        out.append(metrics)
    return out


def measure_water(lat: float, lon: float, *, crs: int | str, grid=None, site_xy: np.ndarray,
                  radius_m: int = 8000, allow_network: bool = True,
                  max_features: int = 2000, payload: dict | None = None,
                  wide_payload: dict | None = None, coast_payload: dict | None = None,
                  window_factor: float = 1.5, window_min_m: float = 12000.0) -> list[WaterMetrics]:
    """Вода в радиусе: линии склеены по имени, затем измерены.

    `payload` позволяет подать заранее сохранённый ответ Overpass — так работают эталоны
    в tests/golden без сети.

    Склеенная линия реки может тянуться на сотни километров, поэтому перед измерением она
    режется окном `max(window_factor·radius, window_min)`: все пороги пакета (2 км на входе и
    выходе, 8 км врат, 2 км на петлю) лежат внутри, а перебор слияний и геометрия в
    water.geojson остаются локальными.
    """
    if payload is None:
        payload = fetch_raw(lat, lon, radius_m, allow_network=allow_network)
    # участок нужен разбору: мультиполигоны собираются и обрезаются по кругу (E-022)
    features = parse_features(payload, lat=lat, lon=lon, radius_m=radius_m)[:max_features]
    merged = merge_by_name(features, near=(lat, lon))

    window_m = max(window_factor * float(radius_m), window_min_m)
    chunks: list[MergedLine] = []
    for line in merged:
        pts = to_utm(line.coords, crs) - site_xy
        runs = clip_runs_near(pts, window_m) if len(pts) > 1 else []
        if not runs:
            continue
        if len(runs) == 1 and runs[0] == (0, len(line.coords) - 1):
            chunks.append(line)
            continue
        for idx, (i0, i1) in enumerate(runs):
            # Узлы берутся из исходной линии в градусах, а не пересчётом нарезанных UTM-точек:
            # обратный пересчёт давал сдвиг ~1e-9 м, из-за которого устье притока (внутренняя
            # вершина главной реки) переставало быть общим узлом, и слияние Флит × Темза
            # пропадало (E-011).
            chunks.append(MergedLine(line.key, line.name, line.kind,
                                     list(line.coords[i0:i1 + 1]), line.ways, line.hidden,
                                     line.hidden_share, uid=f"{line.uid}-{idx}",
                                     tags=dict(line.tags), geom=line.geom,
                                     holes=[list(h) for h in line.holes], source=line.source,
                                     names=list(line.names), seasonal=line.seasonal))

    # слияния ищем только у воды, которая вообще рядом с участком: иначе O(n²) по всему городу
    near = []
    for line in chunks:
        pts_near = to_utm(line.coords, crs) - site_xy
        if float(np.hypot(*pts_near.T).min()) < 3000.0:
            near.append(line)
    confluences = find_confluences(near, site_xy=site_xy, crs=crs, radius_m=radius_m,
                                   offset_deg=geo.convergence_deg(lat, lon, crs),
                                   clip_m=float(radius_m))
    # Входы в водоёмы ищем по целым линиям и целым контурам, а не по нарезанным кускам:
    # кусок кольца — не замкнутый контур, и `Polygon` из него закрывает разрыв хордой (E-017).
    links = find_waterbody_links(merged, site_xy=site_xy, crs=crs, radius_m=radius_m)
    # одна река — несколько кусков: слияние, найденное на любом из них, относится к реке
    by_name: dict[str, list[dict]] = {}
    links_by_name: dict[str, list[dict]] = {}
    for line in near:
        key = hydronym.normalize_name(line.name) or line.key
        by_name.setdefault(key, []).extend(confluences.get(line.uid, []))
    for line in merged:
        key = hydronym.normalize_name(line.name) or line.key
        links_by_name.setdefault(key, []).extend(links.get(line.uid, []))

    flow_share = flow_convention_share(features, grid, crs)
    out: list[WaterMetrics] = []
    for m in chunks:
        metrics = measure_feature(m, site_xy=site_xy, lat=lat, lon=lon, crs=crs, grid=grid,
                                  meso_radius_m=radius_m)
        if metrics is None:
            continue
        key = hydronym.normalize_name(metrics.name) or metrics.id
        items = _dedupe_confluences(by_name.get(key, []))
        metrics.confluences = items[:4]
        metrics.waterbody_links = _dedupe_links(links_by_name.get(key, []))[:4]
        if flow_share is not None and is_watercourse(metrics):
            metrics.flow_agree = flow_share >= 0.6
            metrics.flow_ways_share = flow_share
        out.append(metrics)
    out = _one_entry_per_name(out, site_xy=site_xy, crs=crs, radius_m=radius_m)
    # G3.4: вода ближе километра к берегу приливная. Это не догадка: измеряется расстояние
    # до `natural=coastline`, и если берега в выборке нет — поле остаётся None, а не False.
    coast = coast_lines(coast_payload)
    if coast:
        for metric in out:
            if metric.tidal is not None:
                continue
            points = list(getattr(metric, "coords", []) or [])
            if not points:
                continue
            dist = _coast_distance_m(coast, points)
            if dist is not None and dist <= COAST_MAX_M:
                metric.tidal = True
                metric.tidal_source = "coastline"
                metric.tidal_reason = (f"берег в {dist:.0f} м: море рядом, вода следует приливу")
            elif dist is not None:
                metric.tidal_reason = (f"до берега {dist / 1000:.1f} км — прилив по берегу "
                                       "не подтверждён")
    # Берег реки (`natural=water, water=river` или `waterway=riverbank`) привязывается к линии
    # русла с тем же именем: отдельной «рекой» он не выводится, у него только поля водоёма (E-018).
    area_tags = {line.key: dict(getattr(line, "tags", {}) or {}) for line in merged
                 if getattr(line, "geom", "line") == "area"}
    line_ids: dict[str, str] = {}
    for metric in out:
        if is_watercourse(metric) and metric.name:
            line_ids.setdefault(hydronym.normalize_name(metric.name) or metric.name, metric.id)
    for metric in out:
        if metric.geom != "area":
            continue
        tags = area_tags.get(metric.id, {})
        if not ((tags.get("water") or "") == "river" or tags.get("waterway") == "riverbank"):
            continue
        if metric.name:
            metric.riverbank_of = line_ids.get(hydronym.normalize_name(metric.name) or metric.name)
    # Берег без имени: Темза размечена отношением `natural=water, water=river` без `name`,
    # и по имени его ни с чем не связать (E-022). Привязка по геометрии: ближайшая видимая
    # осевая того же вида в пределах 300 м — половина ширины Темзы около 80–100 м, поэтому
    # порог в 60 м не нашёл бы ничего. Имя берётся у линии: у контура его просто нет.
    unnamed = [m for m in out if m.geom == "area" and not m.name and not m.riverbank_of
               and ((area_tags.get(m.id, {}).get("water") or "") == "river"
                    or area_tags.get(m.id, {}).get("waterway") == "riverbank")]
    if unnamed:
        from shapely.geometry import LineString, Polygon

        lines_xy: list[tuple[str, str, LineString]] = []
        for metric in out:
            if not is_watercourse(metric) or not metric.name or not metric.coords:
                continue
            if metric.visible is False:
                continue
            pts = to_utm(metric.coords, crs) - site_xy
            if len(pts) > 1:
                lines_xy.append((metric.id, metric.name, LineString(pts)))
        for metric in unnamed:
            ring = to_utm(metric.coords, crs) - site_xy
            if len(ring) < 4:
                continue
            border = Polygon(ring).boundary
            best: tuple[float, str, str] | None = None
            for line_id, line_name, line in lines_xy:
                dist = border.distance(line)
                if dist <= 300.0 and (best is None or dist < best[0]):
                    best = (dist, line_id, line_name)
            if best is not None:
                metric.riverbank_of = best[1]
                metric.name = best[2]
    # Канал внутри дока (`waterway=canal` с именем дока) привязывается к контуру с тем же
    # именем: в отчёте он не идёт как отдельное русло, у дока остаются его поля водоёма.
    dock_ids: dict[str, str] = {}
    for metric in out:
        if metric.geom == "area" and metric.name and metric.kind in {"dock", "sea"}:
            dock_ids.setdefault(hydronym.normalize_name(metric.name) or metric.name, metric.id)
    for metric in out:
        if is_watercourse(metric) and metric.name and metric.geom != "area":
            metric.dock_channel_of = dock_ids.get(hydronym.normalize_name(metric.name) or metric.name)
    out.sort(key=lambda m: m.nearest_m)
    _resolve_tidal_by_tags(out, features, crs=crs, site_xy=site_xy)
    _resolve_tidal_by_reach(out, lat, lon, crs=crs, site_xy=site_xy, radius_m=radius_m,
                            wide_payload=wide_payload, allow_network=allow_network)
    return out


def _dedupe_confluences(items: list[dict], *, merge_m: float = 50.0) -> list[dict]:
    """Одно и то же слияние приходит от каждого куска обеих рек — оставить по одному."""
    out: list[dict] = []
    for item in sorted(items, key=lambda it: it["dist_m"]):
        for kept in out:
            if kept["with"] == item["with"] and abs(kept["dist_m"] - item["dist_m"]) <= merge_m:
                break
        else:
            out.append(item)
    return out


def _dedupe_links(items: list[dict], *, merge_m: float = 50.0) -> list[dict]:
    """Один и тот же вход в водоём приходит от каждого куска линии — оставить по одному."""
    out: list[dict] = []
    for item in sorted(items, key=lambda it: it["dist_m"]):
        if any(kept["water"] == item["water"] and abs(kept["dist_m"] - item["dist_m"]) <= merge_m
               for kept in out):
            continue
        out.append(item)
    return out


def _coverage_m(coords: list[tuple[float, float]] | None, site_xy: np.ndarray, crs: int | str,
                radius_m: float) -> float:
    """Сколько метров русла лежит в радиусе участка — мера «это главное русло, а не стаб»."""
    if not coords or len(coords) < 2:
        return 0.0
    pts = to_utm(coords, crs) - site_xy
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        mid = (a + b) / 2.0
        if float(np.hypot(*mid)) <= radius_m:
            total += float(np.hypot(*(b - a)))
    return total


def _one_entry_per_name(metrics: list[WaterMetrics], *, site_xy: np.ndarray | None = None,
                        crs: int | str | None = None,
                        radius_m: float = 8000.0) -> list[WaterMetrics]:
    """Одна река — одна запись, даже если русло в OSM разбито на несоприкасающиеся куски.

    Записи группируются по нормализованному имени и типу. Остаётся кусок, который покрывает
    больше всего русла в радиусе участка, а при равенстве — ближайший: «ближайший» без
    покрытия выбирал обрывок у шлюза, и у Кингстона запись о Темзе описывала 300-метровый
    стаб вместо главного русла (E-014). Доля подземного участка берётся максимальной по кускам
    (если часть реки в трубе — это важно сказать), а число кусков пишется в `pieces`.
    """
    groups: dict[str, list[WaterMetrics]] = {}
    for metric in metrics:
        # Группа — имя и вид геометрии: русло и его берег с одним именем — разные объекты,
        # и раньше один вытеснял другой, поэтому у контура не было видно полей водоёма (E-018).
        key = f"{hydronym.normalize_name(metric.name) or metric.id}|{metric.geom}"
        groups.setdefault(key, []).append(metric)

    out: list[WaterMetrics] = []
    for items in groups.values():
        if site_xy is not None and crs is not None:
            # Покрытие считается только для линейной воды: у озера «покрытие» — это периметр
            # замкнутого кольца, и правило выбирало дальний пруд вместо ближнего.
            items.sort(key=lambda m: (-_coverage_m(m.coords, site_xy, crs, radius_m)
                                      if is_watercourse(m) else 0.0,
                                      m.nearest_m))
        else:
            items.sort(key=lambda m: m.nearest_m)
        keep = items[0]
        if len(items) > 1:
            keep.pieces = len(items)
            shares = [m.hidden_share for m in items if m.hidden_share is not None]
            if shares:
                keep.hidden_share = max(shares)
                keep.visible = keep.hidden_share < 0.8
        out.append(keep)
    return out


def _resolve_tidal_by_tags(metrics: list[WaterMetrics], features: list[Feature], *,
                           crs: int | str, site_xy: np.ndarray) -> None:
    """Приливность по тегу OSM ближайшего куска той же реки.

    У склеенной линии теги могут расходиться (граница прилива проходит внутри): берём кусок,
    который ближе всего к участку, — он и описывает воду рядом с точкой.
    """
    for metric in metrics:
        if not is_watercourse(metric) or metric.tidal is not None or not metric.name:
            continue
        best = None
        for feature in features:
            if feature.name != metric.name or len(feature.coords) < 2:
                continue
            raw = (feature.tags.get("tidal") or "").lower()
            if raw not in {"yes", "no"}:
                continue
            pts = to_utm(feature.coords, crs) - site_xy
            dist = float(np.hypot(*pts.T).min())
            if best is None or dist < best[0]:
                best = (dist, raw)
        if best is not None:
            metric.tidal = best[1] == "yes"
            metric.tidal_source = "osm_tag"


def _resolve_tidal_by_reach(metrics: list[WaterMetrics], lat: float, lon: float, *,
                            crs: int | str, site_xy: np.ndarray, radius_m: int,
                            wide_radius_m: int = 30000, wide_payload: dict | None = None,
                            allow_network: bool = True) -> None:
    """Приливность по положению вдоль русла: граница прилива может лежать дальше 8 км.

    Для таких рек берётся более широкая выгрузка (кешируется), граница прилива ставится
    на русло, и сравниваются расстояния вдоль него. Сравнение по координате ошибается на
    изгибах: Кингстон западнее Теддингтона, но стоит выше него по течению.
    """
    pending = [m for m in metrics if is_watercourse(m) and m.tidal is None and m.name
               and tides.limit_entry(m.name)]
    if not pending:
        return
    if wide_payload is not None:
        wide = wide_payload
    else:
        try:
            wide = fetch_raw(lat, lon, wide_radius_m, allow_network=allow_network)
        except Exception:
            # Без выгрузки на 30 км границу прилива не поставить: пишем причину, а не молчим.
            # Только код: текст исключения тянет в пакет путь кеша, и два пакета перестают
            # сравниваться побитово на разных машинах.
            for metric in pending:
                metric.tidal_reason = "no_wide_payload"
            return
    # нужна линия, на которой лежат ОБА: участок и граница прилива.
    # Из линий одной реки выбираем ту, где участок ближе 2 км, а граница — ближайшая.
    wide_lines: dict[str, tuple[float, object]] = {}
    limit_xy_cache: dict[str, np.ndarray] = {}
    for line in merge_by_name(parse_features(wide)):
        if not line.name:
            continue
        pts_line = to_utm(line.coords, crs) - site_xy
        if len(pts_line) < 2:
            continue
        dist_site = float(np.hypot(*pts_line.T).min())
        if dist_site > 2000.0:
            continue
        entry_line = tides.limit_entry(line.name)
        if entry_line is None:
            continue
        if line.name not in limit_xy_cache:
            from pyproj import Transformer as _T
            limit_xy_cache[line.name] = np.array(_T.from_crs("EPSG:4326", geo.crs_str(crs), always_xy=True)
                                                 .transform(float(entry_line["limit_lon"]),
                                                            float(entry_line["limit_lat"]))) - site_xy
        limit_pt = limit_xy_cache[line.name]
        dist_limit = float(np.hypot(pts_line[:, 0] - limit_pt[0], pts_line[:, 1] - limit_pt[1]).min())
        current = wide_lines.get(line.name)
        if current is None or dist_limit < current[0]:
            wide_lines[line.name] = (dist_limit, line)
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(crs), always_xy=True)
    for metric in pending:
        line = (wide_lines.get(metric.name) or (None, None))[1]
        entry = tides.limit_entry(metric.name)
        if line is None or entry is None:
            metric.tidal_reason = "limit_not_on_line"
            continue
        pts = to_utm(line.coords, crs) - site_xy
        if len(pts) < 3:
            continue
        seg = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        site_idx = int(np.argmin(np.hypot(pts[:, 0], pts[:, 1])))
        limit_xy = np.array(tr.transform(float(entry["limit_lon"]), float(entry["limit_lat"]))) - site_xy
        d_limit = np.hypot(pts[:, 0] - limit_xy[0], pts[:, 1] - limit_xy[1])
        limit_idx = int(np.argmin(d_limit))
        if float(d_limit[limit_idx]) > 5000.0:
            metric.tidal_reason = "limit_far_from_line"
            continue
        # downstream_toward_end: по течению индекс растёт
        flow_end = (metric.flow_chosen or "downstream_toward_end") == "downstream_toward_end"
        tide = tides.status_by_distance(entry, cum[site_idx], cum[limit_idx], flow_toward_end=flow_end)
        metric.tidal = tide["tidal"]
        metric.tidal_source = tide["tidal_source"]
        metric.tidal_reason = None
