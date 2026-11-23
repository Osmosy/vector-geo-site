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

from . import geo, hydronym, tides

CACHE_DIR = Path(os.environ.get("GEOSITE_CACHE", Path.home() / ".cache" / "vector-geo-site")) / "osm"
ALGO_VERSION = "6.1.0"

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


@dataclass
class Feature:
    osm_id: int
    osm_type: str
    name: str | None
    kind: str                      # river / stream / canal / lake / sea / dock / culvert / road
    tags: dict = field(default_factory=dict)
    coords: list[tuple[float, float]] = field(default_factory=list)   # (lon, lat)
    closed: bool = False


def _query(lat: float, lon: float, radius_m: int) -> str:
    r = int(radius_m)
    return (
        "[out:json][timeout:90];"
        f'(way(around:{r},{lat},{lon})["waterway"~"^(river|stream|canal|drain|ditch|riverbank)$"];'
        f'way(around:{r},{lat},{lon})["natural"="water"];'
        f'relation(around:{r},{lat},{lon})["natural"="water"];'
        f'way(around:{r},{lat},{lon})["waterway"="dam"];'
        ");out geom tags;"
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


RELATION_KINDS = "^(river|stream|canal)$"
RELATION_ROLES = {"", "main_stream", "side_stream", "waterway"}


def _relation_query(lat: float, lon: float, radius_m: int) -> str:
    r = int(radius_m)
    return ("[out:json][timeout:120];"
            f'rel(around:{r},{lat},{lon})["waterway"~"{RELATION_KINDS}"]["name"];out body;')


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
                  retries: int = 3, max_relations: int = 8, max_ways: int = 600,
                  batch: int = 40, keep_factor: float = 2.5) -> list[dict]:
    """Куски рек из отношений OSM — с именем отношения как запасным именем куска.

    Отношение `waterway=river` описывает реку целиком: его члены стыкуются узлами, поэтому
    устье притока не теряется. Геометрия тянется пачками по `batch` ways — `out geom` по
    отношению целиком отдаёт 0.5–3 МБ и валит перегруженное зеркало в 502.
    Куски дальше `keep_factor * radius` от участка выбрасываются: они не нужны ни метрикам,
    ни слияниям, а память и склейка на 3000 км стоят секунд.
    """
    path = _cache_path(lat, lon, radius_m, kind="relations")
    if path.exists():
        return json.loads(path.read_text()).get("elements", [])
    if not allow_network or os.environ.get("GEOSITE_OFFLINE") == "1":
        return []

    try:
        rel = _ask(_relation_query(lat, lon, radius_m), retries=retries)
        relations: list[tuple[int, list[int]]] = []        # (rel_id, его ways)
        rel_name: dict[int, str] = {}
        rel_of: dict[int, int] = {}
        for el in rel.get("elements", [])[:max_relations]:
            name = _name_of(el.get("tags") or {})
            if not name:
                continue
            ways: list[int] = []
            for member in el.get("members", []):
                if member.get("type") != "way" or (member.get("role") or "") not in RELATION_ROLES:
                    continue
                ref = int(member.get("ref") or 0)
                if not ref or ref in rel_name:
                    continue
                rel_name[ref] = name
                rel_of[ref] = int(el.get("id") or 0)
                ways.append(ref)
                if len(ways) >= max_ways:
                    break
            if ways:
                relations.append((int(el.get("id") or 0), ways))
        if not relations:
            return []

        keep_m = keep_factor * float(radius_m)
        out: list[dict] = []
        try:
            payload = _ask(_relations_ways_query([rel_id for rel_id, _ in relations]), retries=retries)
        except Exception:
            payload = {"elements": []}
            for rel_id, ways in relations:
                for start in range(0, len(ways), batch):
                    payload["elements"].extend(
                        _ask(_ways_query(ways[start:start + batch]), retries=retries).get("elements", []))
        for el in payload.get("elements", []):
            geometry = el.get("geometry") or []
            if len(geometry) < 2 or not _near_site(geometry, lat, lon, keep_m):
                continue
            tags = dict(el.get("tags") or {})
            tags.setdefault("name", rel_name.get(int(el.get("id") or 0), ""))
            tags["_rel"] = str(rel_of.get(int(el.get("id") or 0), ""))
            out.append({"type": "way", "id": el.get("id"), "tags": tags,
                        "geometry": geometry})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"elements": out}))
        return out
    except Exception:
        # без отношений вода всё равно измеряется, просто устье притока может не найтись
        return []


def is_hidden(tags: dict) -> bool:
    """Вода под землёй: culvert, covered, underground — для «скрытой воды» канона."""
    if (tags.get("tunnel") or "").lower() == "culvert":
        return True
    if (tags.get("covered") or "").lower() in {"yes", "culvert"}:
        return True
    return (tags.get("location") or "").lower() == "underground"


def _kind_of(tags: dict) -> str:
    natural = tags.get("natural")
    waterway = tags.get("waterway")
    if natural == "water":
        if tags.get("water") in {"river", "canal"} or tags.get("waterway") == "riverbank":
            return "river"
        if tags.get("water") == "dock" or tags.get("landuse") == "dock":
            return "dock"
        if tags.get("place") == "sea" or tags.get("water") == "sea":
            return "sea"
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


def _name_of(tags: dict) -> str | None:
    """Имя водотока: основное, иначе английское, иначе альтернативное."""
    for key in ("name", "name:en", "alt_name", "name:ru"):
        value = (tags.get(key) or "").strip()
        if value:
            return value.split(";")[0].strip() or None
    return None


def parse_features(payload: dict) -> list[Feature]:
    out: list[Feature] = []
    for el in payload.get("elements", []):
        tags = el.get("tags", {}) or {}
        if el.get("type") == "way" and el.get("geometry"):
            coords = [(g["lon"], g["lat"]) for g in el["geometry"] if g]
            kind = _kind_of(tags)
            if not coords or kind == "other":
                continue
            closed = len(coords) > 3 and abs(coords[0][0] - coords[-1][0]) < 1e-9 and abs(coords[0][1] - coords[-1][1]) < 1e-9
            out.append(Feature(el.get("id", 0), "way", _name_of(tags), kind, tags, coords, closed))
        elif el.get("type") == "relation" and el.get("members"):
            # члены отношения не склеиваем в один контур: иначе между ними появляются
            # хорды через полгорода и доля воды раздувается
            for idx, member in enumerate(el["members"]):
                geom = [(g["lon"], g["lat"]) for g in member.get("geometry") or [] if g]
                if len(geom) >= 2:
                    closed = (len(geom) > 3
                              and abs(geom[0][0] - geom[-1][0]) < 1e-9
                              and abs(geom[0][1] - geom[-1][1]) < 1e-9)
                    out.append(Feature(el.get("id", 0) * 1000 + idx, "relation", _name_of(tags),
                                       _kind_of(tags), tags, geom, closed))
    return out


# ---------- геометрия ----------

def to_utm(coords: list[tuple[float, float]], epsg: int) -> np.ndarray:
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
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

    def __post_init__(self) -> None:
        if not self.uid:
            self.uid = self.key


# одна река бывает размечена и river, и stream: тип куска берётся по старшему, а не по самому
# длинному way — иначе один старый кусок waterway=stream переименовывает Волгу в ручей
KIND_RANK = {"river": 5, "canal": 4, "stream": 3, "lake": 2, "dock": 1, "sea": 0}


def merge_by_name(features: list[Feature], *, max_gap_m: float = 150.0) -> list[MergedLine]:
    """Склеить отрезки одной реки в одну линию: OSM хранит реку десятками ways.

    Подряд идущие куски сшиваются по ближайшим концам, если разрыв не больше max_gap_m;
    иначе линия остаётся отдельной. Анонимные куски (без имени) не склеиваются вовсе.
    """
    groups: dict[tuple[str, str], list[Feature]] = {}
    for f in features:
        if f.kind not in {"river", "stream", "canal", "lake", "sea", "dock"}:
            continue
        # ключ склейки — нормализованное имя: «River Fleet» и «The Fleet» — один водоток
        # тип в ключ не входит: одна река в OSM бывает и river, и stream на разных участках
        key = (hydronym.normalize_name(f.name) or f"#{f.osm_id}", "")
        groups.setdefault(key, []).append(f)

    merged: list[MergedLine] = []
    for (name, kind), items in groups.items():
        longest = max(items, key=lambda f: len(f.coords))
        display = longest.name or name
        kind = max((f.kind for f in items), key=lambda k: KIND_RANK.get(k, 0))
        pieces = [f.coords for f in items]
        hidden = all(is_hidden(f.tags) for f in items)
        total_len = sum(_line_len(f.coords) for f in items)
        hidden_len = sum(_line_len(f.coords) for f in items if hydronym.is_hidden(f.tags))
        hidden_share = None if total_len <= 0 else round(hidden_len / total_len, 2)
        # жадная сшивка: начинаем с самого длинного куска
        pieces = sorted(pieces, key=len, reverse=True)
        chain = list(pieces[0])
        rest = pieces[1:]
        while rest:
            tail = chain[-1]
            best = None
            for idx, piece in enumerate(rest):
                for rev in (False, True):
                    cand = piece[::-1] if rev else piece
                    d = _haversine_deg(tail, cand[0])
                    if best is None or d < best[0]:
                        best = (d, idx, cand)
            d, idx, cand = best
            if d <= max_gap_m:
                chain.extend(cand)
                rest.pop(idx)
            else:
                merged.append(MergedLine(f"osm-{name}-{kind}", display, kind, chain,
                                         len(items) - len(rest), hidden, hidden_share))
                chain = list(rest.pop(0))
        merged.append(MergedLine(f"osm-{name}-{kind}", display, kind, chain,
                                 len(items) - len(rest), hidden, hidden_share))
    # ключ куска неуникален: у одной реки их несколько, и словарь слияний молча терял все,
    # кроме последнего. uid различает куски, key остаётся именем реки для потребителя.
    for idx, line in enumerate(merged):
        line.uid = f"{line.key}#{idx}"
    return merged


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
    straight_at_site: bool
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
    loop: dict | None = None
    site_position: str | None = None
    hidden_share: float | None = None
    visible: bool | None = None
    pieces: int = 1
    between_branches: bool | None = None
    between_pair_deg: list[int] | None = None
    crossings_2km: int = 0
    crossing_angle_deg: float | None = None
    flow_ways_share: float | None = None
    coords: list[tuple[float, float]] = field(default_factory=list)


def measure_feature(feature, *, site_xy: np.ndarray, lat: float, lon: float, epsg: int,
                    grid, in_radius_m: float = 2000.0, meso_radius_m: float = 8000.0) -> WaterMetrics | None:
    if len(feature.coords) < 2:
        return None
    pts = to_utm(feature.coords, epsg) - site_xy          # координаты относительно участка
    dists = _segment_distances(pts, np.zeros(2)) if len(pts) > 1 else np.array([np.hypot(*pts[0])])
    nearest_seg = int(np.argmin(dists))
    nearest_m = float(dists[nearest_seg])
    if nearest_m > meso_radius_m:
        return None

    tags = getattr(feature, "tags", {}) or {}
    hidden = getattr(feature, "hidden", False) or is_hidden(tags)
    kind = "culvert" if hidden else feature.kind

    # --- приливность (W5): тег OSM, иначе таблица известных границ ---
    from pyproj import Transformer
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    nearest_abs = pts[nearest_seg] + site_xy
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
    in_bearing = geo.normalize_deg(math.degrees(math.atan2(upstream[0], upstream[1])))
    out_bearing = geo.normalize_deg(math.degrees(math.atan2(downstream[0], downstream[1])))
    in_m = geo.mountain_name(in_bearing)
    out_m = geo.mountain_name(out_bearing)
    crossing_angle = geo.angle_distance(in_bearing, out_bearing)

    # изгиб: центр кривизны в точке nearest; участок внутри кривой = обнимает
    i = min(max(nearest_seg, 1), len(pts) - 2)
    center = _circumcenter(pts[i - 1], pts[i], pts[i + 1])
    if center is None:                      # ровный участок — берём окно шире
        lo, hi = max(1, i - 6), min(len(pts) - 2, i + 6)
        center = _circumcenter(pts[lo], pts[i], pts[hi])
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
        mouth = _water_mouth(grid, exit_pt, pts[i], site_xy)

    visible = None
    if mouth is not None and grid is not None:
        visible = _line_of_sight(grid, np.zeros(2), np.array([mouth.pop("_x"), mouth.pop("_y")]))

    loop_raw = _detect_loop(pts) if len(pts) > 20 else None
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

    return WaterMetrics(
        id=getattr(feature, "key", None) or f"osm-{feature.osm_type}-{feature.osm_id}",
        name=feature.name,
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
        source="openstreetmap",
        loop=loop,
        site_position=site_position,
        between_branches=between.get("between_branches"),
        between_pair_deg=between.get("pair_deg"),
        crossings_2km=len(cross),
        crossing_angle_deg=round(crossing_angle, 1),
        coords=list(feature.coords),
    )


def _water_mouth(grid, exit_pt: np.ndarray, flow_pt: np.ndarray, site_xy: np.ndarray,
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
    if math.isnan(z0):
        return None

    half_widths, hills, capped_sides = [], [], 0
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
        if stop_reason != "slope":
            capped_sides += 1
        half_widths.append(half_width)
        hills.append(top)

    width = half_widths[0] + half_widths[1]
    hill_left, hill_right = hills[0], hills[1]
    mean_hill = (hill_left + hill_right) / 2.0
    half = width / 2.0
    enclosure = None
    closure_index = None
    reason = None
    if capped_sides >= 2:
        reason = "flat_valley"
    elif half <= 1.0:
        enclosure = 99.0
        closure_index = 1.0
    else:
        enclosure = mean_hill / half
        closure_index = round(enclosure / (1.0 + enclosure), 3)

    bearing = geo.normalize_deg(math.degrees(math.atan2(exit_pt[0], exit_pt[1])))
    return {
        "mountain": geo.mountain_name(bearing),
        "width_m": round(width, 1),
        "left_hill_m": round(hill_left, 1),
        "right_hill_m": round(hill_right, 1),
        "width_capped": capped_sides > 0,
        "capped_sides": capped_sides,
        "enclosure": None if enclosure is None else round(enclosure, 3),
        "closure_index": closure_index,
        "closure_reason": reason,
        # П7: в застройке DSM даёт крыши вместо бортов; очень малая замкнутость означает,
        # что настоящих бортов долины мы не нашли, — это low, а не «средне»
        "closure_confidence": ("low" if (capped_sides > 0 or (enclosure is not None and enclosure < 0.05))
                               else ("high" if (enclosure or 0.0) >= 0.2 else "medium")),
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


def _detect_loop(pts: np.ndarray, *, near_m: float = 2000.0, apart_m: float = 1500.0,
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
    bearing = geo.normalize_deg(math.degrees(math.atan2(mid[0], mid[1])))

    is_loop = (ratio >= min_ratio and turn >= min_turn_deg and area >= area_k * gap * gap)
    return {
        "gap_m": round(gap, 1), "along_m": round(along, 1), "ratio": round(ratio, 2),
        "turn_deg": round(turn, 1), "area_m2": round(area), "area_k": round(area / (gap * gap), 3),
        "mountain": geo.mountain_name(bearing),
        "is_loop": bool(is_loop),
        "site_position": _site_position(ring, mid, far),
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


def find_confluences(features, *, site_xy: np.ndarray, epsg: int,
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
    for idx, f in enumerate(features):
        if f.kind in {"river", "stream", "canal"} and len(f.coords) > 1:
            uid = (getattr(f, "uid", None) or getattr(f, "key", None)
                   or f"osm-{f.osm_type}-{f.osm_id}-{idx}")
            full = to_utm(f.coords, epsg) - site_xy
            pieces = clip_near(full, clip_m) or []
            if not pieces:
                continue
            lines[uid] = np.vstack(pieces) if len(pieces) == 1 else max(pieces, key=len)
            kinds[uid] = f.kind
            labels[uid] = getattr(f, "key", None) or uid

    boxes = {key: _bbox(pts) for key, pts in lines.items()}
    result: dict[str, list[dict]] = {key: [] for key in lines}

    keys = list(lines)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key, b_key = keys[i], keys[j]
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
                bearing = geo.normalize_deg(math.degrees(math.atan2(point[0], point[1])))
                entry = {"dist_m": round(dist, 1), "mountain": geo.mountain_name(bearing),
                         "kind": _confluence_kind(kinds[a_key], kinds[b_key])}
                result[a_key].append({"with": labels[b_key], **entry})
                result[b_key].append({"with": labels[a_key], **entry})

    for key, items in result.items():
        items.sort(key=lambda item: item["dist_m"])
        result[key] = items[:max_per_line]
    return result


def flow_convention_share(features: list[Feature], grid, epsg: int, *,
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
        if f.kind not in {"river", "stream"} or len(f.coords) < 2:
            continue
        pts = to_utm([f.coords[0], f.coords[-1]], epsg)
        z0 = grid.point_value(*pts[0])
        z1 = grid.point_value(*pts[1])
        if math.isnan(z0) or math.isnan(z1):
            continue
        total += 1
        if z1 < z0 - min_drop_m:
            agree += 1
    return None if total == 0 else round(agree / total, 2)


def measure_water(lat: float, lon: float, *, epsg: int, grid=None, site_xy: np.ndarray,
                  radius_m: int = 8000, allow_network: bool = True,
                  max_features: int = 2000, payload: dict | None = None,
                  wide_payload: dict | None = None,
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
    features = parse_features(payload)[:max_features]
    merged = merge_by_name(features)

    window_m = max(window_factor * float(radius_m), window_min_m)
    from pyproj import Transformer
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    chunks: list[MergedLine] = []
    for line in merged:
        pts = to_utm(line.coords, epsg) - site_xy
        pieces = clip_near(pts, window_m) if len(pts) > 1 else []
        if not pieces:
            continue
        if len(pieces) == 1 and len(pieces[0]) == len(pts):
            chunks.append(line)
            continue
        for idx, piece in enumerate(pieces):
            lons, lats = inv.transform(piece[:, 0] + site_xy[0], piece[:, 1] + site_xy[1])
            chunks.append(MergedLine(line.key, line.name, line.kind,
                                     list(zip(lons, lats)), line.ways, line.hidden,
                                     line.hidden_share, uid=f"{line.uid}-{idx}"))

    # слияния ищем только у воды, которая вообще рядом с участком: иначе O(n²) по всему городу
    near = []
    for line in chunks:
        pts_near = to_utm(line.coords, epsg) - site_xy
        if float(np.hypot(*pts_near.T).min()) < 3000.0:
            near.append(line)
    confluences = find_confluences(near, site_xy=site_xy, epsg=epsg, radius_m=radius_m,
                                   clip_m=float(radius_m))
    # одна река — несколько кусков: слияние, найденное на любом из них, относится к реке
    by_name: dict[str, list[dict]] = {}
    for line in near:
        key = hydronym.normalize_name(line.name) or line.key
        by_name.setdefault(key, []).extend(confluences.get(line.uid, []))

    flow_share = flow_convention_share(features, grid, epsg)
    out: list[WaterMetrics] = []
    for m in chunks:
        metrics = measure_feature(m, site_xy=site_xy, lat=lat, lon=lon, epsg=epsg, grid=grid,
                                  meso_radius_m=radius_m)
        if metrics is None:
            continue
        key = hydronym.normalize_name(metrics.name) or metrics.id
        items = _dedupe_confluences(by_name.get(key, []))
        metrics.confluences = items[:4]
        if flow_share is not None and metrics.kind in {"river", "stream", "canal"}:
            metrics.flow_agree = flow_share >= 0.6
            metrics.flow_ways_share = flow_share
        out.append(metrics)
    out = _one_entry_per_name(out)
    out.sort(key=lambda m: m.nearest_m)
    _resolve_tidal_by_tags(out, features, epsg=epsg, site_xy=site_xy)
    _resolve_tidal_by_reach(out, lat, lon, epsg=epsg, site_xy=site_xy, radius_m=radius_m,
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


def _one_entry_per_name(metrics: list[WaterMetrics]) -> list[WaterMetrics]:
    """Одна река — одна запись, даже если русло в OSM разбито на несоприкасающиеся куски.

    Записи группируются по нормализованному имени и типу. Остаётся кусок, ближайший к участку;
    доля подземного участка берётся максимальной по кускам (если часть реки в трубе — это
    важно сказать), а число кусков пишется в `pieces`.
    """
    groups: dict[str, list[WaterMetrics]] = {}
    for metric in metrics:
        key = hydronym.normalize_name(metric.name) or metric.id
        groups.setdefault(key, []).append(metric)

    out: list[WaterMetrics] = []
    for items in groups.values():
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
                           epsg: int, site_xy: np.ndarray) -> None:
    """Приливность по тегу OSM ближайшего куска той же реки.

    У склеенной линии теги могут расходиться (граница прилива проходит внутри): берём кусок,
    который ближе всего к участку, — он и описывает воду рядом с точкой.
    """
    for metric in metrics:
        if metric.tidal is not None or not metric.name:
            continue
        best = None
        for feature in features:
            if feature.name != metric.name or len(feature.coords) < 2:
                continue
            raw = (feature.tags.get("tidal") or "").lower()
            if raw not in {"yes", "no"}:
                continue
            pts = to_utm(feature.coords, epsg) - site_xy
            dist = float(np.hypot(*pts.T).min())
            if best is None or dist < best[0]:
                best = (dist, raw)
        if best is not None:
            metric.tidal = best[1] == "yes"
            metric.tidal_source = "osm_tag"


def _resolve_tidal_by_reach(metrics: list[WaterMetrics], lat: float, lon: float, *,
                            epsg: int, site_xy: np.ndarray, radius_m: int,
                            wide_radius_m: int = 30000, wide_payload: dict | None = None,
                            allow_network: bool = True) -> None:
    """Приливность по положению вдоль русла: граница прилива может лежать дальше 8 км.

    Для таких рек берётся более широкая выгрузка (кешируется), граница прилива ставится
    на русло, и сравниваются расстояния вдоль него. Сравнение по координате ошибается на
    изгибах: Кингстон западнее Теддингтона, но стоит выше него по течению.
    """
    pending = [m for m in metrics if m.tidal is None and m.name and tides.limit_entry(m.name)]
    if not pending:
        return
    if wide_payload is not None:
        wide = wide_payload
    else:
        try:
            wide = fetch_raw(lat, lon, wide_radius_m, allow_network=allow_network)
        except Exception:
            return
    # нужна линия, на которой лежат ОБА: участок и граница прилива.
    # Из линий одной реки выбираем ту, где участок ближе 2 км, а граница — ближайшая.
    wide_lines: dict[str, tuple[float, object]] = {}
    limit_xy_cache: dict[str, np.ndarray] = {}
    for line in merge_by_name(parse_features(wide)):
        if not line.name:
            continue
        pts_line = to_utm(line.coords, epsg) - site_xy
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
            limit_xy_cache[line.name] = np.array(_T.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
                                                 .transform(float(entry_line["limit_lon"]),
                                                            float(entry_line["limit_lat"]))) - site_xy
        limit_pt = limit_xy_cache[line.name]
        dist_limit = float(np.hypot(pts_line[:, 0] - limit_pt[0], pts_line[:, 1] - limit_pt[1]).min())
        current = wide_lines.get(line.name)
        if current is None or dist_limit < current[0]:
            wide_lines[line.name] = (dist_limit, line)
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    for metric in pending:
        line = (wide_lines.get(metric.name) or (None, None))[1]
        entry = tides.limit_entry(metric.name)
        if line is None or entry is None:
            continue
        pts = to_utm(line.coords, epsg) - site_xy
        if len(pts) < 3:
            continue
        seg = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        site_idx = int(np.argmin(np.hypot(pts[:, 0], pts[:, 1])))
        limit_xy = np.array(tr.transform(float(entry["limit_lon"]), float(entry["limit_lat"]))) - site_xy
        d_limit = np.hypot(pts[:, 0] - limit_xy[0], pts[:, 1] - limit_xy[1])
        limit_idx = int(np.argmin(d_limit))
        if float(d_limit[limit_idx]) > 5000.0:
            continue
        # downstream_toward_end: по течению индекс растёт
        flow_end = (metric.flow_chosen or "downstream_toward_end") == "downstream_toward_end"
        tide = tides.status_by_distance(entry, cum[site_idx], cum[limit_idx], flow_toward_end=flow_end)
        metric.tidal = tide["tidal"]
        metric.tidal_source = tide["tidal_source"]
