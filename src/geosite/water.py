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

from . import geo

CACHE_DIR = Path(os.environ.get("GEOSITE_CACHE", Path.home() / ".cache" / "vector-geo-site")) / "osm"
USER_AGENT = "vector-geo-site/0.1 (fengshui landform measurement; https://github.com/Osmosy/vector-geo-site)"

# Зеркала Overpass: первое доступное. Каждое может быть перегружено — поэтому список и повторы.
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

WATERWAY_KINDS = {"river", "stream", "canal", "drain", "ditch"}


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


def _cache_path(lat: float, lon: float, radius_m: int) -> Path:
    key = hashlib.sha256(f"{lat:.4f},{lon:.4f},{radius_m}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"water-{key}.json"


def fetch_raw(lat: float, lon: float, radius_m: int = 8000, *, allow_network: bool = True,
              retries: int = 3) -> dict:
    """Сырой ответ Overpass с кешем. Сеть трогается только если кеша нет."""
    path = _cache_path(lat, lon, radius_m)
    if path.exists():
        return json.loads(path.read_text())
    if not allow_network or os.environ.get("GEOSITE_OFFLINE") == "1":
        raise FileNotFoundError(f"нет кеша воды: {path}")

    query = _query(lat, lon, radius_m)
    last_error: Exception | None = None
    for attempt in range(retries):
        for mirror in MIRRORS:
            try:
                req = urllib.request.Request(
                    mirror, data=urllib.parse.urlencode({"data": query}).encode(),
                    headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as resp:
                    payload = json.loads(resp.read().decode())
                if "elements" in payload:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload))
                    return payload
                last_error = RuntimeError(f"неожиданный ответ: {str(payload)[:120]}")
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(2 + attempt * 3)
    raise RuntimeError(f"Overpass недоступен: {last_error}")


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
            out.append(Feature(el.get("id", 0), "way", tags.get("name"), kind, tags, coords, closed))
        elif el.get("type") == "relation" and el.get("members"):
            # члены отношения не склеиваем в один контур: иначе между ними появляются
            # хорды через полгорода и доля воды раздувается
            for idx, member in enumerate(el["members"]):
                geom = [(g["lon"], g["lat"]) for g in member.get("geometry") or [] if g]
                if len(geom) >= 2:
                    out.append(Feature(el.get("id", 0) * 1000 + idx, "relation", tags.get("name"),
                                       _kind_of(tags), tags, geom, True))
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


@dataclass
class MergedLine:
    key: str
    name: str | None
    kind: str
    coords: list[tuple[float, float]]
    ways: int
    hidden: bool


def merge_by_name(features: list[Feature], *, max_gap_m: float = 150.0) -> list[MergedLine]:
    """Склеить отрезки одной реки в одну линию: OSM хранит реку десятками ways.

    Подряд идущие куски сшиваются по ближайшим концам, если разрыв не больше max_gap_m;
    иначе линия остаётся отдельной. Анонимные куски (без имени) не склеиваются вовсе.
    """
    groups: dict[tuple[str, str], list[Feature]] = {}
    for f in features:
        if f.kind not in {"river", "stream", "canal", "lake", "sea", "dock"}:
            continue
        key = (f.name or f"#{f.osm_id}", f.kind)
        groups.setdefault(key, []).append(f)

    merged: list[MergedLine] = []
    for (name, kind), items in groups.items():
        pieces = [to_utm(f.coords, 4326) if False else f.coords for f in items]
        hidden = all(is_hidden(f.tags) for f in items)
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
                merged.append(MergedLine(f"osm-{(name or '').strip()}-{kind}", name, kind, chain,
                                         len(items) - len(rest), hidden))
                chain = list(rest.pop(0))
        merged.append(MergedLine(f"osm-{(name or '').strip()}-{kind}", name, kind, chain,
                                 len(items) - len(rest), hidden))
    return merged


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
    curvature_radius_m: float | None
    straight_at_site: bool
    confluences: list[dict]
    water_mouth: dict | None
    outflow_visible: bool | None
    flow_dem_dir: str | None
    flow_osm_dir: str | None
    flow_agree: bool | None
    tidal: bool | None
    source: str
    loop: dict | None = None
    between_branches: bool | None = None
    between_pair_deg: list[int] | None = None
    flow_ways_share: float | None = None


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

    # вход и выход из круга 2 км → горы 來水 / 去水
    cross = _crossings_with_circle(pts, in_radius_m)
    in_m = out_m = None
    if len(cross) >= 1:
        first, last = cross[0], cross[-1]
        in_m = geo.mountain_name(geo.normalize_deg(math.degrees(math.atan2(first[0], first[1]))))
        out_m = geo.mountain_name(geo.normalize_deg(math.degrees(math.atan2(last[0], last[1]))))
    else:
        a, b = pts[0], pts[-1]
        in_m = geo.mountain_name(geo.normalize_deg(math.degrees(math.atan2(a[0], a[1]))))
        out_m = geo.mountain_name(geo.normalize_deg(math.degrees(math.atan2(b[0], b[1]))))

    # изгиб: центр кривизны в точке nearest; участок внутри кривой = обнимает
    i = min(max(nearest_seg, 1), len(pts) - 2)
    center = _circumcenter(pts[i - 1], pts[i], pts[i + 1])
    if center is None:                      # ровный участок — берём окно шире
        lo, hi = max(1, i - 6), min(len(pts) - 2, i + 6)
        center = _circumcenter(pts[lo], pts[i], pts[hi])
    bend, radius_m = None, None
    if center is not None:
        radius_m = float(np.hypot(*(center - pts[i])))
        # вектор от русла к центру кривизны и от русла к участку (начало координат)
        to_center = center - pts[i]
        to_site = -pts[i]
        if np.hypot(*to_site) > 1.0 and np.hypot(*to_center) > 1.0:
            cos = float(np.dot(to_center, to_site) / (np.hypot(*to_center) * np.hypot(*to_site)))
            # участок с той же стороны, что центр кривизны → русло обнимает участок
            bend = "inner" if cos > 0 else "outer"
    straight = radius_m is None or radius_m > 5000.0

    # течение по DEM: сравнить высоты двух соседних точек русла
    flow_dem = None
    if grid is not None:
        z_a = grid.point_value(*(pts[i - 1] + site_xy))
        z_b = grid.point_value(*(pts[i + 1] + site_xy))
        if not (math.isnan(z_a) or math.isnan(z_b)):
            flow_dem = "downstream_toward_end" if z_b < z_a else "downstream_toward_start"

    # течение по OSM: направление отрисовки линии (соглашение OSM — по течению)
    flow_osm = "downstream_toward_end" if len(pts) > 1 else None

    # водные врата: где линия покидает круг 8 км
    mouth = None
    outer = _crossings_with_circle(pts, meso_radius_m)
    if outer and grid is not None:
        # из двух выходов воды берём тот, что ниже по рельефу: это и есть 去水
        if len(outer) >= 2:
            za = grid.point_value(*(outer[0] + site_xy))
            zb = grid.point_value(*(outer[-1] + site_xy))
            exit_pt = outer[0] if (not math.isnan(za) and (math.isnan(zb) or za < zb)) else outer[-1]
        else:
            exit_pt = outer[0]
        mouth = _water_mouth(grid, exit_pt, pts[i], site_xy)

    visible = None
    if mouth is not None and grid is not None:
        visible = _line_of_sight(grid, np.zeros(2), np.array([mouth.pop("_x"), mouth.pop("_y")]))

    loop = _detect_loop(pts) if len(pts) > 20 else None
    between = site_between_branches(pts) if len(pts) > 2 else {"between_branches": False}
    tags = getattr(feature, "tags", {}) or {}
    kind = "culvert" if (getattr(feature, "hidden", False) or is_hidden(tags)) else feature.kind

    return WaterMetrics(
        id=getattr(feature, "key", None) or f"osm-{feature.osm_type}-{feature.osm_id}",
        name=feature.name,
        kind=kind,
        order=None,
        nearest_m=round(nearest_m, 1),
        in_mountain=in_m,
        out_mountain=out_m,
        bend=bend,
        curvature_radius_m=None if radius_m is None else round(radius_m, 1),
        straight_at_site=bool(straight and nearest_m < in_radius_m),
        confluences=[],
        water_mouth=mouth,
        outflow_visible=visible,
        flow_dem_dir=flow_dem,
        flow_osm_dir=flow_osm,
        flow_agree=(flow_dem == flow_osm) if flow_dem and flow_osm else None,
        tidal=None,
        source="openstreetmap",
        loop=loop,
        between_branches=between.get("between_branches"),
        between_pair_deg=between.get("pair_deg"),
    )


def _point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    """Точка внутри замкнутого контура (правило чётности)."""
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


def _detect_loop(pts: np.ndarray, *, near_m: float = 1400.0, apart_m: float = 1500.0) -> dict | None:
    """Излучина: русло возвращается к себе, и где относительно этой петли стоит участок.

    Измеряется то, что видно на карте: минимальный разрыв между двумя частями русла,
    разнесёнными по длине не меньше apart_m, и лежит ли участок внутри этой петли
    (Айл-оф-Догс: участок на полуострове — внутри; Вестминстер и Сити — снаружи).
    """
    if len(pts) < 20:
        return None
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    stride = max(1, len(pts) // 500)
    best = None
    for i in range(0, len(pts), stride):
        for j in range(i + 2, len(pts), stride):
            along = cum[j] - cum[i]
            if along < apart_m:
                continue
            d = float(np.hypot(*(pts[i] - pts[j])))
            if d <= near_m and (best is None or d < best[0]):
                best = (d, along, i, j)
    if best is None:
        return None
    gap, along, i, j = best
    ring = np.vstack([pts[i:j + 1], pts[i:i + 1]])
    inside = _point_in_poly(0.0, 0.0, ring)
    mid = (pts[i] + pts[j]) / 2.0
    bearing = geo.normalize_deg(math.degrees(math.atan2(mid[0], mid[1])))
    return {"gap_m": round(gap, 1), "along_m": round(along, 1),
            "site_inside": bool(inside), "mountain": geo.mountain_name(bearing)}


def site_between_branches(pts: np.ndarray, *, near_m: float = 1500.0, tol_deg: float = 30.0) -> dict:
    """Стоит ли участок между двумя ветвями русла (внутри излучины).

    Для каждого из 24 секторов берётся ближайшая точка русла, попавшая в сектор. Если
    есть две почти противоположные стороны (разница азимутов 180°±30°) с расстоянием
    меньше near_m, участок лежит между ветвями — это и есть «внутри петли».
    """
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
    between = False
    pair = None
    for i in sectors:
        for delta in (10, 11, 12, 13, 14):   # 150°…210°
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


def _water_mouth(grid, exit_pt: np.ndarray, flow_pt: np.ndarray, site_xy: np.ndarray) -> dict:
    """Ширина долины и высоты бортов поперёк течения в точке выхода воды."""
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
    half_widths, sides = [], []
    for sign in (1, -1):
        prev = z0
        width = 0.0
        for step in range(1, 61):          # до 3 км в каждую сторону
            d = step * 50.0
            z = grid.point_value(*(site_xy + exit_pt + perp * sign * d))
            if math.isnan(z):
                break
            width = d
            if z - z0 > 20.0:              # борт поднялся на 20 м — это склон долины
                break
            prev = z
        half_widths.append(width)
        sides.append(max(0.0, prev - z0) if half_widths[-1] else 0.0)
    width = half_widths[0] + half_widths[1]
    hill_left = float(max(grid.point_value(*(site_xy + exit_pt + perp * d))
                          for d in (half_widths[0],) if not math.isnan(grid.point_value(*(site_xy + exit_pt + perp * d))))
                      - z0) if half_widths[0] else 0.0
    hill_right = float(max(grid.point_value(*(site_xy + exit_pt - perp * d))
                           for d in (half_widths[1],) if not math.isnan(grid.point_value(*(site_xy + exit_pt - perp * d))))
                       - z0) if half_widths[1] else 0.0
    bearing = geo.normalize_deg(math.degrees(math.atan2(exit_pt[0], exit_pt[1])))
    return {
        "mountain": geo.mountain_name(bearing),
        "width_m": round(width, 1),
        # эвристика «узости ворот»: 1 — узкие (ширина → 0), 0 — шире 4 км
        "closure_index": round(max(0.0, 1.0 - min(width, 4000.0) / 4000.0), 2),
        "left_hill_m": round(hill_left, 1),
        "right_hill_m": round(hill_right, 1),
        "_x": float(x_abs),
        "_y": float(y_abs),
    }


def _line_of_sight(grid, start: np.ndarray, end: np.ndarray, *, eye_m: float = 2.0,
                   step_m: float = 50.0) -> bool:
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
    n = int(length / step_m)
    for k in range(1, n):
        t = k / n
        x, y = start[0] + dx * t, start[1] + dy * t
        z = grid.point_value(x, y)
        if math.isnan(z):
            continue
        line_h = h0 + (h1 - h0) * t
        if z > line_h + 1.0:            # рельеф выше луча зрения
            return False
    return True


def find_confluences(features: list[Feature], *, site_xy: np.ndarray, epsg: int,
                     radius_m: float = 8000.0) -> dict[str, list[dict]]:
    """Слияния: пересечения линий разных водотоков (грубо, по отрезкам)."""
    lines = {}
    for idx, f in enumerate(features):
        if f.kind in {"river", "stream", "canal"} and len(f.coords) > 1:
            key = getattr(f, "key", None) or f"osm-{f.osm_type}-{f.osm_id}-{idx}"
            lines[key] = to_utm(f.coords, epsg) - site_xy
    result: dict[str, list[dict]] = {k: [] for k in lines}
    keys = list(lines)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = lines[keys[i]], lines[keys[j]]
            for p, q in zip(a[:-1], a[1:]):
                for r, s in zip(b[:-1], b[1:]):
                    hit = _seg_intersect(p, q, r, s)
                    if hit is None:
                        continue
                    dist = float(np.hypot(*hit))
                    if dist > radius_m:
                        continue
                    bearing = geo.normalize_deg(math.degrees(math.atan2(hit[0], hit[1])))
                    entry = {"dist_m": round(dist, 1), "mountain": geo.mountain_name(bearing)}
                    result[keys[i]].append({"with": keys[j], **entry})
                    result[keys[j]].append({"with": keys[i], **entry})
    return result


def _seg_intersect(p, q, r, s):
    r_ = q - p
    s_ = s - r
    denom = r_[0] * s_[1] - r_[1] * s_[0]
    if abs(denom) < 1e-12:
        return None
    t = ((r[0] - p[0]) * s_[1] - (r[1] - p[1]) * s_[0]) / denom
    u = ((r[0] - p[0]) * r_[1] - (r[1] - p[1]) * r_[0]) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p + r_ * t
    return None


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
        if f.kind not in {"river", "stream", "canal"} or len(f.coords) < 2:
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
                  max_features: int = 2000) -> list[WaterMetrics]:
    """Вода в радиусе: линии склеены по имени, затем измерены."""
    payload = fetch_raw(lat, lon, radius_m, allow_network=allow_network)
    features = parse_features(payload)[:max_features]
    merged = merge_by_name(features)
    # слияния ищем только у воды, которая вообще рядом с участком: иначе O(n²) по всему городу
    near = []
    for m in merged:
        pts_near = to_utm(m.coords, epsg) - site_xy
        if float(np.hypot(*pts_near.T).min()) < 3000.0:
            near.append(m)
    confluences = find_confluences(near, site_xy=site_xy, epsg=epsg, radius_m=radius_m)
    flow_share = flow_convention_share(features, grid, epsg)
    out: list[WaterMetrics] = []
    for m in merged:
        metrics = measure_feature(m, site_xy=site_xy, lat=lat, lon=lon, epsg=epsg, grid=grid,
                                  meso_radius_m=radius_m)
        if metrics is None:
            continue
        metrics.confluences = confluences.get(metrics.id, [])[:4]
        if flow_share is not None and metrics.kind in {"river", "stream", "canal"}:
            metrics.flow_agree = flow_share >= 0.6
            metrics.flow_ways_share = flow_share
        out.append(metrics)
    out.sort(key=lambda m: m.nearest_m)
    return out
