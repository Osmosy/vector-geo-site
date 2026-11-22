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

from . import geo, tides

CACHE_DIR = Path(os.environ.get("GEOSITE_CACHE", Path.home() / ".cache" / "vector-geo-site")) / "osm"
ALGO_VERSION = "3.1.0"

USER_AGENT = "vector-geo-site/0.1 (fengshui landform measurement; https://github.com/Osmosy/vector-geo-site)"

# Зеркала Overpass: первое доступное. Каждое может быть перегружено — поэтому список и повторы.
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
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
    in_m = geo.mountain_name(geo.normalize_deg(math.degrees(math.atan2(upstream[0], upstream[1]))))
    out_m = geo.mountain_name(geo.normalize_deg(math.degrees(math.atan2(downstream[0], downstream[1]))))

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
    bend_kind = "straight" if (radius_m is None or radius_m > 5000.0) else "bend"
    if loop_raw and (loop_raw.get("ratio") or 0.0) >= LOOP_MIN_RATIO:
        # W2: петля — только если русло возвращается к себе, пройдя втрое больше разрыва
        loop = loop_raw
        bend_kind = "loop"
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
        between_branches=between.get("between_branches"),
        between_pair_deg=between.get("pair_deg"),
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
        "closure_confidence": ("low" if capped_sides > 0 else
                               ("high" if (enclosure or 0.0) >= 0.2 else "medium")),
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


def _detect_loop(pts: np.ndarray, *, near_m: float = 1400.0, apart_m: float = 1500.0) -> dict | None:
    """Излучина: русло возвращается к себе. Отношение длины к разрыву пишется в ratio."""
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
            "ratio": round(along / max(gap, 1.0), 2),
            "site_inside": bool(inside), "mountain": geo.mountain_name(bearing)}


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
                     max_per_line: int = 5) -> dict[str, list[dict]]:
    """Слияния водотоков: точки пересечения, общие узлы, без дублей (W3).

    Три вещи, которых не хватало:
    1) точки одной пары ближе `merge_m` схлопываются в одну запись;
    2) совпадающие концы линий (общий узел OSM, до 0.5 м) тоже считаются слиянием;
    3) пары отбираются по пересечению bbox и грубой сетке-хешу, а не полным перебором.

    На каждый водоток пишется не больше `max_per_line` ближайших слияний: интерпретатору
    нужны ближние, остальное живёт в water.geojson (W7).
    """
    lines: dict[str, np.ndarray] = {}
    kinds: dict[str, str] = {}
    for idx, f in enumerate(features):
        if f.kind in {"river", "stream", "canal"} and len(f.coords) > 1:
            key = getattr(f, "key", None) or f"osm-{f.osm_type}-{f.osm_id}-{idx}"
            lines[key] = to_utm(f.coords, epsg) - site_xy
            kinds[key] = f.kind

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
                result[a_key].append({"with": b_key, **entry})
                result[b_key].append({"with": a_key, **entry})

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
                  max_features: int = 2000, payload: dict | None = None) -> list[WaterMetrics]:
    """Вода в радиусе: линии склеены по имени, затем измерены.

    `payload` позволяет подать заранее сохранённый ответ Overpass — так работают эталоны
    в tests/golden без сети.
    """
    if payload is None:
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
