"""Сборка `WaterObject` из измерений и доля воды по секторам."""

from __future__ import annotations

import math

import numpy as np

from .models import WaterMouth, WaterObject
from .water import WaterMetrics, to_utm


def to_objects(metrics: list[WaterMetrics]) -> list[WaterObject]:
    from .water import is_watercourse

    out: list[WaterObject] = []
    for m in metrics:
        mouth = None
        if m.water_mouth:
            # передаём словарь целиком: выборочная сборка уже теряла уверенность и enclosure
            mouth = WaterMouth(**{k: v for k, v in m.water_mouth.items() if not k.startswith("_")})
        area = not is_watercourse(m)
        out.append(WaterObject(
            id=m.id, name=m.name, names=list(getattr(m, "names", []) or []),
            seasonal=getattr(m, "seasonal", None),
            kind=m.kind if m.kind in
            {"river", "stream", "canal", "lake", "sea", "dock", "culvert", "road"} else "lake",
            order=None if area else m.order,
            discharge_m3s=None if area else m.discharge_m3s,
            hydro_distance_m=None if area else m.hydro_distance_m,
            in_mountain=m.in_mountain, out_mountain=m.out_mountain,
            nearest_m=m.nearest_m, bend=m.bend, straight_at_site=m.straight_at_site,
            # W3/E-042: вид изгиба считается на слое метрик, но в объект не передавался —
            # поле было в схеме и пустым во всех артефактах
            bend_kind=(m.bend_kind if m.bend_kind in {"bend", "loop", "straight"} else None),
            nearest_azimuth_deg=m.nearest_azimuth_deg, nearest_mountain=m.nearest_mountain,
            area_m2=m.area_m2, site_inside=m.site_inside, riverbank_of=m.riverbank_of,
            dock_channel_of=m.dock_channel_of,
            tidal=m.tidal, tidal_source=m.tidal_source,
            # E-028: причина нужна и у положительного ответа — «берег в 420 м» объясняет
            # не отсутствие прилива, а его источник
            tidal_reason=m.tidal_reason,
            confluences=[{"with": c["with"], "dist_m": c["dist_m"],
                                         "mountain": c["mountain"]} for c in m.confluences],
            waterbody_links=list(m.waterbody_links),
            geom=getattr(m, "geom", "line"),
            water_mouth=mouth, outflow_visible=m.outflow_visible,
            curvature_radius_m=m.curvature_radius_m,
            loop=m.loop, site_position=m.site_position,
            between_branches=m.between_branches,
            flow_ways_share=m.flow_ways_share,
            crossings_2km=None if area else m.crossings_2km,
            crossing_angle_deg=m.crossing_angle_deg,
            length_2km_m=m.length_2km_m,
            same_mountain_reason=m.same_mountain_reason,
            flow_dem_dir=m.flow_dem_dir, flow_osm_dir=m.flow_osm_dir, flow_agree=m.flow_agree,
            flow_chosen=m.flow_chosen, flow_rule=m.flow_rule, flow_confidence=m.flow_confidence,
            hidden_share=m.hidden_share, visible=m.visible, pieces=m.pieces,
            source=m.source,
        ))
    return out


def water_mask(features, *, crs: int | str, site_xy: np.ndarray, grid, radius_px: int = 1) -> np.ndarray:
    """Растр воды по сетке рельефа: залитые водоёмы и линии водотоков.

    Многие полигоны OSM невалидны (самопересечения) — rasterio молча их пропускает,
    поэтому геометрия чинится shapely.buffer(0). Линии расширяются на radius_px:
    река шириной 200 м при шаге 90 м иначе не даёт целого пикселя.
    """
    from rasterio.features import rasterize
    from shapely.geometry import LineString, Polygon, mapping

    shapes = []
    for f in features:
        if len(f.coords) < 2:
            continue
        pts = to_utm(f.coords, crs)
        closed = len(pts) > 3 and abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9
        geom = Polygon(pts) if closed else LineString(pts)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        shapes.append((mapping(geom), 1))
    if not shapes:
        return np.zeros(grid.data.shape, dtype=bool)
    mask = rasterize(shapes, out_shape=grid.data.shape, transform=grid.transform,
                     fill=0, dtype="uint8").astype(bool)
    for _ in range(radius_px):
        grown = mask.copy()
        grown[1:, :] |= mask[:-1, :]
        grown[:-1, :] |= mask[1:, :]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return mask


def sector_water_share(mask: np.ndarray, frame, *, ring_km: tuple[float, float]) -> dict[str, float]:
    """Доля пикселей воды в каждом из 24 секторов для кольца."""
    from . import geo
    r0, r1 = ring_km[0] * 1000.0, ring_km[1] * 1000.0
    ring = (frame.dist >= r0) & (frame.dist < r1) & frame.valid
    out: dict[str, float] = {}
    for idx, name in enumerate(geo.MOUNTAINS):
        center = idx * 15.0
        lo, hi = geo.normalize_deg(center - 7.5), geo.normalize_deg(center + 7.5)
        sector = (frame.bearing >= lo) & (frame.bearing < hi) if lo <= hi else (frame.bearing >= lo) | (frame.bearing < hi)
        sel = ring & sector
        total = int(sel.sum())
        out[name] = round(float(mask[sel].mean()), 3) if total else 0.0
    return out


def polygon_features(features) -> list:
    """Только замкнутые полигоны воды: линии русла площади не имеют, и расширять их нельзя."""
    out = []
    for f in features:
        coords = getattr(f, "coords", None) or []
        if len(coords) < 4:
            continue
        if abs(coords[0][0] - coords[-1][0]) > 1e-9 or abs(coords[0][1] - coords[-1][1]) > 1e-9:
            continue
        out.append(f)
    return out


def sector_water_share(features, *, crs: int | str, site_xy: np.ndarray, ring_m: tuple[float, float],
                       min_area_m2: float = 2000.0,
                       lat: float | None = None, lon: float | None = None) -> dict:
    """Доля площади воды в каждом из 24 секторов кольца — по полигонам OSM, без растяжения.

    Считается аналитически в UTM: полигон пересекается с сектором-клином, площадь делится на
    площадь клина. Раньше маска воды растрировалась на сетке 90 м и расширялась на пиксель —
    и доля выходила 40–60 % там, где воды несколько процентов: у маленьких прудов периметр
    больше площади, и один пиксель расширения утраивал «воду».

    Возвращает `{"by_mountain": {...}, "area_km2": ..., "ring_km2": ..., "polygons": n,
    "skipped_invalid": n}`; `share` — доля воды в секторе, 0.0 — вода есть, но не в нём.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from . import geo

    r0, r1 = float(ring_m[0]), float(ring_m[1])
    polys = []
    invalid = 0
    for f in polygon_features(features):
        pts = to_utm([(lon, lat) for lon, lat in f.coords], crs) - site_xy
        holes = [to_utm([(lon, lat) for lon, lat in hole], crs) - site_xy
                 for hole in (getattr(f, "holes", None) or [])]
        poly = Polygon(pts, holes) if holes else Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
            invalid += 1
        if poly.is_empty or poly.area < min_area_m2:
            continue
        polys.append(poly)
    shares: dict[str, float] = {name: 0.0 for name in geo.MOUNTAINS}
    linear = linear_water_by_sector(features, crs=crs, site_xy=site_xy, ring_m=(r0, r1),
                                    lat=lat, lon=lon)
    basis = ("polygons: water=*, natural=water, landuse=reservoir/basin; "
             "осевые линии — в linear_km")
    if not polys:
        return {"by_mountain": shares, "area_km2": 0.0,
                "ring_km2": round(_ring_polygon(r0, r1).area / 1e6, 3), "polygons": 0,
                "skipped_invalid": invalid, "linear_km": linear, "basis": basis}
    water = unary_union(polys)
    ring_area = 0.0
    for idx, name in enumerate(geo.MOUNTAINS):
        lo = idx * 15.0 - 7.5
        arc = [math.radians(lo + 15.0 * k / 6.0) for k in range(7)]
        outer = [(r1 * math.sin(a), r1 * math.cos(a)) for a in arc]
        if r0 > 0:
            inner = [(r0 * math.sin(a), r0 * math.cos(a)) for a in reversed(arc)]
            wedge = Polygon(outer + inner)
        else:
            wedge = Polygon([(0.0, 0.0)] + outer)
        if not wedge.is_valid:
            wedge = wedge.buffer(0)
        piece = water.intersection(wedge)
        wedge_area = wedge.area
        ring_area += wedge_area
        if wedge_area > 0 and not piece.is_empty:
            shares[name] = round(min(1.0, piece.area / wedge_area), 4)
    return {"by_mountain": shares,
            "area_km2": round(water.intersection(_ring_polygon(r0, r1)).area / 1e6, 3),
            "ring_km2": round(ring_area / 1e6, 3), "polygons": len(polys),
            "skipped_invalid": invalid, "linear_km": linear, "basis": basis}


def linear_water_by_sector(features, *, crs: int | str, site_xy: np.ndarray,
                           lat: float | None = None, lon: float | None = None,
                           ring_m: tuple[float, float]) -> dict[str, float]:
    """Длина линейной воды (река осевой, ручей) в каждом секторе кольца, км.

    Площади такие линии не дают — OSM не хранит ширину, — но их наличие в секторе важно:
    в Лондоне Темза нарисована осевой, и доля полигональной воды в её секторах выглядит
    нулевой. Поэтому рядом с долей публикуется длина.
    """
    from . import geo

    r0, r1 = float(ring_m[0]), float(ring_m[1])
    offset = geo.convergence_deg(lat, lon, crs) if lat is not None and lon is not None else 0.0
    out: dict[str, float] = {name: 0.0 for name in geo.MOUNTAINS}
    for f in features:
        coords = getattr(f, "coords", None) or []
        if len(coords) < 2 or f in polygon_features([f]):
            continue
        pts = to_utm(coords, crs) - site_xy
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg <= 0:
                continue
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            dist = math.hypot(mx, my)
            if not (r0 <= dist < r1):
                continue
            bearing = (math.degrees(math.atan2(mx, my)) + offset) % 360.0
            out[geo.mountain_name(bearing)] += seg / 1000.0
    return {k: round(v, 3) for k, v in out.items()}


def _ring_polygon(r0: float, r1: float):
    from shapely.geometry import Point

    ring = Point(0.0, 0.0).buffer(r1, resolution=64)
    if r0 > 0:
        ring = ring.difference(Point(0.0, 0.0).buffer(r0, resolution=64))
    return ring
