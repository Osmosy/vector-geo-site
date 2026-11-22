"""Сборка `WaterObject` из измерений и доля воды по секторам."""

from __future__ import annotations

import numpy as np

from .models import WaterMouth, WaterObject
from .water import WaterMetrics, to_utm


def to_objects(metrics: list[WaterMetrics]) -> list[WaterObject]:
    out: list[WaterObject] = []
    for m in metrics:
        mouth = None
        if m.water_mouth:
            # передаём словарь целиком: выборочная сборка уже теряла уверенность и enclosure
            mouth = WaterMouth(**{k: v for k, v in m.water_mouth.items() if not k.startswith("_")})
        out.append(WaterObject(
            id=m.id, name=m.name, kind=m.kind if m.kind in
            {"river", "stream", "canal", "lake", "sea", "dock", "culvert", "road"} else "lake",
            order=m.order, in_mountain=m.in_mountain, out_mountain=m.out_mountain,
            nearest_m=m.nearest_m, bend=m.bend, straight_at_site=m.straight_at_site,
            tidal=m.tidal, confluences=[{"with": c["with"], "dist_m": c["dist_m"],
                                         "mountain": c["mountain"]} for c in m.confluences],
            water_mouth=mouth, outflow_visible=m.outflow_visible,
            curvature_radius_m=m.curvature_radius_m,
            loop=m.loop, site_position=m.site_position,
            between_branches=m.between_branches,
            flow_ways_share=m.flow_ways_share,
            crossings_2km=m.crossings_2km,
            crossing_angle_deg=m.crossing_angle_deg,
            flow_dem_dir=m.flow_dem_dir, flow_osm_dir=m.flow_osm_dir, flow_agree=m.flow_agree,
            flow_chosen=m.flow_chosen, flow_rule=m.flow_rule, flow_confidence=m.flow_confidence,
            hidden_share=m.hidden_share, visible=m.visible, pieces=m.pieces,
            source=m.source,
        ))
    return out


def water_mask(features, *, epsg: int, site_xy: np.ndarray, grid, radius_px: int = 1) -> np.ndarray:
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
        pts = to_utm(f.coords, epsg)
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
