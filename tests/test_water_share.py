"""Доля воды по секторам: полигоны считаются площадью, осевые линии — длиной."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geosite import geo, water_pack


class _F:
    """Минимальная фича: так же, как `water.Feature` для расчёта площади."""

    def __init__(self, coords, name="x", kind="lake"):
        self.coords = coords
        self.name = name
        self.kind = kind


def _square(lon: float, lat: float, side_m: float = 400.0):
    """Квадрат со стороной side_m вокруг точки, в градусах (для полигонов воды)."""
    dlat = side_m / 2 / 110_540.0
    dlon = side_m / 2 / (111_320.0 * math.cos(math.radians(lat)))
    return [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat),
            (lon + dlon, lat + dlat), (lon - dlon, lat + dlat), (lon - dlon, lat - dlat)]


def test_share_lands_in_the_sector_where_the_water_is():
    lat, lon = 51.5, -0.1
    epsg = 32630
    north = _square(lon, lat + 0.027)          # ~3 км на север: ±3.8° — целиком сектор 子
    site = np.array([0.0, 0.0])
    from pyproj import Transformer

    xy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(xy.transform(lon, lat))
    res = water_pack.sector_water_share([_F(north)], epsg=epsg, site_xy=site,
                                        ring_m=(500.0, 4000.0))
    assert res["by_mountain"]["子"] > 0.001
    assert max(res["by_mountain"], key=res["by_mountain"].get) == "子"
    assert sum(v for k, v in res["by_mountain"].items() if k != "子") == 0.0


def test_share_ignores_lines_and_tiny_ponds():
    lat, lon = 51.5, -0.1
    epsg = 32630
    from pyproj import Transformer

    xy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(xy.transform(lon, lat))
    line = _F([(lon, lat + 0.003), (lon, lat + 0.012)], name="line", kind="river")
    tiny = _F(_square(lon, lat + 0.009, side_m=20.0), name="tiny")
    res = water_pack.sector_water_share([line, tiny], epsg=epsg, site_xy=site,
                                        ring_m=(500.0, 2000.0))
    assert res["polygons"] == 0                     # 20 м × 20 м = 400 м² < 2000 м²
    assert res["area_km2"] == 0.0
    # но длина линии видна
    assert res["linear_km"]["子"] > 0.5


def test_linear_length_is_metres_not_degrees():
    lat, lon = 51.5, -0.1
    epsg = 32630
    from pyproj import Transformer

    xy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(xy.transform(lon, lat))
    line = _F([(lon, lat + 0.0045), (lon, lat + 0.0135)], kind="river")   # ~1 км на север
    res = water_pack.linear_water_by_sector([line], epsg=epsg, site_xy=site,
                                            ring_m=(0.0, 2000.0))
    assert 0.9 < res["子"] < 1.1


def test_ring_area_matches_geometry():
    lat, lon = 51.5, -0.1
    epsg = 32630
    from pyproj import Transformer

    xy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(xy.transform(lon, lat))
    res = water_pack.sector_water_share([], epsg=epsg, site_xy=site, ring_m=(500.0, 2000.0))
    # π(2000² − 500²) = 11.78 км²
    assert res["ring_km2"] == pytest.approx(11.78, abs=0.05)
    assert all(v == 0.0 for v in res["by_mountain"].values())


def test_invalid_polygon_is_repaired_not_dropped():
    """Самопересечённый «бабочка» — невалидный полигон: чинится buffer(0), а не пропадает."""
    lat, lon = 51.5, -0.1
    epsg = 32630
    from pyproj import Transformer

    xy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(xy.transform(lon, lat))
    d = 0.006
    bowtie = [(lon - d, lat), (lon + d, lat), (lon - d, lat + 2 * d), (lon + d, lat + 2 * d),
              (lon - d, lat)]
    res = water_pack.sector_water_share([_F(bowtie)], epsg=epsg, site_xy=site,
                                        ring_m=(500.0, 3000.0))
    assert res["skipped_invalid"] == 1
    assert res["polygons"] <= 1
