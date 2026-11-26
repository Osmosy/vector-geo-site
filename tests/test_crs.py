"""G2/G4: выбор сетки, сходимость меридианов, линия перемены дат, компас, пустоты DEM."""

from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine

from geosite import climate, dem, geo, terrain

LONDON = (51.5138, -0.0984)
SYDNEY = (-33.85, 151.21)
SUVA = (-18.14, 178.44)


def test_small_site_stays_in_utm():
    """8 км внутри зоны — UTM: он привычен и не мешает сравнивать с внешними данными."""
    choice = geo.grid_crs(*SYDNEY, radius_km=8.0)
    assert choice["kind"] == "utm" and choice["reason"] is None
    assert choice["crs"] == "EPSG:32756"


def test_radius_over_50_km_goes_aeqd():
    choice = geo.grid_crs(*SUVA, radius_km=60.0)
    assert choice["kind"] == "aeqd" and choice["reason"] == "radius>50km"
    assert choice["crs"].startswith("+proj=aeqd") and "+lat_0=-18.14" in choice["crs"]


def test_zone_boundary_goes_aeqd():
    """Лондон стоит в 0.1° от границы зон 30/31 — круг 8 км её пересекает."""
    assert geo.crosses_zone_boundary(-0.0984, 8.0, 51.5)
    assert geo.grid_crs(*LONDON, radius_km=8.0)["kind"] == "aeqd"
    assert geo.grid_crs(*LONDON, radius_km=8.0)["reason"] == "zone_boundary"
    # а в середине зоны тот же радиус остаётся в UTM
    assert geo.grid_crs(51.5, 8.0, 8.0)["kind"] == "utm"


def test_polar_uses_aeqd():
    """Выше 84° с. ш. и ниже 80° ю. ш. — только азимутальная: UTM там не определён."""
    assert geo.grid_crs(85.0, 20.0, 8.0)["reason"] == "polar"
    assert geo.grid_crs(-81.0, 0.0, 8.0)["reason"] == "polar"
    assert geo.grid_crs(70.0, 20.0, 8.0)["kind"] == "utm"


def test_antimeridian_is_detected():
    assert geo.crosses_antimeridian(179.99, 8.0, -18.0)
    assert geo.crosses_antimeridian(-179.99, 8.0, 20.0)
    assert not geo.crosses_antimeridian(178.44, 8.0, -18.14)
    assert geo.grid_crs(-18.0, 179.99, 8.0)["reason"] == "antimeridian"


def test_split_bbox_halves_the_query():
    """Сува у линии: bbox не может тянуться за 180°, поэтому он режется на две части."""
    assert geo.split_bbox(179.5, 180.3) == [(179.5, 180.0), (-180.0, -179.7)]
    assert geo.split_bbox(-180.4, -179.6) == [(-180.0, -179.6), (179.6, 180.0)]
    assert geo.split_bbox(10.0, 20.0) == [(10.0, 20.0)]


def test_bbox_parts_do_not_leave_the_globe():
    parts = dem.bbox_parts(-18.14, 179.99, 8.0)
    assert len(parts) == 2
    for _, _, west, east in parts:
        assert -180.0 <= west <= east <= 180.0


def test_convergence_matches_measured_rotation():
    """У Лондона сетка UTM повёрнута на 2.27° — это седьмая часть горы, и её видно числами."""
    gamma = geo.convergence_deg(51.5138, -0.0984, 32630)
    assert gamma == pytest.approx(2.27, abs=0.02)     # участок восточнее меридиана зоны (-3°)
    # азимутальная проекция с центром в участке отдаёт истинный север
    assert geo.convergence_deg(51.5138, -0.0984, geo.AEQD.format(lat=51.5138, lon=-0.0984)) == 0.0


def test_bearing_in_utm_is_corrected_to_true_north():
    """Точка строго на север: в UTM угол сетки не 0°, после поправки — 0°."""
    lat, lon = LONDON
    north = (lat + 0.01, lon)
    true_azimuth = geo.azimuth_geodesic(lat, lon, *north)
    assert true_azimuth == pytest.approx(0.0, abs=0.01)

    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
    x0, y0 = tr.transform(lon, lat)
    x1, y1 = tr.transform(north[1], north[0])
    raw = geo.normalize_deg(math.degrees(math.atan2(x1 - x0, y1 - y0)))
    assert raw == pytest.approx(357.73, abs=0.02)          # на 2.27° меньше истинного
    fixed = geo.azimuth_xy(x1 - x0, y1 - y0, geo.convergence_deg(lat, lon, 32630))
    assert geo.angle_distance(fixed, 0.0) < 0.02


def test_aeqd_azimuth_is_exact_without_correction():
    lat, lon = LONDON
    tr_x = geo.AEQD.format(lat=lat, lon=lon)
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", tr_x, always_xy=True)
    x0, y0 = tr.transform(lon, lat)
    for dlon, dlat in ((0.0, 0.05), (0.07, 0.0), (0.05, -0.05)):
        x1, y1 = tr.transform(lon + dlon, lat + dlat)
        assert geo.angle_distance(geo.azimuth_xy(x1 - x0, y1 - y0, 0.0),
                                  geo.azimuth_geodesic(lat, lon, lat + dlat, lon + dlon)) < 0.05


def test_target_grid_is_centred_and_stable():
    transform, width, height, x0, y0 = dem.target_grid(52.3731, 4.8922, 8.0,
                                                      crs=32631, resolution_m=30.0)
    assert width == height and width == pytest.approx(2 * 8 * 1.02 * 1000 / 30, rel=0.01)
    centre_x = transform.c + width * transform.a / 2
    centre_y = transform.f + height * transform.e / 2
    assert centre_x == pytest.approx(x0, abs=30.0)
    assert centre_y == pytest.approx(y0, abs=30.0)


def test_compass_is_unreliable_near_the_magnetic_pole():
    near_pole = climate.declination(84.0, -120.0, year=2026.5)
    assert near_pole["horizontal_nt"] < climate.HORIZONTAL_MIN_NT
    assert near_pole["confidence"] == "low"
    london = climate.declination(51.5, -0.1, year=2026.5)
    assert london["horizontal_nt"] > 15000 and london["confidence"] == "high"


def _grid(values: np.ndarray, res: float = 100.0) -> dem.DemGrid:
    return dem.DemGrid(values.astype("float32"), Affine(res, 0.0, -values.shape[1] * res / 2,
                                                        0.0, -res, values.shape[0] * res / 2),
                       "EPSG:32630", res)


def test_voids_are_counted_and_do_not_become_zero():
    """Пустоты в кольце: пишем долю, а высоты — null; ноль читался бы как «плоско» (G4.4)."""
    data = np.zeros((81, 81), dtype="float32")
    data[:40, :] = np.nan              # север у кольца — пустота (море)
    grid = _grid(data)
    frame = terrain.polar_frame(grid, 0.0, 0.0, offset_deg=0.0)
    sectors = terrain.mountain_sectors(grid, frame, 0.0)
    north = next(s for s in sectors if s.name == "子")
    ring0 = north.rings[0]
    assert ring0["void_share"] > 0.5
    assert ring0["max_rel_elev_m"] is None or ring0["max_rel_elev_m"] == 0.0
    south = next(s for s in sectors if s.name == "午")
    assert south.rings[0]["void_share"] == 0.0
    assert south.rings[0]["max_rel_elev_m"] == 0.0


def test_negative_elevation_is_allowed():
    """I-09: ниже уровня моря — это факт (польдер, впадина), а не ошибка данных."""
    grid = _grid(np.full((81, 81), -4.0, dtype="float32"))
    frame = terrain.polar_frame(grid, 0.0, 0.0, offset_deg=0.0)
    z0 = float(grid.point_value(0.0, 0.0))
    assert z0 == pytest.approx(-4.0, abs=0.1)
    sectors = terrain.mountain_sectors(grid, frame, z0)
    near = [r for s in sectors for r in s.rings if r["r_km"][1] <= 2.0]
    assert near and all(r["max_rel_elev_m"] == pytest.approx(0.0, abs=0.2) for r in near)
