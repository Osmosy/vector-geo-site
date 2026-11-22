"""Вода: склейка линий, скрытая вода, слияния, излучины."""

import math

import numpy as np
import pytest

from geosite import geo, water
from geosite.water import Feature


def make(name, coords, kind="river", tags=None, osm_id=1):
    f = Feature(osm_id, "way", name, kind, tags or {}, coords, False)
    f.key = f"osm-{name}-{kind}-{osm_id}"
    return f


def test_hidden_water_detection():
    assert water.is_hidden({"tunnel": "culvert"}) is True
    assert water.is_hidden({"covered": "yes"}) is True
    assert water.is_hidden({"location": "underground"}) is True
    assert water.is_hidden({"waterway": "river"}) is False


def test_merge_joins_contiguous_pieces():
    a = make("R", [(-0.10, 51.50), (-0.099, 51.50)])
    b = make("R", [(-0.0991, 51.50), (-0.098, 51.50)])
    merged = water.merge_by_name([a, b])
    assert len(merged) == 1
    assert merged[0].ways == 2
    assert len(merged[0].coords) == 4


def test_merge_keeps_distant_pieces_apart():
    a = make("R", [(-0.10, 51.50), (-0.099, 51.50)])
    b = make("R", [(-0.05, 51.50), (-0.049, 51.50)])   # 3.5 км разрыв
    merged = water.merge_by_name([a, b])
    assert len(merged) == 2


def test_merge_does_not_join_different_rivers():
    a = make("A", [(-0.10, 51.50), (-0.099, 51.50)])
    b = make("B", [(-0.0991, 51.50), (-0.098, 51.50)])
    assert len(water.merge_by_name([a, b])) == 2


def test_site_between_branches_true():
    # две параллельные ветви на ±600 м по оси X
    north = np.array([[x, 600.0] for x in range(-1000, 1001, 50)], dtype=float)
    south = np.array([[x, -600.0] for x in range(-1000, 1001, 50)], dtype=float)
    pts = np.vstack([north, south])
    r = water.site_between_branches(pts)
    assert r["between_branches"] is True


def test_site_between_branches_false():
    # одна ветвь с юга — участок не между ветвями
    south = np.array([[x, -500.0] for x in range(-1000, 1001, 50)], dtype=float)
    r = water.site_between_branches(south)
    assert r["between_branches"] is False


def test_detect_loop_on_u_shape():
    left = np.array([[0.0, y] for y in range(0, 3000, 100)])
    top = np.array([[x, 3000.0] for x in range(0, 1000, 100)])
    right = np.array([[1000.0, y] for y in range(3000, 0, -100)])
    pts = np.vstack([left, top[1:], right[1:]])
    loop = water._detect_loop(pts)
    assert loop is not None
    assert loop["gap_m"] < 1100        # ветви в 1 км друг от друга
    assert loop["along_m"] > 1500


def test_detect_loop_none_on_straight():
    pts = np.array([[x, 0.0] for x in range(0, 10000, 100)])
    assert water._detect_loop(pts) is None


def test_segment_intersection():
    p, q = np.array([0.0, 0.0]), np.array([10.0, 10.0])
    r, s_ = np.array([0.0, 10.0]), np.array([10.0, 0.0])
    hit = water._seg_intersect(p, q, r, s_)
    assert hit is not None
    assert hit == pytest.approx([5.0, 5.0])


def test_confluences_found():
    a = make("A", [(-0.10, 51.50), (-0.095, 51.50)], osm_id=11)
    b = make("B", [(-0.0975, 51.495), (-0.0975, 51.505)], osm_id=22)
    epsg = 32630
    from pyproj import Transformer
    from geosite.water import to_utm
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site_xy = np.array(tr.transform(-0.0975, 51.50))
    res = water.find_confluences([a, b], site_xy=site_xy, epsg=epsg, radius_m=5000)
    ids = list(res)
    assert res[ids[0]] and res[ids[1]]
    assert res[ids[0]][0]["dist_m"] < 200


@pytest.mark.skipif(not (water.CACHE_DIR / "water-" ).parent.exists(), reason="нет кеша OSM")
def test_london_thames_metrics():
    """Интеграционно: Темза у Вестминстера — близко, кривая, участок у банка."""
    from pyproj import Transformer

    lat, lon = 51.4995, -0.1278
    epsg = geo.utm_epsg(lat, lon)
    try:
        payload = water.fetch_raw(lat, lon, 8000, allow_network=False)
    except FileNotFoundError:
        pytest.skip("нет кеша воды для Вестминстера")
    merged = water.merge_by_name(water.parse_features(payload))
    thames = max([m for m in merged if m.name and "Thames" in m.name], key=lambda m: len(m.coords))
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site_xy = np.array(tr.transform(lon, lat))
    metric = water.measure_feature(thames, site_xy=site_xy, lat=lat, lon=lon, epsg=epsg, grid=None)
    assert metric is not None
    assert 150 < metric.nearest_m < 800
    assert metric.curvature_radius_m is not None
    assert metric.kind in {"river", "culvert"}
