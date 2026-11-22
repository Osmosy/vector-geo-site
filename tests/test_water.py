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


# --- W1: водные врата ---

def _valley_grid(slope_deg: float = 30.0, n: int = 141, res: float = 50.0):
    from affine import Affine
    from geosite.dem import DemGrid

    xs = (np.arange(n) + 0.5) * res - n * res / 2
    ys = (np.arange(n) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys)
    z = 100.0 + np.abs(xx) * np.tan(np.radians(slope_deg))
    return DemGrid(z.astype("float32"), Affine(res, 0, 0, 0, -res, n * res), "EPSG:32630", res)


def test_mouth_v_valley_high_closure():
    g = _valley_grid(30.0)
    m = water._water_mouth(g, np.array([0.0, 0.0]), np.array([0.0, -100.0]), np.array([g.data.shape[1] * 25.0, g.data.shape[0] * 25.0]))
    assert m["closure_index"] is not None and m["closure_index"] >= 0.7
    assert m["width_capped"] is False


def test_mouth_flat_plane_is_null():
    from affine import Affine
    from geosite.dem import DemGrid

    res, n = 50.0, 141
    flat = DemGrid(np.full((n, n), 100.0, dtype="float32"), Affine(res, 0, 0, 0, -res, n * res),
                   "EPSG:32630", res)
    m = water._water_mouth(flat, np.array([0.0, 0.0]), np.array([0.0, -100.0]),
                           np.array([n * res / 2, n * res / 2]))
    assert m["closure_index"] is None
    assert m["closure_reason"] == "flat_valley"
    assert m["width_capped"] is True and m["capped_sides"] == 2


def test_mouth_one_side_capped_is_low_confidence():
    g = _valley_grid(10.0)
    m = water._water_mouth(g, np.array([0.0, 0.0]), np.array([0.0, -100.0]),
                           np.array([g.data.shape[1] * 25.0, g.data.shape[0] * 25.0]))
    if m["capped_sides"] > 0:
        assert m["closure_confidence"] == "low"


# --- W2: излучина ---

def _circle_with_neck(radius: float = 1000.0, neck_gap: float = 300.0, steps: int = 240):
    ang = np.linspace(0, 2 * np.pi, steps)
    pts = np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])
    return pts


def test_loop_detected_with_ratio():
    pts = _circle_with_neck()
    loop = water._detect_loop(pts)
    assert loop is not None
    assert loop["ratio"] >= 5.0


def test_loop_absent_on_gentle_sine():
    x = np.linspace(0, 20000, 400)
    pts = np.column_stack([x - 10000, 300 * np.sin(x / 2000)])
    loop = water._detect_loop(pts)
    assert loop is None or loop["ratio"] < 3.0


def test_loop_ratio_threshold_constant():
    assert water.LOOP_MIN_RATIO == 3.0


# --- W3: слияния ---

def test_y_junction_gives_single_confluence():
    from pyproj import Transformer
    from geosite.water import to_utm

    main = make("Main", [(-0.100, 51.500), (-0.095, 51.500)], osm_id=1)
    trib = make("Trib", [(-0.0975, 51.505), (-0.0975, 51.500)], osm_id=2)
    epsg = 32630
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(tr.transform(-0.0975, 51.50))
    res = water.find_confluences([main, trib], site_xy=site, epsg=epsg)
    main_key = [k for k in res if "Main" in k][0]
    assert len(res[main_key]) == 1
    assert res[main_key][0]["kind"] in {"confluence", "canal_junction", "culvert_outfall"}


def test_double_crossing_gives_two_confluences():
    from pyproj import Transformer

    a = make("A", [(-0.11, 51.50), (-0.09, 51.50)], osm_id=11)
    b = make("B", [(-0.105, 51.494), (-0.095, 51.506), (-0.093, 51.494)], osm_id=22)
    epsg = 32630
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(tr.transform(-0.10, 51.50))
    res = water.find_confluences([a, b], site_xy=site, epsg=epsg)
    a_key = [k for k in res if "A-" in k][0]
    distances = sorted(item["dist_m"] for item in res[a_key])
    assert len(distances) == 2
    assert distances[0] != distances[1]


def test_near_duplicate_intersections_are_merged():
    from pyproj import Transformer

    # две почти совпадающие точки пересечения (зигзаг в 20 м) — одно слияние
    a = make("A", [(-0.110, 51.500), (-0.090, 51.500)], osm_id=31)
    b = make("B", [(-0.1000, 51.495), (-0.1000, 51.505)], osm_id=32)
    epsg = 32630
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(tr.transform(-0.10, 51.50))
    res = water.find_confluences([a, b], site_xy=site, epsg=epsg)
    a_key = [k for k in res if "A-" in k][0]
    assert len(res[a_key]) == 1


def test_canal_junction_kind():
    from pyproj import Transformer

    river = make("R", [(-0.110, 51.500), (-0.090, 51.500)], osm_id=41)
    canal = make("C", [(-0.0980, 51.496), (-0.0980, 51.504)], kind="canal", osm_id=42)
    epsg = 32630
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(tr.transform(-0.10, 51.50))
    res = water.find_confluences([river, canal], site_xy=site, epsg=epsg)
    kinds = {item["kind"] for items in res.values() for item in items}
    assert "canal_junction" in kinds


def test_confluence_cap_per_line():
    from pyproj import Transformer

    river = make("R", [(-0.12, 51.500), (-0.06, 51.500)], osm_id=51)
    tribs = [make(f"T{i}", [(-0.115 + i * 0.01, 51.495), (-0.115 + i * 0.01, 51.505)], osm_id=60 + i)
             for i in range(8)]
    epsg = 32630
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site = np.array(tr.transform(-0.10, 51.50))
    res = water.find_confluences([river] + tribs, site_xy=site, epsg=epsg)
    river_key = [k for k in res if "R-" in k][0]
    assert len(res[river_key]) <= 5
    assert res[river_key][0]["dist_m"] <= res[river_key][-1]["dist_m"]


def test_alias_names_merge():
    """R2/E-010: «River X» и «The X» — один водоток, даже с разрывом 20 м."""
    a = make("River Loddon", [(-0.10, 51.50), (-0.0995, 51.50)])
    b = make("The Loddon", [(-0.0993, 51.50), (-0.0985, 51.50)], kind="stream")
    merged = water.merge_by_name([a, b])
    assert len(merged) == 1, [m.name for m in merged]
    assert merged[0].ways == 2


def test_hidden_share_hides_from_visible():
    """R2: водоток, идущий в трубе на 90% длины, не считается видимым."""
    covered = make("Culverted Brook", [(-0.10, 51.50), (-0.0982, 51.50)],
                   tags={"tunnel": "culvert"})
    open_part = make("Culverted Brook", [(-0.0981, 51.50), (-0.0979, 51.50)])
    merged = water.merge_by_name([covered, open_part])
    assert merged[0].hidden_share is not None
    assert merged[0].hidden_share > 0.8
    site = water.to_utm([(-0.099, 51.50)], 32630)[0]
    metric = water.measure_feature(merged[0], site_xy=site, lat=51.50, lon=-0.099,
                                   epsg=32630, grid=None)
    assert metric.visible is False
