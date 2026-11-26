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
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(epsg), always_xy=True)
    site_xy = np.array(tr.transform(-0.0975, 51.50))
    res = water.find_confluences([a, b], site_xy=site_xy, crs=epsg, radius_m=5000)
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
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(epsg), always_xy=True)
    site_xy = np.array(tr.transform(lon, lat))
    metric = water.measure_feature(thames, site_xy=site_xy, lat=lat, lon=lon, crs=epsg, grid=None)
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


def test_mouth_capped_side_has_no_zero_height():
    """I-12: упёршаяся сторона не знает высоту борта — там null, а не 0 (E-025).

    Ноль читается как «борт ровно нулевой», то есть как измерение. На плоском месте борт
    просто не нашли в пределах 3 км, и честный ответ — отсутствие значения.
    """
    from affine import Affine
    from geosite.dem import DemGrid

    res, n = 50.0, 141
    flat = DemGrid(np.full((n, n), 100.0, dtype="float32"), Affine(res, 0, 0, 0, -res, n * res),
                   "EPSG:32630", res)
    m = water._water_mouth(flat, np.array([0.0, 0.0]), np.array([0.0, -100.0]),
                           np.array([n * res / 2, n * res / 2]))
    assert m["left_hill_m"] is None and m["right_hill_m"] is None
    assert m["enclosure"] is None


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
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(epsg), always_xy=True)
    site = np.array(tr.transform(-0.0975, 51.50))
    res = water.find_confluences([main, trib], site_xy=site, crs=epsg)
    main_key = [k for k in res if "Main" in k][0]
    assert len(res[main_key]) == 1
    assert res[main_key][0]["kind"] in {"confluence", "canal_junction", "culvert_outfall"}


def test_double_crossing_gives_two_confluences():
    from pyproj import Transformer

    a = make("A", [(-0.11, 51.50), (-0.09, 51.50)], osm_id=11)
    b = make("B", [(-0.105, 51.494), (-0.095, 51.506), (-0.093, 51.494)], osm_id=22)
    epsg = 32630
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(epsg), always_xy=True)
    site = np.array(tr.transform(-0.10, 51.50))
    res = water.find_confluences([a, b], site_xy=site, crs=epsg)
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
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(epsg), always_xy=True)
    site = np.array(tr.transform(-0.10, 51.50))
    res = water.find_confluences([a, b], site_xy=site, crs=epsg)
    a_key = [k for k in res if "A-" in k][0]
    assert len(res[a_key]) == 1


def test_canal_junction_kind():
    from pyproj import Transformer

    river = make("R", [(-0.110, 51.500), (-0.090, 51.500)], osm_id=41)
    canal = make("C", [(-0.0980, 51.496), (-0.0980, 51.504)], kind="canal", osm_id=42)
    epsg = 32630
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(epsg), always_xy=True)
    site = np.array(tr.transform(-0.10, 51.50))
    res = water.find_confluences([river, canal], site_xy=site, crs=epsg)
    kinds = {item["kind"] for items in res.values() for item in items}
    assert "canal_junction" in kinds


def test_confluence_cap_per_line():
    from pyproj import Transformer

    river = make("R", [(-0.12, 51.500), (-0.06, 51.500)], osm_id=51)
    tribs = [make(f"T{i}", [(-0.115 + i * 0.01, 51.495), (-0.115 + i * 0.01, 51.505)], osm_id=60 + i)
             for i in range(8)]
    epsg = 32630
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(epsg), always_xy=True)
    site = np.array(tr.transform(-0.10, 51.50))
    res = water.find_confluences([river] + tribs, site_xy=site, crs=epsg)
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
                                   crs=32630, grid=None)
    assert metric.visible is False


def _circle_with_neck(radius_m=1200.0, neck_m=250.0, step_deg=6.0):
    """Кольцо с перешейком: полный оборот и короткий вход. Возвращает точки в метрах."""
    pts = []
    angle = 0.0
    while angle <= 360.0:
        a = math.radians(angle)
        pts.append((radius_m * math.sin(a), radius_m * (1 - math.cos(a))))
        angle += step_deg
    # перешеек: от конца кольца к началу
    pts.append((neck_m / 2, -neck_m / 2))
    return np.array(pts)


def test_loop_circle_neck_positions():
    """R4: петля одна и та же, положение участка — разное."""
    ring = _circle_with_neck()
    inside = water._detect_loop(ring - np.array([0.0, 1200.0]))    # точка в центре кольца
    outside = water._detect_loop(ring - np.array([0.0, -600.0]))   # точка под перешейком
    assert inside is not None and outside is not None
    assert inside["is_loop"] is True, inside
    assert outside["is_loop"] is True, outside
    assert inside["site_position"] == "inside"
    assert outside["site_position"].startswith("outside")
    assert inside["turn_deg"] >= 180.0
    assert inside["area_k"] >= 0.05


# --- R5: река из отношения OSM ---

def test_merge_kind_priority_river_over_stream():
    """Один старый кусок waterway=stream не переименовывает реку в ручей."""
    big = make("Волга", [(-0.100 + i * 0.001, 51.50) for i in range(30)], kind="stream")
    part = make("Волга", [(-0.07005, 51.50), (-0.065, 51.50)], kind="river")
    merged = water.merge_by_name([big, part])
    assert len(merged) == 1
    assert merged[0].kind == "river"


def test_merge_gives_unique_uid_per_chunk():
    """Куски одной реки получают разные uid: иначе словарь слияний теряет все, кроме одного."""
    a = make("R", [(-0.10, 51.50), (-0.099, 51.50)], osm_id=1)
    b = make("R", [(-0.06, 51.50), (-0.059, 51.50)], osm_id=2)     # 2.8 км разрыв
    merged = water.merge_by_name([a, b])
    assert len(merged) == 2
    assert merged[0].key == merged[1].key
    assert merged[0].uid != merged[1].uid


def test_confluence_found_on_second_chunk_of_same_river():
    """Слияние ищется у всех кусков реки, а не только у последнего в словаре."""
    from pyproj import Transformer

    epsg = 32630
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(epsg), always_xy=True)
    site = np.array(tr.transform(-0.100, 51.500))

    # река разорвана надвое: ближний кусок у участка, дальний — в 4 км восточнее
    near_chunk = make("R", [(-0.106, 51.498), (-0.100, 51.500)], osm_id=1)
    far_chunk = make("R", [(-0.060, 51.500), (-0.050, 51.500)], osm_id=2)
    trib = make("T", [(-0.055, 51.494), (-0.055, 51.506)], osm_id=3)
    merged = water.merge_by_name([near_chunk, far_chunk, trib])
    res = water.find_confluences(merged, site_xy=site, crs=epsg, clip_m=8000.0)
    r_uid = [m.uid for m in merged if m.name == "R"][1]
    assert res[r_uid], "слияние на дальнем куске реки потеряно"
    assert res[r_uid][0]["with"] == "osm-t-river"


def test_loop_not_taken_from_far_reach():
    """Петля ищется рядом с участком: у длинной реки самая петлистая пара лежит далеко."""
    ring = _circle_with_neck() + np.array([40000.0, 0.0])   # петля в 40 км от участка
    straight = np.array([[x, 0.0] for x in np.arange(-20000.0, 20001.0, 100.0)])
    pts = np.vstack([straight, ring])
    far = water._detect_loop(pts, near_site_m=3000.0)
    assert far is None, far


def test_clip_near_keeps_boundary_points():
    """Окно режет линию по границе круга, а не по последнему узлу внутри."""
    pts = np.array([[x, 0.0] for x in np.arange(-20000.0, 20001.0, 500.0)])
    pieces = water.clip_near(pts, 12000.0)
    assert len(pieces) == 1
    assert abs(float(np.hypot(*pieces[0][0])) - 12000.0) < 1.0
    assert abs(float(np.hypot(*pieces[0][-1])) - 12000.0) < 1.0


def test_relation_closed_flag_only_for_rings():
    """Члены отношения-линии не помечаются замкнутыми: иначе линия читается как полигон."""
    payload = {"elements": [
        {"type": "relation", "id": 7, "tags": {"waterway": "river", "name": "R"},
         "members": [
             {"type": "way", "role": "main_stream",
              "geometry": [{"lon": 0.0, "lat": 0.0}, {"lon": 0.01, "lat": 0.0}]},
             {"type": "way", "role": "outer",
              "geometry": [{"lon": 0.0, "lat": 0.0}, {"lon": 0.01, "lat": 0.0},
                           {"lon": 0.01, "lat": 0.01}, {"lon": 0.0, "lat": 0.0}]},
         ]},
    ]}
    features = water.parse_features(payload)
    assert [f.closed for f in features] == [False, True]


def test_relation_ways_take_name_from_relation(monkeypatch, tmp_path):
    """Кусок без своего имени получает имя отношения — иначе он не склеивается с рекой."""
    from geosite import water as w

    monkeypatch.setattr(w, "CACHE_DIR", tmp_path)
    answers = []

    def fake_ask(query, **kwargs):
        answers.append(query)
        if "out body;" in query:
            return {"elements": [{"type": "relation", "id": 99,
                                  "tags": {"waterway": "river", "name": "Тест"},
                                  "members": [{"type": "way", "ref": 555, "role": "main_stream"}]}]}
        return {"elements": [{"type": "way", "id": 555, "tags": {"waterway": "stream"},
                              "geometry": [{"lon": 44.0, "lat": 56.33},
                                           {"lon": 44.01, "lat": 56.335}]}]}

    monkeypatch.setattr(w, "_ask", fake_ask)
    ways = w.relation_ways(56.33, 44.01, 8000)
    assert len(ways) == 1
    assert ways[0]["tags"]["name"] == "Тест"
    assert ways[0]["tags"]["_rel"] == "99"


def test_relation_ways_drop_far_members(monkeypatch, tmp_path):
    """Куски дальше 2.5 радиуса не тянутся в пакет: Волга на 3000 км иначе стоит секунд."""
    from geosite import water as w

    monkeypatch.setattr(w, "CACHE_DIR", tmp_path)

    def fake_ask(query, **kwargs):
        if "out body;" in query:
            return {"elements": [{"type": "relation", "id": 99,
                                  "tags": {"waterway": "river", "name": "Тест"},
                                  "members": [{"type": "way", "ref": 1, "role": "main_stream"},
                                              {"type": "way", "ref": 2, "role": "main_stream"}]}]}
        return {"elements": [
            {"type": "way", "id": 1, "tags": {"waterway": "river"},
             "geometry": [{"lon": 44.0, "lat": 56.33}, {"lon": 44.01, "lat": 56.335}]},
            {"type": "way", "id": 2, "tags": {"waterway": "river"},
             "geometry": [{"lon": 44.0, "lat": 54.0}, {"lon": 44.01, "lat": 54.005}]},
        ]}

    monkeypatch.setattr(w, "_ask", fake_ask)
    ways = w.relation_ways(56.33, 44.01, 8000)
    assert [el["id"] for el in ways] == [1]


# --- теги склейки: приливность и правило течения (water 6.1.1) ---

def test_merge_takes_tags_of_piece_nearest_site():
    """Теги — у куска, ближайшего к участку: приливность свойство места, а не всей реки."""
    tidal = make("T", [(-0.10, 51.50), (-0.099, 51.50)],
                 tags={"waterway": "river", "tidal": "yes"})
    dry = make("T", [(-0.0989, 51.50), (-0.098, 51.50)],
               tags={"waterway": "river", "tidal": "no"})
    merged = water.merge_by_name([tidal, dry], near=(51.50, -0.10))
    assert len(merged) == 1
    assert merged[0].tags.get("tidal") == "yes"


def test_merge_without_site_keeps_only_common_tags():
    """Без участка остаются только общие теги: один кусок в трубе не делает трубой всю реку."""
    open_part = make("C", [(-0.10, 51.50), (-0.099, 51.50)],
                     tags={"waterway": "river", "tidal": "yes"})
    culvert = make("C", [(-0.0989, 51.50), (-0.098, 51.50)],
                   tags={"waterway": "river", "tidal": "yes", "tunnel": "culvert"})
    merged = water.merge_by_name([open_part, culvert])
    assert merged[0].tags.get("tidal") == "yes"
    assert "tunnel" not in merged[0].tags


def test_tidal_tag_reaches_flow_rule():
    """Приливный кусок у участка даёт правило 2, а не 5 «рельеф молчит»."""
    piece = make("Т", [(-0.100, 51.500), (-0.097, 51.500)],
                 tags={"waterway": "river", "tidal": "yes"})
    merged = water.merge_by_name([piece], near=(51.50, -0.10))
    site = water.to_utm([(-0.10, 51.50)], 32630)[0]
    metric = water.measure_feature(merged[0], site_xy=site, lat=51.50, lon=-0.10,
                                   crs=32630, grid=None)
    assert metric.tidal is True
    assert metric.flow_rule == 2, metric.flow_rule

def test_partly_buried_river_is_not_a_culvert():
    """Труба — свойство линии: река с коротким засыпанным куском остаётся рекой (Moselle 0.10)."""
    coords = [(-0.100 + 0.0005 * i, 51.5004) for i in range(6)]
    mixed = water.MergedLine("osm-mixed-river", "Mixed", "river", coords, 2, False,
                             hidden_share=0.1,
                             tags={"waterway": "river", "tunnel": "culvert"})
    site = water.to_utm([(-0.100, 51.500)], 32630)[0]
    metric = water.measure_feature(mixed, site_xy=site, lat=51.500, lon=-0.100,
                                   crs=32630, grid=None)
    assert metric.kind == "river"
    assert metric.visible is True


def test_almost_fully_buried_river_is_a_culvert():
    coords = [(-0.100 + 0.0005 * i, 51.5004) for i in range(6)]
    buried = water.MergedLine("osm-buried-river", "Buried", "river", coords, 3, False,
                              hidden_share=0.93, tags={"waterway": "river"})
    site = water.to_utm([(-0.100, 51.500)], 32630)[0]
    metric = water.measure_feature(buried, site_xy=site, lat=51.500, lon=-0.100,
                                   crs=32630, grid=None)
    assert metric.kind == "culvert"
    assert metric.visible is False

def test_clip_runs_keep_original_vertices():
    """T2: нарезка окна возвращает диапазоны индексов, а не пересчитанные точки."""
    import numpy as np
    pts = np.array([[0.0, 0.0], [5000.0, 0.0], [20000.0, 0.0], [25000.0, 0.0]])
    runs = water.clip_runs_near(pts, 12000.0)
    assert runs == [(0, 2)]
    coords = [(-0.1, 51.0), (-0.05, 51.0), (0.0, 51.0), (0.05, 51.0)]
    i0, i1 = runs[0]
    assert coords[i0:i1 + 1][-1] == coords[2]      # последний узел — исходный, не пересчитанный


def test_clip_runs_inside_window_is_whole_line():
    import numpy as np
    pts = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    assert water.clip_runs_near(pts, 12000.0) == [(0, 2)]


def test_culvert_gets_no_flow_rule():
    """T3: труба — всегда правило 0, независимо от того, как ею стала (тег или доля)."""
    coords = [(-0.100 + 0.0004 * i, 51.5004) for i in range(8)]
    site = water.to_utm([(-0.100, 51.500)], 32630)[0]
    cases = (({"waterway": "river", "tunnel": "culvert"}, True, None),   # тег: склейка уже поставила hidden
             ({"waterway": "river"}, False, 0.95))                       # доля скрытого
    for tags, is_hidden, share in cases:
        buried = water.MergedLine("osm-buried", "Buried", "river", coords, 3, is_hidden,
                                  hidden_share=share, tags=tags)
        metric = water.measure_feature(buried, site_xy=site, lat=51.500, lon=-0.100,
                                       crs=32630, grid=None)
        assert metric.kind == "culvert"
        assert metric.flow_rule == 0


def test_hidden_predicates_agree_on_both_tag_spellings():
    """T4: один предикат для вида и для доли — `covered=culvert` и `tunnel=yes` не расходятся."""
    for tags in ({"covered": "culvert"}, {"tunnel": "yes"}, {"tunnel": "culvert"},
                 {"location": "underground"}, {"waterway": "river"}):
        assert water.is_hidden(tags) == water.hydronym.is_hidden(tags)

def test_merge_grows_from_middle_in_both_directions():
    """E-014: река растёт в обе стороны от самого длинного куска, а не только по течению."""
    def seg(lon0, lon1, lat=51.0, n=5):
        return [(lon0 + (lon1 - lon0) * k / (n - 1), lat) for k in range(n)]
    fees = [make("R", seg(0.000, 0.004)), make("R", seg(0.004, 0.010)),
            make("R", seg(0.010, 0.014))]
    merged = water.merge_by_name(fees)
    assert len(merged) == 1
    assert merged[0].ways == 3
    lons = [c[0] for c in merged[0].coords]
    assert lons == sorted(lons), "линия должна идти по порядку, без возвратов"


def test_island_channels_stay_separate_lines():
    """Два русла вокруг острова — две линии, а не одна петля из перемешанных кусков."""
    fork, merge_pt = (0.004, 51.000), (0.010, 51.000)
    fees = [make("R", [(0.000, 51.000), (0.002, 51.000), fork]),
            make("R", [fork, (0.006, 51.002), (0.008, 51.002), merge_pt]),
            make("R", [fork, (0.006, 50.998), (0.008, 50.998), merge_pt]),
            make("R", [merge_pt, (0.012, 51.000), (0.014, 51.000)])]
    merged = water.merge_by_name(fees)
    assert len(merged) == 2, "главное русло и второе русло острова — две линии"
    main = max(merged, key=lambda m: len(m.coords))
    assert len(main.coords) >= 8
    assert main.coords[0][0] == 0.0 and main.coords[-1][0] == 0.014, "ствол идёт от подхода до продолжения"


def test_same_river_pieces_are_not_a_confluence():
    """E-014: куски одной реки, сходящиеся у острова, — это не впадение реки в себя."""
    import numpy as np
    def seg(lon0, lon1, lat, n=5):
        return [(lon0 + (lon1 - lon0) * k / (n - 1), lat) for k in range(n)]
    site = np.array([0.0, 0.0])
    epsg = 32630
    north = make("R", seg(0.0000, 0.0100, 51.002))
    south = make("R", seg(0.0000, 0.0100, 50.998))
    north.uid, south.uid = "r-n", "r-s"
    north.key = south.key = "osm-r-river"
    out = water.find_confluences([north, south], site_xy=site, crs=epsg, radius_m=8000)
    assert all(not items for items in out.values())


def test_one_entry_per_name_prefers_main_channel_over_nearest_stub():
    """E-014: запись о реке описывает главное русло, а не ближайший обрывок у шлюза."""
    import numpy as np
    site = water.to_utm([(0.0, 51.0)], 32630)[0]
    main = water.MergedLine("osm-r-river", "R", "river",
                            [(-0.004, 51.0004), (0.0, 51.0004), (0.009, 51.0004)], 5, False, 0.0,
                            uid="a", tags={})
    stub = water.MergedLine("osm-r-river", "R", "river",
                            [(-0.00020, 51.0002), (-0.00020, 51.0010)], 1, False, 0.0,
                            uid="b", tags={})
    m_main = water.measure_feature(main, site_xy=site, lat=51.0, lon=0.0, crs=32630, grid=None)
    m_stub = water.measure_feature(stub, site_xy=site, lat=51.0, lon=0.0, crs=32630, grid=None)
    assert m_main is not None and m_stub is not None
    assert m_stub.nearest_m < m_main.nearest_m
    kept = water._one_entry_per_name([m_stub, m_main], site_xy=site, crs=32630)
    assert len(kept) == 1
    assert kept[0].nearest_m == m_main.nearest_m
    assert kept[0].pieces == 2

def test_confluence_found_in_shorter_arm_of_horseshoe():
    """E-015: река подковой входит в круг дважды, приток впадает в рукав с меньшим числом вершин.

    Прежний поиск брал внутри круга только самый длинный кусок линии, и слияние терялось.
    """
    import numpy as np

    epsg = 32630
    site = water.to_utm([(0.0, 51.0)], epsg)[0]

    # рукав A: 6 вершин, в 830 м от участка; рукав B: 2 вершины, в 55 м; между ними река
    # уходит далеко за круг — поэтому в круге это два разных куска одной линии
    arm_a = [(-0.010 + 0.004 * k, 51.0075) for k in range(6)]
    far = [(0.010, 51.2000), (-0.010, 51.2000)]
    arm_b = [(-0.010, 51.0005), (0.010, 51.0005)]
    horseshoe = make("Horseshoe", arm_a + far + arm_b[::-1])
    horseshoe.uid = "osm-horseshoe-0"
    horseshoe.key = "osm-horseshoe-river"
    trib = make("Tributary", [(-0.005, 50.9900), (-0.005, 51.0020)])
    trib.uid = "osm-tributary-0"
    trib.key = "osm-tributary-river"

    out = water.find_confluences([horseshoe, trib], site_xy=site, crs=epsg, radius_m=8000)
    assert out.get(trib.uid), "слияние через короткий рукав потеряно"
    assert min(h["dist_m"] for h in out[trib.uid]) < 1500

def test_tidal_null_has_reason_when_wide_payload_absent():
    """E-016: без 30-км выгрузки приливность остаётся null, но с причиной, а не молча."""
    import numpy as np

    epsg = 32630
    site = water.to_utm([(-0.10, 51.51)], epsg)[0]
    metric = water.WaterMetrics(
        id="osm-thames-river", name="River Thames", kind="river", order=None, nearest_m=400.0,
        in_mountain=None, out_mountain=None, bend=None, bend_kind=None, curvature_radius_m=None,
        straight_at_site=True, confluences=[], water_mouth=None, outflow_visible=None,
        flow_dem_dir=None, flow_osm_dir=None, flow_agree=None, flow_chosen="downstream_toward_end",
        flow_rule=5, flow_confidence="medium", dem_slope_m_per_km=None, dem_slope_snr=None,
        dem_reason=None, tidal=None, tidal_source=None, source="openstreetmap",
        coords=[(-0.10, 51.51), (-0.09, 51.51)],
    )
    water._resolve_tidal_by_reach([metric], 51.51, -0.10, crs=epsg, site_xy=site,
                                  radius_m=8000, wide_payload=None, allow_network=False)
    assert metric.tidal is None
    assert metric.tidal_reason and metric.tidal_reason.startswith("no_wide_payload")


def test_waterbody_outlines_do_not_make_confluences():
    """E-017: контуры водоёмов не дают слияний — проверка на реальных данных OSM.

    Фикстура `tests/fixtures/waterbodies-stratford.json` — четыре `way` из OSM вокруг точки
    51.530, -0.014 (Стратфорд, Боу-Бэк-Риверс): линия St Thomas Creek (`waterway=river`,
    16 точек) и три замкнутых `natural=water` (Боу-Крик, канал в трубе, бассейн).
    До 6.5.0 контуры объявлялись водотоками, и пересечения берегов засчитывались как
    впадения: 8 ложных записей. Теперь слияний нет, а вход ручья в водоём пишется
    отдельным списком `enters_waterbody`, которого слияния не касаются.
    """
    import json
    from pathlib import Path

    from geosite import geo

    fixture = Path(__file__).parent / "fixtures" / "waterbodies-stratford.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    epsg = geo.utm_epsg(51.5300, -0.0140)
    site = water.to_utm([(-0.0140, 51.5300)], epsg)[0]
    feats = water.parse_features(payload)

    kinds = {f.osm_id: (f.kind, f.geom) for f in feats}
    assert kinds[4054869] == ("river", "line")          # ручей — линия
    for way in (59041257, 59042050, 368175646):          # три контура — площади
        assert kinds[way][1] == "area", kinds[way]

    conf = water.find_confluences(feats, site_xy=site, crs=epsg, radius_m=8000)
    assert not any(conf.values()), {k: v for k, v in conf.items() if v}

    links = water.find_waterbody_links(feats, site_xy=site, crs=epsg, radius_m=8000)
    creek = next(v for k, v in links.items() if k.startswith("osm-way-4054869"))
    assert creek, links
    assert all(item["kind"] == "enters_waterbody" for item in creek)


def test_area_polygon_never_joins_confluence_search():
    """E-017: прямоугольник `natural=water` и линия, пересекающая его контур дважды.

    Ложное слияние «берег × берег» возникает не из-за геометрии, а из-за классификации:
    замкнутый водоём объявлялся рекой. Теперь вид площади — `lake`, геометрия — `area`,
    и в поиск слияний она не попадает вовсе. Линия внутри площади даёт только вход.
    """
    lat, lon = 51.5000, -0.1000
    d = 0.0009                                   # ~100 м

    def ring(sign: int) -> list[dict]:
        return [{"lat": lat + sign * d, "lon": lon - d}, {"lat": lat + sign * d, "lon": lon + d},
                {"lat": lat - sign * d, "lon": lon + d}, {"lat": lat - sign * d, "lon": lon - d},
                {"lat": lat + sign * d, "lon": lon - d}]

    payload = {"elements": [
        {"type": "way", "id": 1, "tags": {"natural": "water", "water": "canal"}, "geometry": ring(2)},
        {"type": "way", "id": 2, "tags": {"natural": "water", "water": "river"}, "geometry": ring(1)},
        {"type": "way", "id": 3, "tags": {"waterway": "river", "name": "Test Brook"},
         "geometry": [{"lat": lat, "lon": lon - 0.004}, {"lat": lat, "lon": lon},
                      {"lat": lat, "lon": lon + 0.004}]},
    ]}
    from geosite import geo

    epsg = geo.utm_epsg(lat, lon)
    site = water.to_utm([(lon, lat)], epsg)[0]
    feats = water.parse_features(payload)
    assert [f.geom for f in feats] == ["area", "area", "line"]
    # вид описывает поверхность воды: у контура это не русло (E-018). Разделяет их `geom`.
    assert [f.kind for f in feats] == ["lake", "lake", "river"]

    conf = water.find_confluences(feats, site_xy=site, crs=epsg, radius_m=8000)
    assert not any(conf.values()), conf

    # инвариант I-13: участники слияний — только линии; здесь площадь не участвует ни разу
    line_keys = {f"osm-{f.osm_type}-{f.osm_id}-{i}" for i, f in enumerate(feats) if f.geom == "line"}
    for key, items in conf.items():
        if items:
            assert key in line_keys
            for item in items:
                assert item["with"] in line_keys


def test_closed_waterway_with_area_no_stays_a_line():
    """E-017: замкнутый `waterway=canal` с `area=no` — всё-таки линия, а не контур."""
    lat, lon = 51.5000, -0.1000
    ring = [{"lat": lat, "lon": lon}, {"lat": lat, "lon": lon + 0.001},
            {"lat": lat + 0.001, "lon": lon + 0.001}, {"lat": lat, "lon": lon}]
    payload = {"elements": [
        {"type": "way", "id": 1, "tags": {"waterway": "canal", "area": "no"}, "geometry": ring},
    ]}
    feats = water.parse_features(payload)
    assert feats[0].geom == "line"


def test_confluence_found_in_short_arm_real_data_gloucester_streams():
    """E-015 на реальных линиях: ручьи у Глостера, 51.8620, -2.2500.

    Фикстура `tests/fixtures/horseshoe-gloucester-streams.json` — два безымянных
    `waterway=stream` из OSM; Северна в фикстуре нет, ручьи лежат в ~8 км от точки. Сток
    `#41965525` входит в круг 8 км **двумя** куском (2 и 4 вершины), и слияние с `#84733745`
    на 7974.5 м лежит на меньшем куске. Пока перебирался только самый длинный кусок, этой
    записи не было ни у одной из двух рек: проверяется ниже подменой `clip_near`.
    """
    import json
    from pathlib import Path

    from geosite import geo

    lat, lon = 51.8620, -2.2500
    fixture = Path(__file__).parent / "fixtures" / "horseshoe-gloucester-streams.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    epsg = geo.utm_epsg(lat, lon)
    site = water.to_utm([(lon, lat)], epsg)[0]
    merged = water.merge_by_name(water.parse_features(payload), near=(lat, lon))
    short = next(line for line in merged if "#41965525" in line.uid)

    pieces = water.clip_near(water.to_utm(short.coords, epsg) - site, 8000.0) or []
    assert sorted(len(p) for p in pieces) == [2, 4], [len(p) for p in pieces]

    conf = water.find_confluences(merged, site_xy=site, crs=epsg, radius_m=8000, clip_m=8000.0)
    distances = [c["dist_m"] for c in conf[short.uid]]
    assert 7974.5 in distances, distances

    # старый выбор «самый длинный кусок» эту запись терял — воспроизводим подменой
    original = water.clip_near
    water.clip_near = lambda pts, limit: (lambda out: [max(out, key=len)] if out else out)(
        original(pts, limit) or [])
    try:
        old_style = water.find_confluences(merged, site_xy=site, crs=epsg,
                                           radius_m=8000, clip_m=8000.0)
    finally:
        water.clip_near = original
    assert 7974.5 not in [c["dist_m"] for c in old_style[short.uid]]



def _ring(lat: float, lon: float, d: float) -> list[dict]:
    return [{"lat": lat - d, "lon": lon - d}, {"lat": lat - d, "lon": lon + d},
            {"lat": lat + d, "lon": lon + d}, {"lat": lat + d, "lon": lon - d},
            {"lat": lat - d, "lon": lon - d}]


def test_area_gets_only_waterbody_fields():
    """E-018: у контура водоёма нет речных полей — ни значений, ни нулей.

    Замкнутый `natural=water, water=river` раньше мерился как русло: получал вход и выход
    воды, изгиб, направление, и у него могла «найтись» петля (у контура начало совпадает с
    концом). Теперь у площади только площадь, сторона, ближайшая точка и вход русла.
    """
    lat, lon = 51.5000, -0.1000
    payload = {"elements": [
        {"type": "way", "id": 1, "tags": {"natural": "water", "water": "river",
                                          "name": "Test River"}, "geometry": _ring(lat, lon, 0.002)},
        {"type": "way", "id": 2, "tags": {"waterway": "river", "name": "Test River"},
         "geometry": [{"lat": lat, "lon": lon - 0.006}, {"lat": lat, "lon": lon},
                      {"lat": lat, "lon": lon + 0.006}]},
    ]}
    epsg = geo.utm_epsg(lat, lon)
    site = water.to_utm([(lon, lat)], epsg)[0]
    metrics = water.measure_water(lat, lon, crs=epsg, grid=None, site_xy=site,
                                  payload=payload, allow_network=False)

    area = next(m for m in metrics if m.geom == "area")
    assert area.kind in {"lake", "dock"}, area.kind
    for field in ("in_mountain", "out_mountain", "bend", "bend_kind", "loop", "site_position",
                  "flow_chosen", "flow_rule", "flow_confidence", "flow_agree", "flow_ways_share",
                  "water_mouth", "outflow_visible", "tidal", "tidal_source",
                  "crossing_angle_deg", "between_branches"):
        assert getattr(area, field) is None, (field, getattr(area, field))
    assert area.crossings_2km == 0                       # у водоёма пересечений с кругом нет

    # поля водоёма есть
    assert area.area_m2 and area.area_m2 > 1000.0
    assert area.nearest_mountain and area.nearest_azimuth_deg is not None
    assert area.site_inside is True                       # участок внутри контура
    # берег привязан к линии с тем же именем, отдельной «рекой» не выводится
    line = next(m for m in metrics if m.geom == "line")
    assert area.riverbank_of == line.id, (area.riverbank_of, line.id)


def test_dock_channel_links_to_area_and_gets_low():
    """E-020 п.1-2: канал внутри дока привязан к контуру и не выглядит надёжным руслом.

    Линия `waterway=canal` с именем дока раньше шла в отчёте отдельным руслом с входом и
    выходом в одной горе и уверенностью `high`: у канала по правилу 0 «течения нет», и
    уверенность описывала правило направления, а не годность пары 來水/去水.
    """
    lat, lon = 51.5000, -0.1000
    payload = {"elements": [
        {"type": "way", "id": 11, "tags": {"natural": "water", "water": "canal",
                                           "dock": "yes", "name": "Test Dock"},
         "geometry": _ring(lat, lon, 0.002)},
        # канал лежит в 500 м к северу и короток: оба конца в одном секторе, 子 → 子
        {"type": "way", "id": 12, "tags": {"waterway": "canal", "name": "Test Dock"},
         "geometry": [{"lat": lat + 0.0045, "lon": lon - 0.0004},
                      {"lat": lat + 0.0045, "lon": lon + 0.0004}]},
    ]}
    epsg = geo.utm_epsg(lat, lon)
    site = water.to_utm([(lon, lat)], epsg)[0]
    metrics = water.measure_water(lat, lon, crs=epsg, grid=None, site_xy=site,
                                  payload=payload, allow_network=False)

    area = next(m for m in metrics if m.geom == "area")
    line = next(m for m in metrics if m.geom == "line")
    assert area.kind == "dock"
    assert line.dock_channel_of == area.id, (line.dock_channel_of, area.id)
    assert line.in_mountain == line.out_mountain
    assert line.same_mountain_reason, line.same_mountain_reason
    assert line.flow_confidence == "low", line.flow_confidence


def test_far_course_has_no_entry_and_exit_mountains():
    """Русло вне круга 2 км: пара 來水/去水 не измерена, остаются расстояние и сторона.

    Раньше в этом случае писались одинаковые горы входа и выхода с причиной «вне круга
    выборки» и уверенностью low: числа выглядели замеренными, хотя внутри круга от русла нет
    ни метра. Теперь `in_mountain` и `out_mountain` отсутствуют (`None`), а сторона участка
    остаётся в `nearest_mountain`.
    """
    lat, lon = 51.5000, -0.1000
    payload = {"elements": [
        {"type": "way", "id": 21, "tags": {"waterway": "river"},
         "geometry": [{"lat": lat + 0.030, "lon": lon - 0.0005},
                      {"lat": lat + 0.030, "lon": lon + 0.0005}]},
    ]}
    epsg = geo.utm_epsg(lat, lon)
    site = water.to_utm([(lon, lat)], epsg)[0]
    metrics = water.measure_water(lat, lon, crs=epsg, grid=None, site_xy=site,
                                  payload=payload, allow_network=False)
    far = next(m for m in metrics if m.geom == "line")
    assert far.length_2km_m == 0.0, far.length_2km_m
    assert far.in_mountain is None and far.out_mountain is None
    assert far.flow_confidence is None and far.flow_chosen is None   # I-10: поле без пары нельзя
    assert far.nearest_mountain == "子", far.nearest_mountain      # сторона северная
    assert far.same_mountain_reason == "вне круга выборки"


def test_flow_share_ignores_area_contours():
    """E-018: контур не участвует в доле совпадения направления течения.

    У замкнутого контура первая и последняя точки совпадают, поэтому «конец ниже начала»
    для него всегда ложно: каждая площадь тянула долю вниз. Высоты здесь монотонны по
    широте, поэтому линия идёт вниз, а у контура перепад нулевой.
    """
    class Grid:
        def point_value(self, x, y):
            return -float(y)

    epsg = geo.utm_epsg(51.5, -0.1)
    line = Feature(1, "way", "Test Brook", "river", {"waterway": "river"},
                   [(0.0, 0.0), (0.0, 0.001)], False)
    ring = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001), (0.0, 0.0)]
    area = Feature(2, "way", None, "lake", {"natural": "water", "water": "river"}, ring, True)
    area.geom = "area"

    assert water.flow_convention_share([line], Grid(), epsg) == 1.0
    assert water.flow_convention_share([line, area], Grid(), epsg) == 1.0


def test_loop_confidence_margins():
    """E-021: у петли видно, насколько она превышает каждый порог, и её уверенность.

    Петля Темзы в Сити проходит по ratio 3.05 при пороге 3 — запас 1.7 %, это `low`.
    Настоящая излучина с запасом вдвое — `high`; между 10 и 30 % — `medium`.
    """
    margin, confidence = water.loop_confidence(3.05, 183.2, 1.16)
    assert confidence == "low", (margin, confidence)
    assert margin["ratio"] < 0.02 and margin["turn"] < 0.10, margin

    margin, confidence = water.loop_confidence(6.0, 400.0, 0.5)
    assert confidence == "high", (margin, confidence)

    margin, confidence = water.loop_confidence(3.6, 200.0, 0.06)
    assert confidence == "medium", (margin, confidence)

# ---------- E-022: площадные отношения OSM ----------

def _square_ring(lat: float, lon: float, half_m: float) -> list[dict]:
    dlat = half_m / 111132.0
    dlon = half_m / (111132.0 * math.cos(math.radians(lat)))
    ring = [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat),
            (lon + dlon, lat + dlat), (lon - dlon, lat + dlat), (lon - dlon, lat - dlat)]
    return [{"lon": x, "lat": y} for x, y in ring]


def _relation_payload(*, outer_half_m: float = 1500.0, hole_half_m: float | None = 500.0,
                      tags: dict | None = None, rel_id: int = 28934) -> dict:
    lat, lon = 51.5138, -0.0984
    members = [{"type": "way", "ref": 1, "role": "outer",
                "geometry": _square_ring(lat, lon, outer_half_m)}]
    if hole_half_m:
        members.append({"type": "way", "ref": 2, "role": "inner",
                        "geometry": _square_ring(lat, lon, hole_half_m)})
    return {"elements": [{"type": "relation", "id": rel_id,
                          "tags": tags or {"type": "multipolygon", "natural": "water",
                                           "water": "river"},
                          "members": members}]}


def test_relation_multipolygon_becomes_one_area():
    """E-022: отношение собирается в одну площадь, а не в десяток обрывков берега."""
    payload = _relation_payload()
    feats = water.parse_features(payload, lat=51.5138, lon=-0.0984, radius_m=8000.0)

    assert len(feats) == 1, [(f.osm_id, len(f.coords), f.geom) for f in feats]
    feat = feats[0]
    assert feat.geom == "area" and feat.closed
    assert feat.source == "osm_relation"
    assert len(feat.holes) == 1


def test_multipolygon_area_excludes_holes():
    """Остров внутри озера — не вода: площадь считается за вычетом дыр."""
    payload = _relation_payload()
    feats = water.parse_features(payload, lat=51.5138, lon=-0.0984, radius_m=8000.0)
    epsg = geo.utm_epsg(51.5138, -0.0984)
    site = water.to_utm([(-0.0984, 51.5138)], epsg)[0]

    metric = water.measure_feature(feats[0], site_xy=site, lat=51.5138, lon=-0.0984,
                                   crs=epsg, grid=None)
    # 3000 × 3000 м минус 1000 × 1000 м = 8 000 000 м² (граница круга не задевает)
    assert metric.geom == "area"
    # 3000 × 3000 м минус 1000 × 1000 м: допуск 2 % на перевод градусов в UTM
    assert abs(metric.area_m2 - 8_000_000.0) < 160_000.0, metric.area_m2
    assert metric.area_m2 < 8_500_000.0, "дыра не вычтена"


def test_multipolygon_is_clipped_to_sampling_circle():
    """Берег, уходящий на десятки километров, обрезается по кругу до пересечений."""
    payload = _relation_payload(outer_half_m=30_000.0, hole_half_m=None)
    feats = water.parse_features(payload, lat=51.5138, lon=-0.0984, radius_m=8000.0)
    epsg = geo.utm_epsg(51.5138, -0.0984)
    site = water.to_utm([(-0.0984, 51.5138)], epsg)[0]

    ring = water.to_utm(feats[0].coords, epsg) - site
    far = float(np.hypot(*ring.T).max())
    assert 8000.0 < far < 8800.0, far          # круг 8 км с запасом 5 %


def test_relation_surface_counts_in_water_share():
    """Площадь отношения попадает в долю воды, а не только в water.geojson."""
    from geosite import water_pack

    payload = _relation_payload(outer_half_m=1200.0, hole_half_m=None)
    feats = water.parse_features(payload, lat=51.5138, lon=-0.0984, radius_m=8000.0)
    epsg = geo.utm_epsg(51.5138, -0.0984)
    site = water.to_utm([(-0.0984, 51.5138)], epsg)[0]

    share = water_pack.sector_water_share(feats, crs=epsg, site_xy=site, ring_m=(500.0, 2000.0))
    assert share["area_km2"] > 0.0
    assert max(share["by_mountain"].values()) > 0.5, share["by_mountain"]


def test_unnamed_riverbank_links_to_named_line():
    """Берег без имени привязывается к ближайшей осевой по геометрии (Темза в OSM без name)."""
    lat, lon = 51.5138, -0.0984
    bank = {"elements": [{"type": "relation", "id": 28934,
                          "tags": {"type": "multipolygon", "natural": "water", "water": "river"},
                          "members": [{"type": "way", "ref": 1, "role": "outer",
                                       "geometry": _square_ring(lat, lon, 200.0)}]}]}
    line = {"elements": [{"type": "way", "id": 4791944,
                          "tags": {"waterway": "river", "name": "River Thames"},
                          "geometry": _line_near(lat, lon)}]}
    epsg = geo.utm_epsg(lat, lon)
    site = water.to_utm([(lon, lat)], epsg)[0]
    fused = {"elements": bank["elements"] + line["elements"]}

    metrics = water.measure_water(lat, lon, crs=epsg, grid=None, site_xy=site,
                                  radius_m=8000, payload=fused, allow_network=False)
    bank_metric = next(m for m in metrics if m.geom == "area")
    assert bank_metric.source == "osm_relation"
    assert bank_metric.riverbank_of and bank_metric.riverbank_of.startswith("osm-thames-river")
    assert bank_metric.name == "River Thames"


def _line_near(lat: float, lon: float) -> list[dict]:
    """Осевая в 60 м к северу от контура: ближе порога 300 м."""
    return [{"lon": lon - 0.002, "lat": lat + 0.00054},
            {"lon": lon + 0.002, "lat": lat + 0.00054}]
