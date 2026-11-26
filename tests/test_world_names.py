"""G3: имена любой письменности, сезонная вода, приливы по берегу, разреженный OSM."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from geosite import dem, geo, water, water_pack, worldnames

# эталоны ТЗ: по одному языку на строку, чтобы «покрыто тестами» значило буквально это
LANGUAGE_CASES = [
    ("黃河", "river", "zh"),                     # Пекин/Гонконг: родовое слово в конце
    ("城門河", "river", "zh"),
    ("太湖", "lake", "zh"),
    ("維多利亞港", "sea", "zh"),                  # порт сводится к морю в коротком списке пакета
    ("نهر النيل", "river", "ar"),                # Каир
    ("بحيرة فكتوريا", "lake", "ar"),
    ("Sungai Klang", "river", "id"),             # Джакуarta
    ("Danau Toba", "lake", "id"),
    ("Kâğıthane Deresi", "stream", "tr"),        # Стамбул
    ("Nehir Kızılırmak", "river", "tr"),
    ("Río Negro", "river", "es"),                # Буэнос-Айрес
    ("Arroyo Maldonado", "stream", "es"),
    ("River Thames", "river", "en"),             # Лондон
    ("Silverwater Creek", "stream", "en"),
    ("Река Москва", "river", "ru"),
    ("Чёрная речка", "river", "ru"),
]


def _feature_kind(name: str) -> str:
    """Куда попадает имя в пакете: виды у пакета короче таблицы родовых слов."""
    guess = worldnames.kind_from_name(name)
    assert guess is not None, f"родовое слово не распознано: {name}"
    return {("pond"): "lake", ("wetland"): "lake", ("harbour"): "sea", ("bay"): "sea"}.get(
        guess["kind"], guess["kind"])


@pytest.mark.parametrize("name,kind,language", LANGUAGE_CASES)
def test_generic_words_of_each_language(name, kind, language):
    found = worldnames.kind_from_name(name)
    assert found and found["language"] == language
    assert _feature_kind(name) == kind


def test_pack_kind_falls_back_to_the_name():
    """Тег молчит — вид воды берётся из родового слова, а не остаётся «other»."""
    assert water._kind_with_name({"natural": "water"}, "太湖") == "lake"
    assert water._kind_with_name({"waterway": "canal"}, "運河") == "canal"   # тег главнее
    assert water._kind_with_name({}, "Sungai Ciliwung") == "river"
    assert water._kind_with_name({}, "Никакого слова") == "other"


def test_nfkc_gives_one_key_for_one_name():
    assert worldnames.nfkc("  Ｔｈａｍｅｓ　ﬁ ") == "Thames fi"
    assert worldnames.names_of({"name": "Thames", "name:zh": "泰晤士河",
                                "alt_name": "The Thames;Tamesis"}) == [
        "Thames", "泰晤士河", "The Thames", "Tamesis"]


def test_seasonal_water_is_a_fact_not_a_guess():
    assert worldnames.is_seasonal({"intermittent": "yes"}) is True
    assert worldnames.is_seasonal({"waterway": "wadi"}) is True
    assert worldnames.is_seasonal({"name": "Wadi Rum"}) is True
    # тегов о сезонности нет и имя не говорит — «не смотрели», а не «постоянная»
    assert worldnames.is_seasonal({"waterway": "river"}) is None
    assert worldnames.is_seasonal({"waterway": "stream", "name": "Fleet"}) is None
    assert worldnames.is_seasonal({}) is None
    # «постоянная» — только когда это сказано тегом
    assert worldnames.is_seasonal({"waterway": "river", "intermittent": "no"}) is False


def test_seasonal_reaches_the_pack():
    payload = {"elements": [{
        "type": "way", "id": 11,
        "tags": {"waterway": "stream", "name": "Wadi Hanifa", "intermittent": "yes"},
        "geometry": [{"lon": 0.0, "lat": 0.0}, {"lon": 0.01, "lat": 0.01}]}]}
    features = water.parse_features(payload, lat=0.0, lon=0.0, radius_m=8000)
    assert features and features[0].seasonal is True
    assert features[0].kind == "stream"
    # имя тоже говорит о сезонности: «wadi» — русло, по которому вода идёт временами
    assert worldnames.is_seasonal({"waterway": "stream", "name": "Wadi Hanifa"}) is True


def test_coastline_marks_water_tidal():
    """Прилив по берегу: вода ближе километра к `natural=coastline` — приливная."""
    line = [{"lon": 0.02, "lat": 0.00}, {"lon": 0.02, "lat": 0.01}]
    payload = {"elements": [{"type": "way", "id": 5, "geometry": line}]}
    coast = water.coast_lines(payload)
    assert coast and len(coast[0]) == 2
    assert water._coast_distance_m(coast, [(0.0, 0.0)]) > 1000.0
    assert water._coast_distance_m(coast, [(0.015, 0.0)]) <= 1000.0


def test_truncated_payload_is_detected_by_remark():
    assert water.truncated({"elements": [], "remark": "runtime error: Query timed out"})
    assert not water.truncated({"elements": [{"type": "way", "id": 1}]})
    assert not water.truncated(None)


def test_tiles_are_merged_without_duplicates():
    a = {"elements": [{"type": "way", "id": 1}, {"type": "way", "id": 2}]}
    b = {"elements": [{"type": "way", "id": 2}, {"type": "way", "id": 3}],
         "remark": "runtime error: Query timed out"}
    merged = water.merge_payloads([a, b])
    assert [el["id"] for el in merged["elements"]] == [1, 2, 3]
    assert water.truncated(merged)          # обрезанность не теряется при склейке


def test_tile_boxes_cover_the_circle():
    boxes = water._tile_boxes(51.5138, -0.0984, 8000)
    assert len(boxes) == water.TILE_SIDE ** 2
    lat_min = min(b[0] for b in boxes)
    lat_max = max(b[2] for b in boxes)
    assert lat_max - lat_min == pytest.approx(2 * 8 / 111.32, rel=0.02)


def test_hydrorivers_reaches_become_water_objects():
    """Разреженный OSM: русло HydroRIVERS — объект воды с порядком и расходом, без имени."""
    reach = {"HYRIV_ID": 77, "ORD_STRA": 5, "DIS_AV_CMS": 12.5, "distance_m": 420.0,
             "_points": [(0.0, 0.0), (0.01, 0.01), (0.02, 0.02)]}
    site_xy = water.to_utm([(0.0, 0.0)], 32631)[0]
    grid = dem.DemGrid(np.full((60, 60), 10.0, dtype="float32"),
                       Affine(100.0, 0.0, site_xy[0] - 3000.0, 0.0, -100.0, site_xy[1] + 3000.0),
                       "EPSG:32631", 100.0)
    metrics = water.hydrorivers_metrics([reach], site_xy=site_xy, lat=0.0, lon=0.0,
                                       crs=32631, grid=grid, max_km=8.0)
    assert metrics, "русло не измерено"
    one = metrics[0]
    assert one.order == 5 and one.discharge_m3s == 12.5
    assert one.source == "hydrorivers" and one.name is None
    objects = water_pack.to_objects(metrics)
    assert objects[0].source == "hydrorivers"


def test_name_aliases_reach_the_pack_object():
    payload = {"elements": [{
        "type": "way", "id": 12,
        "tags": {"waterway": "river", "name": "River Thames", "name:zh": "泰晤士河",
                 "int_name": "Thames"},
        "geometry": [{"lon": 0.0, "lat": 0.0}, {"lon": 0.01, "lat": 0.0}]}]}
    features = water.parse_features(payload, lat=0.0, lon=0.0, radius_m=8000)
    assert features[0].names == ["River Thames", "泰晤士河", "Thames"]
    site_xy = water.to_utm([(0.0, 0.0)], 32631)[0]
    grid = dem.DemGrid(np.full((60, 60), 10.0, dtype="float32"),
                       Affine(100.0, 0.0, site_xy[0] - 3000.0, 0.0, -100.0, site_xy[1] + 3000.0),
                       "EPSG:32631", 100.0)
    objects = water_pack.to_objects([water.measure_feature(
        features[0], site_xy=site_xy, lat=0.0, lon=0.0, crs=32631, grid=grid)])
    assert objects[0].names == ["River Thames", "泰晤士河", "Thames"]
