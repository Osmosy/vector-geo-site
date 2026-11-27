"""Море в кольце: согласие признаков, конфликт и устаревшее поле (S2/E-038).

Проверки синтетические: ни один тест не ходит в сеть. Живые числа (Сува, Сидней, Каир)
записаны в docs/calc/sea.md и в expect.yaml эталонов.
"""

from __future__ import annotations

import pytest

from geosite import sea


def poly(name: str, distance: float, *, kind: str = "sea", area: float = 1e7,
         tags: dict | None = None) -> dict:
    return {"kind": kind, "name": name, "nearest_m": distance, "area_m2": area,
            "nearest_mountain": "庚", "tags": tags or {"natural": "water", "water": "bay"}}


def test_coastline_in_ring_is_sea():
    got = sea.detect(water=[], coast_m=1500.0)
    assert got["present"] is True
    assert got["kind"] == "ocean"
    assert got["confidence"] == "high"
    assert got["deprecated_sea_in_ring"] is True
    assert [e["kind"] for e in got["evidence"]] == ["coastline"]


def test_bay_polygon_without_coastline_is_sea():
    """Сидней: залив нарисован полигоном, береговой линии в выборке нет — море всё равно есть."""
    got = sea.detect(water=[poly("Port Jackson", 666.5, tags={"natural": "bay"})],
                     coast_m=None, wc_share=0.35)
    assert got["present"] is True
    assert got["kind"] == "bay"
    assert got["kind_ru"] == "залив"
    assert {e["kind"] for e in got["evidence"]} == {"polygon", "satellite"}
    assert got["evidence"][0]["name"] == "Port Jackson"


def test_bay_wins_over_harbour():
    """Гавань обычно вложена в залив: у Сиднея в кольце и Port Jackson, и Sydney Harbour."""
    # подтверждение вторым признаком: снимок видит воду — без него гавань и залив не признак
    got = sea.detect(water=[poly("Sydney Harbour", 700, tags={"water": "harbour"}),
                            poly("Port Jackson", 700, tags={"natural": "bay"})],
                     coast_m=None, wc_share=0.35)
    assert got["kind"] == "bay"


def test_satellite_alone_is_conflict_not_absence():
    """OSM молчит, снимок видит 62 % воды: ответ «не знаю», а не «моря нет»."""
    got = sea.detect(water=[], coast_m=None, gsw_share=0.62)
    assert got["present"] is None
    assert got["confidence"] == "low"
    warning = sea.conflict_warning(got)
    assert warning and warning.startswith("sea_conflict")
    assert "62.0" in warning


def test_inland_city_has_no_sea():
    got = sea.detect(water=[{"kind": "river", "name": "Нил", "nearest_m": 300.0}],
                     coast_m=None, wc_share=0.04, gsw_share=0.03)
    assert got["present"] is False
    assert got["kind"] is None
    assert got["deprecated_sea_in_ring"] is False
    assert sea.conflict_warning(got) is None


def test_wide_coastline_is_medium_confidence():
    got = sea.detect(water=[], coast_m=12000.0, coast_wide_m=12000.0)
    assert got["present"] is True
    assert got["confidence"] == "medium"
    assert got["nearest_m"] == 12000.0
    assert "coastline_wide" in {e["kind"] for e in got["evidence"]}
    # 12 км — море есть, но в кольцо 2 км оно не попадает: старое поле обязано отличаться
    assert got["deprecated_sea_in_ring"] is False


def test_tidal_water_counts_only_with_coast_nearby():
    tidal = {"kind": "river", "name": "Темза", "nearest_m": 400.0, "tidal": True}
    near = sea.detect(water=[tidal], coast_m=9000.0)
    assert "tidal" in {e["kind"] for e in near["evidence"]}
    # приливная река без берега рядом (внутренний прилив) морем не становится
    far = sea.detect(water=[tidal], coast_m=None)
    assert far["present"] is False


def test_evidence_records_numbers_and_source():
    got = sea.detect(water=[poly("Sydney Harbour", 666.5)], coast_m=None, wc_share=0.9)
    polygon = next(e for e in got["evidence"] if e["kind"] == "polygon")
    assert polygon["distance_m"] == 666.5
    assert polygon["area_m2"] == 1e7
    assert "OSM" in polygon["source"]
    satellite = next(e for e in got["evidence"] if e["kind"] == "satellite")
    assert satellite["share"] == 0.9 and satellite["threshold"] == sea.SATELLITE_MIN


def test_radius_is_recorded():
    got = sea.detect(water=[], coast_m=1500.0, radius_km=8.0)
    assert got["radius_km"] == 8.0
    assert got["evidence"][0]["ring_km"] == 8.0


@pytest.mark.parametrize("share,expected", [(0.0, False), (0.29, False), (0.30, None)])
def test_satellite_threshold_boundary(share, expected):
    got = sea.detect(water=[], coast_m=None, gsw_share=share)
    assert got["present"] is expected


def test_lake_explains_satellite_water_so_not_sea():
    """Ханчжоу 39.8 % и Чунцин 34.7 % воды по снимку — это озеро и реки, а не море."""
    got = sea.detect(water=[], coast_m=None, coast_wide_m=None, wc_share=0.398,
                     osm_water_share=0.398)
    assert got["present"] is False
    assert "внутренние воды" in (got["note"] or "")


def test_unexplained_satellite_water_stays_unknown():
    """Если OSM воду не объясняет, ответ остаётся «не знаю», а не «моря нет»."""
    got = sea.detect(water=[], coast_m=None, coast_wide_m=None, wc_share=0.398,
                     osm_water_share=0.01)
    assert got["present"] is None
