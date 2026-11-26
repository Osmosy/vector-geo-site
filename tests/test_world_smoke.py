"""E-027…E-029: находки живых прогонов по миру — море в кольце, пустые кольца, отношения."""

from __future__ import annotations

import pytest

from geosite import calibrate, cli, water


def test_relation_query_asks_for_area_water_not_only_named_rivers():
    """E-027: Нил в Каире — анонимное `natural=water`, и старый запрос его не видел."""
    query = water._relation_query(30.0444, 31.2357, 8000)
    assert '["natural"="water"]' in query
    assert '["landuse"~"^(reservoir|basin)$"]' in query
    assert "riverbank" in water.RELATION_KINDS
    # имя больше не требуется: требование имени и было причиной нулевой воды в Каире
    assert '["name"]' not in query


def test_sea_reason_wins_over_stale_osm():
    """E-028: море объясняет расхождение целиком — «устаревший OSM» отправил бы искать ошибку."""
    assert calibrate.reason_for(0.001, 0.417, sea=True) == "sea_in_ring"
    assert calibrate.reason_for(0.001, 0.417, sea=False) == "stale_osm"
    # в допуске объяснять нечего, даже у моря
    assert calibrate.reason_for(0.30, 0.29, sea=True) is None


def test_sea_never_replaces_osm_with_the_snapshot():
    """Замена OSM снимком записала бы в воду участка 42 % моря — этого не происходит."""
    assert calibrate.choose(0.001, 0.417, reason="sea_in_ring") == 0.001
    assert calibrate.choose(0.001, 0.417, reason="stale_osm") == 0.417
    # море не понижает уверенность: расхождение не про долю воды
    assert calibrate.confidence_for(0.001, 0.417, ring_diff=0.01, reason="sea_in_ring") == "high"
    assert calibrate.confidence_for(0.001, 0.417, ring_diff=0.20, reason="sea_in_ring") == "medium"


def _sector(mountain: str, far: float | None, void: float | None = 0.0) -> dict:
    return {"mountain": mountain,
            "rings": [{"r_km": [0.0, 0.5], "max_rel_elev_m": None, "void_share": void},
                      {"r_km": [0.5, 2.0], "max_rel_elev_m": None, "void_share": void},
                      {"r_km": [2.0, 8.0], "max_rel_elev_m": far, "void_share": void}]}


def test_facing_survives_empty_rings():
    """E-029: точка в воде — все кольца `null`; это не повод падать и не повод врать «на север»."""
    measured = {"sectors": [_sector(m, None, 1.0) for m in ("子", "丑", "寅")]}
    facing, why = cli._model_facing(measured)
    assert facing is None
    assert "не определена" in why and "воде" in why


def test_facing_picks_the_tallest_and_ignores_nulls():
    measured = {"sectors": [_sector("子", None), _sector("辰", 40.4), _sector("酉", 12.0)]}
    facing, why = cli._model_facing(measured)
    assert facing is not None and "спина к 辰" in why
    # спина к 辰 (120°) — фасад в противоположную сторону
    assert facing == pytest.approx(300.0, abs=0.01)


def test_coast_distance_is_a_measurement_not_a_guess():
    payload = {"elements": [{"type": "way", "id": 1,
                            "geometry": [{"lon": 0.02, "lat": 0.0}, {"lon": 0.02, "lat": 0.05}]}]}
    assert water.nearest_coast_m(payload, 0.0, 0.0) == pytest.approx(2220, rel=0.05)
    assert water.nearest_coast_m({"elements": []}, 0.0, 0.0) is None
    assert water.nearest_coast_m(None, 0.0, 0.0) is None
