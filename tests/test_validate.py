"""Инварианты пакета: на каждый код — испорченный пакет."""

from __future__ import annotations

import copy

import pytest

from geosite import validate as V


def base_pack() -> dict:
    return {
        "schema": "site_pack/1",
        "query": {"input": "тест", "kind": "point", "resolved_by": "test"},
        "site": {"lat": 0.0, "lon": 0.0, "elev_m": 30.0, "precision": "point", "utm_epsg": 32630},
        "orientation": {"facing_true_deg": 180.0, "facing_mountain": "午", "sitting_mountain": "子",
                        "source": "model", "confidence": "low", "near_boundary": False},
        "chart_year": {"value": 2026, "source": "current_year"},
        "sectors": [{"mountain": "子", "from_deg": 352.5, "to_deg": 7.5,
                     "rings": [{"r_km": [2.0, 8.0], "max_rel_elev_m": 10.0, "mean_rel_elev_m": 2.0,
                                "horizon_deg": 0.5, "hill_share": 0.2, "water_share": None, "peaks": []}]}],
        "four_spirits": {"tortoise": {"rel_elev_m": 5.0, "dist_km": 2.0, "continuity": 0.5},
                         "bird": {"rel_elev_m": 0.0, "dist_km": 1.0, "continuity": 0.2},
                         "dragon": {"rel_elev_m": 3.0, "dist_km": 1.5, "continuity": 0.4},
                         "tiger": {"rel_elev_m": 2.0, "dist_km": 1.2, "continuity": 0.3},
                         "dragon_vs_tiger": 1.0},
        "water": [{"id": "osm-A-river", "name": "A", "kind": "river", "nearest_m": 500.0,
                   "in_mountain": "坤", "out_mountain": "辰", "bend": "outer", "crossings_2km": 2, "crossing_angle_deg": 90.0,
                   "loop": None, "confluences": [], "flow_chosen": "downstream_toward_end",
                   "flow_rule": 1, "flow_confidence": "high", "water_mouth": None}],
        "quality": {"dem": "test", "warnings": [], "notes": []},
        "algo": {"water": "3.1.0", "terrain": "2.0.0"},
        "attribution": [],
    }


def codes(pack: dict) -> set[str]:
    return {v["code"] for v in V.validate(pack)}


def test_clean_pack_has_no_violations():
    assert V.validate(base_pack()) == []


def test_i01_bad_mountain_name():
    pack = base_pack()
    pack["sectors"][0]["mountain"] = "XX"
    assert "I-01" in codes(pack)


def test_i01_azimuth_out_of_range():
    pack = base_pack()
    pack["sectors"][0]["to_deg"] = 400.0
    assert "I-01" in codes(pack)


def test_i02_negative_distance():
    pack = base_pack()
    pack["water"][0]["nearest_m"] = -5.0
    assert "I-02" in codes(pack)


def test_i02_distance_beyond_radius():
    pack = base_pack()
    pack["water"][0]["nearest_m"] = 50_000.0
    assert "I-02" in codes(pack)


def test_i03_share_out_of_range():
    pack = base_pack()
    pack["sectors"][0]["rings"][0]["hill_share"] = 1.4
    assert "I-03" in codes(pack)


def test_i04_closure_with_both_sides_capped():
    pack = base_pack()
    pack["water"][0]["water_mouth"] = {"mountain": "乙", "closure_index": 0.4,
                                       "width_capped": True, "capped_sides": 2, "width_m": 4000.0}
    assert "I-04" in codes(pack)


def test_i05_loop_without_ratio():
    pack = base_pack()
    pack["water"][0]["loop"] = {"gap_m": 1000.0, "along_m": 1500.0, "site_inside": False}
    assert "I-05" in codes(pack)


def test_i05_loop_ratio_too_small():
    pack = base_pack()
    pack["water"][0]["loop"] = {"gap_m": 1000.0, "along_m": 1500.0, "ratio": 1.5}
    assert "I-05" in codes(pack)


def test_i05_loop_ok():
    pack = base_pack()
    pack["water"][0]["loop"] = {"gap_m": 300.0, "along_m": 2000.0, "ratio": 6.7, "site_inside": True}
    assert "I-05" not in codes(pack)


def test_i06_duplicate_confluence():
    pack = base_pack()
    pack["water"][0]["confluences"] = [
        {"with": "osm-B-river", "dist_m": 500.0, "mountain": "卯"},
        {"with": "osm-B-river", "dist_m": 510.0, "mountain": "卯"},
    ]
    assert "I-06" in codes(pack)


def test_i06_two_distinct_confluences_allowed():
    pack = base_pack()
    pack["water"][0]["confluences"] = [
        {"with": "osm-B-river", "dist_m": 500.0, "mountain": "卯"},
        {"with": "osm-B-river", "dist_m": 900.0, "mountain": "卯"},
    ]
    assert "I-06" not in codes(pack)


def test_i07_same_mountain_in_and_out():
    pack = base_pack()
    pack["water"][0]["out_mountain"] = pack["water"][0]["in_mountain"]
    assert "I-07" in codes(pack)


def test_i08_flow_without_rule():
    pack = base_pack()
    pack["water"][0]["flow_rule"] = None
    assert "I-08" in codes(pack)


def test_i09_elevation_out_of_range():
    pack = base_pack()
    pack["site"]["elev_m"] = 12_000.0
    assert "I-09" in codes(pack)


def test_i09_elevation_must_match_dem():
    pack = base_pack()
    assert "I-09" in {v["code"] for v in V.validate(pack, dem_point=35.0)}


def test_i10_low_confidence_needs_warning():
    pack = base_pack()
    pack["water"][0]["flow_confidence"] = "low"
    assert "I-10" in codes(pack)


def test_i10_low_confidence_with_warning_ok():
    pack = base_pack()
    pack["water"][0]["flow_confidence"] = "low"
    pack["quality"]["warnings"] = ["water.flow: low — плоский профиль"]
    assert "I-10" not in codes(pack)


def test_i11_missing_algo():
    pack = base_pack()
    pack["algo"] = {}
    assert "I-11" in codes(pack)


def test_i12_zero_with_cap_flag():
    pack = base_pack()
    pack["water"][0]["water_mouth"] = {"mountain": "乙", "closure_index": 0.0,
                                       "width_capped": True, "capped_sides": 1, "width_m": 4850.0}
    assert "I-12" in codes(pack)


def test_multiple_violations_reported_together():
    pack = base_pack()
    pack["site"]["elev_m"] = -500.0
    pack["water"][0]["nearest_m"] = -1.0
    found = codes(pack)
    assert {"I-02", "I-09"} <= found


def test_apply_to_pack_marks_invalid():
    pack = base_pack()
    pack["water"][0]["nearest_m"] = -1.0
    violations = V.validate(pack)
    V.apply_to_pack(pack, violations)
    assert pack["quality"]["valid"] is False
    assert pack["quality"]["violations"]


def test_i10_requires_confidence_pair():
    """R1: эвристическое поле без парного confidence — нарушение I-10."""
    from geosite.validate import CONFIDENCE_PAIRS  # noqa: F401

    pack = base_pack()
    pack["water"][0]["water_mouth"] = {"mountain": "乙", "closure_index": 0.01,
                                       "closure_confidence": "low", "width_m": 100.0}
    assert "I-10" not in codes(pack), "с уверенностью нарушений быть не должно"

    del pack["water"][0]["water_mouth"]["closure_confidence"]
    assert "I-10" in codes(pack)

    pack2 = base_pack()
    del pack2["water"][0]["flow_confidence"]
    assert "I-10" in codes(pack2)


def test_i07_same_mountains():
    """R2: видимая вода с одинаковыми горами входа и выхода — I-07."""
    pack = base_pack()
    pack["water"][0]["visible"] = True
    pack["water"][0]["out_mountain"] = pack["water"][0]["in_mountain"]
    pack["water"][0]["crossings_2km"] = 2
    assert "I-07" in codes(pack)

    pack["water"][0]["visible"] = False      # подземная вода — исключение
    assert "I-07" not in codes(pack)
