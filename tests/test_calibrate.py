"""Калибровка воды (W3, W4): причина, уверенность, выбор значения и инварианты I-14/I-15."""

from __future__ import annotations

import pytest

from geosite import calibrate, validate


def test_within_tolerance_needs_no_reason():
    assert calibrate.reason_for(0.20, 0.215) is None
    assert calibrate.confidence_for(0.20, 0.215, ring_diff=0.002, reason=None) == "high"


def test_ring_gap_downgrades_to_medium():
    """Клин в норме, кольцо нет — уверенность medium: расхождение есть, но не в этом клине."""
    assert calibrate.confidence_for(0.05, 0.06, ring_diff=0.02, reason=None) == "medium"


def test_axis_line_when_osm_has_no_area():
    """Река осевой линией: OSM площади не видит, длина линии есть — причина axis_line."""
    reason = calibrate.reason_for(0.0, 0.35, linear_km=1.2)
    assert reason == "axis_line"
    assert calibrate.confidence_for(0.0, 0.35, ring_diff=0.1, reason=reason) == "low"


def test_axis_without_line_is_stale_osm():
    assert calibrate.reason_for(0.001, 0.2, linear_km=0.0) == "stale_osm"


def test_shadow_in_built_up_area():
    """Тень — эффект на единицы процентов в плотной застройке, а не объяснение любого разрыва."""
    assert calibrate.reason_for(0.02, 0.09, built_share=0.8) == "shadow"
    assert calibrate.reason_for(0.02, 0.09, built_share=0.1) == "stale_osm"
    # разрыв 15 п.п. в городе — это не тень: в OSM просто нет этой воды
    assert calibrate.reason_for(0.042, 0.194, built_share=0.8) == "stale_osm"


def test_tidal_flat_when_osm_wider():
    """OSM шире на приливной реке: в полигон входит осушка, снимок её водой не считает."""
    assert calibrate.reason_for(0.49, 0.39, tidal=True) == "tidal_flat"
    assert calibrate.reason_for(0.49, 0.39, tidal=False) == "stale_osm"


def test_choose_keeps_osm_except_when_osm_blind():
    """Правило выбора: OSM основа; снимок подменяет её только при слепом OSM."""
    assert calibrate.choose(0.49, 0.39, reason="tidal_flat") == 0.49
    assert calibrate.choose(0.02, 0.09, reason="shadow") == 0.02
    assert calibrate.choose(0.0, 0.35, reason="axis_line") == 0.35
    assert calibrate.choose(0.004, 0.12, reason="stale_osm") == 0.12
    # в четыре раза меньше снимка — тоже «почти не видит» (водохранилища Кингстона)
    assert calibrate.choose(0.042, 0.194, reason="stale_osm") == 0.194
    assert calibrate.choose(0.10, 0.20, reason="stale_osm") == 0.10
    # причина есть, но OSM видит сопоставимо — значение не подменяется
    assert calibrate.choose(0.04, 0.06, reason="stale_osm") == 0.04


def _pack(sector: dict, cover: dict | None = None) -> dict:
    base = {"mountain": "子", "from_deg": 352.5, "to_deg": 7.5, "rings": []}
    base.update(sector)
    pack = {"algo": {}, "sectors": [base]}
    if cover is not None:
        pack["land_cover"] = cover
    return pack


def _codes(pack: dict) -> list[str]:
    return [v["code"] for v in validate.validate(pack) if v["code"] in {"I-14", "I-15"}]


def test_i14_requires_reason_above_tolerance():
    bad = _pack({"water_share": 0.49, "water_share_osm": 0.49, "water_share_wc": 0.39,
                 "water_share_confidence": "high", "water_share_reason": None})
    assert _codes(bad) == ["I-14", "I-14"]      # нет причины и уверенность high

    ok = _pack({"water_share": 0.49, "water_share_osm": 0.49, "water_share_wc": 0.39,
                "water_share_confidence": "low", "water_share_reason": "tidal_flat"})
    assert _codes(ok) == []


def test_i14_rejects_high_confidence_with_reason():
    pack = _pack({"water_share": 0.2, "water_share_osm": 0.2, "water_share_wc": 0.2,
                  "water_share_confidence": "high", "water_share_reason": "shadow"})
    assert _codes(pack) == ["I-14"]


def test_i15_shares_and_sum():
    outside = _pack({"water_share": 1.4, "water_share_osm": 1.4, "water_share_wc": 1.2,
                     "water_share_confidence": "low", "water_share_reason": "tidal_flat"})
    assert "I-15" in _codes(outside)

    over = _pack({}, {"water": 0.30, "water_wc": 0.20, "built": 0.70, "tree": 0.10, "open": 0.05})
    violations = [v for v in validate.validate(over) if v["code"] == "I-15"]
    assert violations and "сумма" in violations[0]["message"]

    fine = _pack({}, {"water": 0.30, "water_wc": 0.28, "built": 0.60, "tree": 0.10, "open": 0.02})
    assert [v for v in validate.validate(fine) if v["code"] == "I-15"] == []


def test_reasons_cover_tz_list():
    """Коды причин — закрытый список ТЗ: без выдуманных значений."""
    assert set(calibrate.REASONS) == {"axis_line", "bridge", "dock_lock", "tidal_flat",
                                      "stale_osm", "shadow"}
    assert calibrate.explain("bridge") == "мост"
    assert calibrate.explain(None) is None
