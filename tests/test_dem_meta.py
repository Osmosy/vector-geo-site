"""Паспорт слоя рельефа (D1, D2): kind, система высот, П7 и dem_layer у полей."""

from __future__ import annotations

import pytest

from geosite import dem, validate


def test_file_block_copernicus():
    block = dem.file_block("copernicus")
    assert block == {"source": "copernicus", "kind": "dsm_smoothed", "resolution_m": 30.0,
                     "license": "Copernicus DEM free licence", "vertical_datum": "EGM2008"}


def test_file_block_srtm_is_surface_too():
    """E-023: SRTM — радарная поверхность, а не земля, и в другой системе высот."""
    block = dem.file_block("srtm")
    assert block["kind"] == "dsm_smoothed"
    assert block["vertical_datum"] == "EGM96"


def test_file_block_keeps_explicit_kind():
    """Когда появится DTM, kind придёт от вызывающего, а не из таблицы источников."""
    block = dem.file_block("gedtm30", kind="dtm", resolution_m=30.0)
    assert block["kind"] == "dtm" and block["resolution_m"] == 30.0


@pytest.mark.parametrize("built,expected", [(0.0, "high"), (0.19, "high"), (0.2, "medium"),
                                            (0.49, "medium"), (0.5, "low"), (0.82, "low")])
def test_p7_dsm_confidence_by_built(built, expected):
    """П7: DSM в застройке завышает землю на высоту зданий — уверенность падает."""
    assert dem.relief_confidence("dsm_smoothed", built) == expected


def test_p7_dtm_is_better_but_not_perfect_in_city():
    """DTM — земля, но GEDTM30 в плотной застройке груб: не выше medium (D3)."""
    assert dem.relief_confidence("dtm", 0.0) == "high"
    assert dem.relief_confidence("dtm", 0.82) == "medium"


def test_p7_without_land_cover_assumes_open():
    assert dem.relief_confidence("dsm_smoothed", None) == "high"


def _pack(**quality) -> dict:
    base = {"algo": {}, "sectors": [], "quality": {"dem": "Copernicus DEM GLO-30 (DSM)", **quality}}
    return base


def _codes(pack: dict) -> list[str]:
    return [v["code"] for v in validate.validate(pack)]


def test_i10_requires_dem_layer_on_relief_fields():
    pack = _pack(dem_kind="dsm_smoothed", dem_confidence="high")
    pack["point"] = {"slope_deg": 1.0}
    pack["dragon_vein"] = {"length_km": 1.0}
    assert _codes(pack).count("I-10") == 2

    pack["point"]["dem_layer"] = "dsm_smoothed"
    pack["dragon_vein"]["dem_layer"] = "dsm_smoothed"
    assert _codes(pack).count("I-10") == 0


def test_i10_requires_warning_when_relief_confidence_is_low():
    pack = _pack(dem_kind="dsm_smoothed", dem_confidence="low")
    assert "I-10" in _codes(pack)
    pack["quality"]["warnings"] = ["рельеф: уверенность полей по рельефу low (П7): слой dsm_smoothed"]
    assert _codes(pack).count("I-10") == 0


def test_no_dem_kind_no_dem_check():
    """Пакет, собранный до D2, не получает нарушений за отсутствие новых полей."""
    pack = _pack()
    pack["point"] = {"slope_deg": 1.0}
    assert "I-10" not in _codes(pack)
