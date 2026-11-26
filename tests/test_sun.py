"""Солнце (H2): полдни, сторона полудня, счёт дней, горизонт и освещённость."""

from __future__ import annotations

import numpy as np
from affine import Affine

from geosite import geo, sun
from geosite.dem import DemGrid

LONDON = (51.5138, -0.0984)
SYDNEY = (-33.8568, 151.2153)
SINGAPORE = (1.3521, 103.8198)


def _grid(value: float = 0.0, size: int = 120, res: float = 100.0) -> DemGrid:
    data = np.full((size, size), value, dtype="float32")
    transform = Affine(res, 0, 0, 0, -res, size * res)
    return DemGrid(data, transform, "EPSG:32630", res)


def test_noon_is_south_in_north_and_north_in_south():
    """Полдень в Северном полушарии на юге, в Южном — на севере: это и есть вся разница."""
    london = {row["event"]: row for row in sun.noons(*LONDON)}
    sydney = {row["event"]: row for row in sun.noons(*SYDNEY)}

    assert london["june_solstice"]["side8"] == "S"
    assert 60.0 < london["june_solstice"]["altitude_deg"] < 64.0
    assert london["december_solstice"]["side8"] == "S"
    assert 14.0 < london["december_solstice"]["altitude_deg"] < 17.0

    assert sydney["june_solstice"]["side8"] == "N"
    assert 31.0 < sydney["june_solstice"]["altitude_deg"] < 35.0
    assert sydney["december_solstice"]["side8"] == "N"
    assert sydney["december_solstice"]["altitude_deg"] > 78.0


def test_noon_side_days_replace_equatorial_threshold():
    """Счёт дней полуденного солнца: у Лондона 0 с севера, у Сингапура — почти поровну.

    Порог «около экватора» берётся по этим числам, а не по ±1° широты: в Сингапуре
    (1.35° с. ш.) солнце всё равно бывает с обеих сторон, и школы расходятся слабее.
    """
    london = sun.noon_side(*LONDON)
    sydney = sun.noon_side(*SYDNEY)
    singapore = sun.noon_side(*SINGAPORE)

    assert london["days_noon_north"] == 0 and london["days_noon_south"] == 365
    assert sydney["days_noon_north"] == 365 and sydney["days_noon_south"] == 0
    assert 150 < singapore["days_noon_north"] < 215
    assert 150 < singapore["days_noon_south"] < 215
    assert abs(singapore["days_noon_north"] - singapore["days_noon_south"]) < 30
    # сторона полудня — та, где солнце чаще, но обе доли видно
    assert singapore["side8"] in {"N", "S"}


def test_horizon_profile_sees_wall_in_the_east():
    """Стена в 2 км на востоке поднимает горизонт именно на востоке."""
    grid = _grid(0.0)
    x0, y0 = 6000.0, 6000.0
    # стена высотой 200 м в 2 км к востоку (азимут 90°) и ещё одна полоса в 2.5 км
    for x in (8000.0, 8500.0):
        col = int(round(x / 100.0))
        grid.data[:, col] = 200.0
    profile = sun.horizon_profile(grid, x0, y0, 0.0, max_km=4.0, bins=72)
    east = profile[18]      # 90°
    west = profile[54]      # 270°
    assert east > 2.0, profile
    assert west == 0.0


def test_illumination_counts_less_with_wall():
    """Стена на востоке отнимает утренние часы, но не все."""
    grid = _grid(0.0)
    flat = sun.illumination(*LONDON, sun.horizon_profile(grid, 6000.0, 6000.0, 0.0, max_km=4.0))
    walled_grid = _grid(0.0)
    for x in (8000.0, 8500.0):
        walled_grid.data[:, int(round(x / 100.0))] = 300.0
    walled = sun.illumination(*LONDON, sun.horizon_profile(walled_grid, 6000.0, 6000.0, 0.0, max_km=4.0))

    assert walled["hours_by_side"]["E"] < flat["hours_by_side"]["E"], "стена должна отнять часы"
    assert walled["blocked_hours"] > flat["blocked_hours"]
    assert walled["hours_by_side"]["W"] == flat["hours_by_side"]["W"], "запад стеной не закрыт"
    assert flat["total_hours"] > 4000.0


def test_illumination_without_dem_is_astronomical():
    block = sun.illumination(*LONDON, None)
    assert block["dem_layer"] is None
    assert block["blocked_hours"] == 0.0
    assert block["total_hours"] > 4000.0


def test_block_carries_hemisphere_free_facts():
    """Блок `sun` не знает про школы: только высоты, азимуты, стороны и счёт дней."""
    data = sun.block(*SYDNEY)
    assert data["days_noon_south"] == 0
    assert data["noon_side"]["side8"] == "N"
    assert {row["event"] for row in data["noons"]} == {"june_solstice", "december_solstice", "equinox"}
    text = str(data)
    for forbidden in ("青龍", "白虎", "玄武", "朱雀", "dragon", "tiger", "风水", "фэншуй"):
        assert forbidden not in text
