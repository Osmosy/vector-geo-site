"""Этап 4: склонение (WMM) и ветер (ERA5 через Open-Meteo)."""

from __future__ import annotations

import math

import pytest

from geosite import climate, geo


def _series(directions, speeds_kmh):
    days = [f"2020-{1 + (i // 28) % 12:02d}-{1 + i % 28:02d}" for i in range(len(directions))]
    return {"time": days, "direction": directions, "speed_kmh": speeds_kmh,
            "source": "test"}


def test_declination_is_east_positive_in_london():
    d = climate.declination(51.5138, -0.0984, year=2026.9)
    assert 0.5 < d["deg"] < 2.0          # реально ~+1.2° на восток
    assert "WMM" in d["source"]


def test_declination_sign_flips_across_the_atlantic():
    west = climate.declination(-34.9, -56.2, year=2026.9)    # Монтевидео
    assert west["deg"] < 0


def test_magnetic_bearing_subtracts_east_declination():
    # истинный 180 при восточном склонении 1.2° → компас показывает 178.8
    assert climate.magnetic_bearing(180.0, 1.238) == pytest.approx(178.8, abs=0.05)
    assert climate.magnetic_bearing(5.0, 1.238) == pytest.approx(3.8, abs=0.05)   # через ноль


def test_prevailing_wind_is_the_direction_air_comes_from():
    """Ветер «откуда» 225° (юго-запад) — перенос идёт на северо-восток; в пакет идёт 225."""
    data = _series([225] * 100, [20.0] * 100)
    out = climate.summarize(data)
    assert out["prevailing_from_deg"] == pytest.approx(225.0, abs=1.0)
    assert out["prevailing_from_mountain"] == geo.mountain_name(225.0)
    assert out["mean_speed_ms"] == pytest.approx(20.0 / 3.6, abs=0.01)


def test_shares_sum_to_one_and_follow_the_days():
    data = _series([0] * 50 + [180] * 50, [10.0] * 100)
    out = climate.summarize(data)
    assert sum(out["by_mountain"].values()) == pytest.approx(1.0, abs=0.001)
    assert out["by_mountain"][geo.mountain_name(0.0)] == pytest.approx(0.5, abs=0.01)
    assert out["by_mountain"][geo.mountain_name(180.0)] == pytest.approx(0.5, abs=0.01)


def test_calms_are_counted_not_averaged_into_direction():
    data = _series([90] * 10 + [None] * 0, [1.0] * 10)
    out = climate.summarize(data)
    assert out["calm_share"] == pytest.approx(1.0, abs=0.01)
    assert out["days"] == 10


def test_seasons_are_split_by_month():
    winter = [f"2020-01-{d:02d}" for d in range(1, 29)]
    summer = [f"2020-07-{d:02d}" for d in range(1, 29)]
    data = {"time": winter + summer,
            "direction": [270] * 28 + [90] * 28,
            "speed_kmh": [10.0] * 56, "source": "test"}
    out = climate.summarize(data)
    assert out["seasonal"]["winter"]["prevailing_from_deg"] == pytest.approx(270.0, abs=1.0)
    assert out["seasonal"]["summer"]["prevailing_from_deg"] == pytest.approx(90.0, abs=1.0)
    assert out["seasonal"]["winter"]["days"] == 28


def test_empty_archive_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        climate.summarize({"time": [], "direction": [], "speed_kmh": []})


def test_decimal_year_is_a_fraction():
    import datetime as dt

    value = climate.decimal_year(dt.date(2026, 7, 2))
    assert value == pytest.approx(2026.5, abs=0.01)
