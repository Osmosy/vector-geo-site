"""Геометрия гор и азимутов."""

import math

import pytest

from geosite import geo


def test_mountains_start_at_zi_north():
    assert geo.MOUNTAINS[0] == "子"
    assert geo.mountain_name(0.0) == "子"
    assert geo.mountain_name(359.0) == "子"
    assert geo.mountain_name(7.0) == "子"


def test_mountain_bounds_wrap_north():
    lo, hi = geo.mountain_bounds("子")
    assert (lo, hi) == (352.5, 7.5)


def test_mountain_bounds_others():
    assert geo.mountain_bounds("卯") == (82.5, 97.5)
    assert geo.mountain_bounds("午") == (172.5, 187.5)


def test_mountain_name_by_bearing():
    assert geo.mountain_name(185) == "午"
    assert geo.mountain_name(90) == "卯"
    assert geo.mountain_name(270) == "酉"
    assert geo.mountain_name(315) == "乾"


def test_angle_distance():
    assert geo.angle_distance(350, 10) == pytest.approx(20)
    assert geo.angle_distance(10, 350) == pytest.approx(20)
    assert geo.angle_distance(0, 180) == pytest.approx(180)


def test_near_boundary():
    assert geo.near_boundary(0.0) is False        # центр сектора 子
    assert geo.near_boundary(6.0) is True         # 6° от центра при половине 7.5°
    assert geo.near_boundary(174.0) is True       # 午: 6° от центра, до границы 172.5 осталось 1.5°
    assert geo.near_boundary(180.0) is False      # центр 午


def test_bearing_london_to_north():
    # из центра Лондона строго на север
    b = geo.bearing_between(51.5, -0.1, 51.6, -0.1)
    assert b == pytest.approx(0.0, abs=0.5)


def test_haversine():
    d = geo.haversine_m(51.5, -0.1, 51.6, -0.1)
    assert 11_000 < d < 11_200


def test_utm_epsg():
    assert geo.utm_epsg(51.5, -0.1) == 32630
    assert geo.utm_epsg(-33.9, 151.2) == 32756
    assert geo.utm_epsg(55.75, 37.62) == 32637


def test_all_24_sectors_tile_the_circle():
    covered = sum(geo.SECTOR_DEG for _ in geo.MOUNTAINS)
    assert covered == 360.0
    assert len(set(geo.MOUNTAINS)) == 24
    for name in geo.MOUNTAINS:
        lo, hi = geo.mountain_bounds(name)
        center = geo.normalize_deg((lo + hi) / 2 if lo < hi else (lo + hi + 360) / 2)
        assert geometric_close(center, geo.MOUNTAINS.index(name) * 15.0)


def geometric_close(a: float, b: float) -> bool:
    return abs(math.remainder(a - b, 360.0)) < 1e-6
