"""Геокодер: разбор координат, кеш, неоднозначность."""

import math

import pytest

from geosite import geocode


def test_decimal_coordinates():
    assert geocode.parse_coordinates("51.5138, -0.0984") == (51.5138, -0.0984)
    assert geocode.parse_coordinates("55,75 37,62") == (55.75, 37.62)
    assert geocode.parse_coordinates("  -33.9;151.2 ") == (-33.9, 151.2)


def test_dms_coordinates():
    lat, lon = geocode.parse_coordinates("51°30'49.7\"N 0°05'54.2\"W")
    assert lat == pytest.approx(51.5138, abs=1e-3)
    assert lon == pytest.approx(-0.0984, abs=1e-3)


def test_not_coordinates():
    assert geocode.parse_coordinates("Лондон, Сити") is None
    assert geocode.parse_coordinates("Красная площадь 1") is None


def test_coordinates_bypass_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("сеть не должна вызываться для координат")

    monkeypatch.setattr(geocode, "_get", boom)
    r = geocode.resolve("51.5, -0.1")
    assert r.kind == "point" and r.precision == "point" and r.resolved_by == "input"


@pytest.mark.network
def test_nominatim_city_resolution():
    r = geocode.resolve("Лондон, Сити")
    assert math.isclose(r.lat, 51.51, abs_tol=0.08)
    assert r.resolved_by == "nominatim"
    assert r.precision in {"district", "point", "settlement"}
