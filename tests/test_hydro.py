"""Порядок реки и расход: HydroRIVERS (озаглавленный участок → река у участка)."""

from __future__ import annotations

import math

import pytest

from geosite import hydro


def _metric(name, kind, nearest_m, coords):
    from geosite.water import WaterMetrics

    return WaterMetrics(
        id=f"osm-{name}", name=name, kind=kind, order=None, nearest_m=nearest_m,
        in_mountain=None, out_mountain=None, bend=None, bend_kind=None,
        curvature_radius_m=None, straight_at_site=False, confluences=[], water_mouth=None,
        outflow_visible=None, flow_dem_dir=None, flow_osm_dir=None, flow_agree=None,
        flow_chosen=None, flow_rule=None, flow_confidence=None, dem_slope_m_per_km=None,
        dem_slope_snr=None, dem_reason=None, tidal=None, tidal_source=None, source="osm",
        coords=coords)


def test_distance_to_polyline_is_metres_not_degrees():
    # 0.01° по широте — это ~1105 м
    d = hydro._distance_m(51.5, -0.1, [(-0.1, 51.51), (-0.1, 51.52)])
    assert 1090 < d < 1120


def test_distance_to_segment_uses_perpendicular_not_vertices():
    # точка в ~500 м сбоку от середины отрезка длиной 20 км: до вершин было бы ~10 км
    d = hydro._distance_m(51.6045, -0.1, [(-0.25, 51.6), (0.05, 51.6)])
    assert 450 < d < 550


def test_site_reach_takes_order_not_the_nearest_channel():
    """«Ближайший» участок в эстуарии — ручей; главный — с наибольшим порядком."""
    reaches = [
        {"distance_m": 120.0, "ORD_STRA": 1, "ORD_CLAS": 3, "DIS_AV_CMS": 0.18, "HYRIV_ID": 1},
        {"distance_m": 380.0, "ORD_STRA": 5, "ORD_CLAS": 1, "DIS_AV_CMS": 89.9, "HYRIV_ID": 2},
    ]
    site = hydro.site_reach(51.5138, -0.0984, reaches=reaches)
    assert site["order"] == 5
    assert site["discharge_m3s"] == 89.9
    assert site["distance_m"] == 380.0
    assert site["channels_within_km"] == 2
    assert "HydroRIVERS" in site["source"]


def test_assign_requires_channel_to_follow_the_reach():
    """Порядок получает река, вдоль которой идёт участок HydroRIVERS, а не ближайшая к точке."""
    points = [(-0.100, 51.500), (-0.095, 51.500), (-0.090, 51.500)]
    reach = {"order": 5, "discharge_m3s": 89.9, "distance_m": 300.0, "_points": points}
    along = _metric("Thames", "river", 460.0, [(-0.100, 51.5002), (-0.095, 51.5002),
                                              (-0.090, 51.5002)])
    aside = _metric("Fleet", "river", 404.0, [(-0.100, 51.509), (-0.095, 51.509)])
    dock = _metric("Dock", "dock", 200.0, [(-0.100, 51.5002), (-0.095, 51.5002)])
    warnings = hydro.assign([aside, along, dock], reach)
    assert along.order == 5 and along.discharge_m3s == 89.9
    assert aside.order is None      # далеко от участка — не «река здесь»
    assert dock.order is None       # доки и пруды порядка не получают
    assert warnings == []


def test_assign_refuses_to_guess_when_geometry_disagrees():
    reach = {"order": 5, "discharge_m3s": 89.9, "distance_m": 1500.0,
             "_points": [(-0.100, 51.520), (-0.090, 51.520)]}
    far = _metric("Fleet", "river", 300.0, [(-0.100, 51.500), (-0.090, 51.500)])
    warnings = hydro.assign([far], reach)
    assert far.order is None
    assert warnings and "site.hydro" in warnings[0]


def test_assign_without_source_says_so():
    metric = _metric("Thames", "river", 400.0, [(-0.1, 51.5), (-0.09, 51.5)])
    warnings = hydro.assign([metric], None)
    assert metric.order is None
    assert warnings and "HydroRIVERS" in warnings[0]
