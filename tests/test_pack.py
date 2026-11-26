"""Пакет участка: схема, полнота, повторяемость на реальном DEM (Лондон).

Интеграционные тесты используют кеш тайлов Copernicus DEM; если тайлов нет — пропускаются,
чтобы не тянуть 60 МБ в обычном прогоне.
"""

import json
from pathlib import Path

import pytest

from geosite import geo, terrain
from geosite.dem import DEM_CACHE, tile_path
from geosite.models import SitePack, json_schema

LONDON = (51.5138, -0.0984)
EPSG = 32630
HAS_TILES = tile_path(51, 0).exists() and tile_path(51, -1).exists()


def test_schema_has_core_fields():
    schema = json_schema()
    props = schema["properties"]
    for key in ("schema", "query", "site", "orientation", "chart_year", "sectors", "quality", "attribution"):
        assert key in props, key
    assert props["schema"]["default"] == "site_pack/2"


def test_pack_roundtrip(tmp_path):
    from geosite.pack import build_pack

    # синтетический «замер»: маленькая сетка с одним холмом
    import numpy as np
    from affine import Affine

    from geosite.dem import DemGrid

    n, res = 101, 100.0
    data = np.zeros((n, n), dtype="float32")
    g = DemGrid(data, Affine(res, 0, 0, 0, -res, n * res), f"EPSG:{EPSG}", res)
    frame = terrain.polar_frame(g, 5000, 5000)
    measured = {
        "elev_m": 0.0, "elev_meso_m": 0.0,
        "sectors": [{"mountain": s.name, "rings": s.rings} for s in terrain.mountain_sectors(g, frame, 0.0)],
        "grid_macro": g, "x0": 5000, "y0": 5000,
        "point": {"slope_deg": 0.0, "aspect_deg": 0.0, "shape": "плоско / уступ", "elev_m": 0.0},
        "four_spirits": {"back": {"rel_elev_m": 0.0, "dist_km": 1.0, "continuity": 0.0},
                          "front": {"rel_elev_m": 0.0, "dist_km": 1.0, "continuity": 0.0},
                          "left": {"rel_elev_m": 0.0, "dist_km": 1.0, "continuity": 0.0},
                          "right": {"rel_elev_m": 0.0, "dist_km": 1.0, "continuity": 0.0},
                          "left_vs_right": 0.0},
        "dragon_vein": {"to_peak": None, "length_km": 0.0, "drop_m": 0.0, "bends": 0, "saddles": 0, "alive": False},
    }
    pack = build_pack(query={"input": "тест", "kind": "point", "resolved_by": "test", "alternatives": []},
                      site={"lat": 0.0, "lon": 0.0, "elev_m": 0.0, "precision": "point", "utm_epsg": EPSG},
                      orientation={"facing_true_deg": 180.0, "declination_deg": None, "facing_mag_deg": None,
                                   "facing_mountain": "午", "sitting_mountain": "子", "source": "user_compass",
                                   "confidence": "high", "near_boundary": False},
                      chart_year={"value": 2026, "source": "current_year"},
                      measured=measured, warnings=["тест"], notes=[], dem_resolution_m=100.0,
                      out_dir=tmp_path)
    assert (tmp_path / "site_pack.json").exists()
    assert (tmp_path / "figures" / "rose24.svg").exists()
    assert (tmp_path / "figures" / "horizon.svg").exists()
    assert pack.sectors and len(pack.sectors) == 24
    reloaded = SitePack.model_validate(json.loads((tmp_path / "site_pack.json").read_text()))
    assert reloaded.schema_id == "site_pack/2"
    # H4: в пакете нейтральные имена, имена школ присваивает навык
    payload = json.loads((tmp_path / "site_pack.json").read_text())
    assert set(payload["four_spirits"]) == {"back", "front", "left", "right", "left_vs_right"}
    for old_name in ("tortoise", "bird", "dragon", "tiger", "dragon_vs_tiger"):
        assert old_name not in payload["four_spirits"], old_name


@pytest.mark.skipif(not HAS_TILES, reason="нет тайлов Copernicus DEM в кеше")
def test_london_sectors_are_reproducible():
    """Критерий приёмки этапа 2: сектора повторяются при перезапуске."""
    runs = []
    for _ in range(2):
        m = terrain.measure_site(LONDON[0], LONDON[1], epsg=EPSG, resolution_m=30.0, macro_km=50.0)
        runs.append([(s["mountain"], [r["max_rel_elev_m"] for r in s["rings"]]) for s in m["sectors"]])
    assert runs[0] == runs[1]


@pytest.mark.skipif(not HAS_TILES, reason="нет тайлов Copernicus DEM в кеше")
def test_london_terrain_matches_known_ground():
    """Высоты Лондона: Сити 20–45 м (DSM с крышами), Темза ниже, север выше."""
    from geosite.dem import load_dem, smooth
    from pyproj import Transformer

    g = smooth(load_dem(*LONDON, radius_km=20, epsg=EPSG, resolution_m=90.0), 300.0)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
    city = g.point_value(*tr.transform(-0.0984, 51.5138))
    thames = g.point_value(*tr.transform(-0.1278, 51.5074))
    hampstead = g.point_value(*tr.transform(-0.1630, 51.5607))
    assert 15 < city < 45
    assert thames < city
    assert hampstead > city + 40


@pytest.mark.skipif(not HAS_TILES, reason="нет тайлов Copernicus DEM в кеше")
def test_london_vein_goes_uphill():
    m = terrain.measure_site(*LONDON, epsg=EPSG, resolution_m=30.0, macro_km=50.0, facing_deg=180.0)
    vein = m["dragon_vein"]
    assert vein["length_km"] > 2
    assert vein["drop_m"] > 20


def test_mouth_confidence_reaches_pack():
    """R1/E-009: уверенность врат и правило течения обязаны попадать в пакет."""
    from geosite.water import WaterMetrics
    from geosite.water_pack import to_objects

    metric = WaterMetrics(
        id="osm-A-river", name="A", kind="river", order=None, nearest_m=500.0,
        in_mountain="坤", out_mountain="辰", bend="outer", bend_kind="bend",
        curvature_radius_m=2000.0, straight_at_site=False, confluences=[],
        water_mouth={"mountain": "乙", "closure_index": 0.01, "closure_reason": None,
                     "closure_confidence": "low", "enclosure": 0.01, "width_m": 4850.0,
                     "left_hill_m": 5.7, "right_hill_m": 44.5, "width_capped": True,
                     "capped_sides": 1, "_x": 0.0, "_y": 0.0},
        outflow_visible=True, flow_osm_dir="downstream_toward_end",
        flow_dem_dir="downstream_toward_end", flow_chosen="downstream_toward_end",
        flow_rule=2, flow_confidence="medium", flow_agree=True, flow_ways_share=0.5,
        dem_slope_m_per_km=-0.083, dem_slope_snr=0.39, dem_reason=None,
        tidal=True, tidal_source="osm_tag", source="osm",
        loop=None, between_branches=False,
        crossings_2km=2, crossing_angle_deg=90.0,
    )
    obj = to_objects([metric])[0]
    assert obj.water_mouth is not None
    assert obj.water_mouth.closure_confidence == "low"
    assert obj.water_mouth.width_capped is True
    assert obj.water_mouth.capped_sides == 1
    assert obj.water_mouth.enclosure == 0.01
    assert obj.flow_rule == 2 and obj.flow_confidence == "medium"
    assert obj.flow_chosen == "downstream_toward_end"


def test_i13_confluence_participants_are_lines():
    """I-13: в слиянии участвуют только линии русла, никогда — контуры водоёмов.

    Инвариант проверяется по зафиксированным пакетам: если однажды контур берега снова
    попадёт в поиск слияний (E-017), пакет это покажет — участник будет с `geom: area`.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    checked = 0
    for pack_path in sorted(root.glob("examples/*/site_pack.json")):
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        by_id = {obj["id"]: obj for obj in pack.get("water") or []}
        for obj in pack.get("water") or []:
            for conf in obj.get("confluences") or []:
                assert obj.get("geom", "line") == "line", (pack_path.name, obj["id"])
                partner = by_id.get(conf["with"])
                assert partner is not None, (pack_path.name, obj["id"], conf["with"])
                assert partner.get("geom", "line") == "line", (pack_path.name, obj["id"], conf["with"])
                checked += 1
    assert checked > 0, "пакетов с слияниями не нашлось — проверять нечего"


def test_algo_is_built_from_module_versions():
    """P0: в ALGO нет ручных строк — все версии берутся из констант модулей."""
    from geosite import climate, geocode, pack as pack_mod, sun, terrain, water

    assert pack_mod.ALGO == {
        "water": water.ALGO_VERSION,
        "terrain": terrain.ALGO_VERSION,
        "geocode": geocode.ALGO_VERSION,
        "climate": climate.ALGO_VERSION,
        "sun": sun.ALGO_VERSION,
        "pack": pack_mod.ALGO_VERSION,
    }
    # версия сборки пакета — одна и та же в обоих местах
    assert pack_mod.ALGO["pack"] == pack_mod.ALGO_VERSION


def test_axis_does_not_depend_on_hemisphere():
    """H1: ось по модели нейтральна — гора на юге даёт «спину к югу» в обоих полушариях.

    Разница полушарий живёт в толковании (навык), а не в оси и не в геоданных: функция оси
    не принимает широту вовсе, и одинаковый рельеф даёт одинаковый фасад.
    """
    import inspect

    from geosite.cli import _model_facing

    signature = inspect.signature(_model_facing)
    assert "lat" not in signature.parameters and "hemisphere" not in signature.parameters

    def measured_with_hill(mountain: str) -> dict:
        rings = [{"max_rel_elev_m": 0.0}] * 3
        sectors = []
        for name in geo.MOUNTAINS:
            rings_here = [{"max_rel_elev_m": 120.0 if name == mountain else 5.0}] * 3
            sectors.append({"mountain": name, "rings": rings_here})
        return {"sectors": sectors}

    facing, why = _model_facing(measured_with_hill("午"))     # гора на юге
    assert facing == 0.0, (facing, why)                       # фасад на север, спина к югу
    assert "午" in why

    # та же гора и та же ось — независимо от того, где участок
    north = _model_facing(measured_with_hill("午"))[0]
    south = _model_facing(measured_with_hill("午"))[0]
    assert north == south == 0.0
