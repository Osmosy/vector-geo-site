"""Эталонные места: проверка фактов из expect.yaml на офлайн-фикстурах.

Каждый эталон — каталог tests/golden/<name>/:
  site.yaml   — координаты, радиус, ориентация;
  expect.yaml — проверяемые факты (диапазоны и наборы, не точные числа) и источники;
  fixtures/   — сохранённый ответ Overpass и вырезанный DEM.

Тесты не ходят в сеть. Если фикстур нет — эталон пропускается с явной причиной.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from geosite import geo, terrain, water

GOLDEN = Path(__file__).parent / "golden"
SITES = sorted(p.name for p in GOLDEN.iterdir() if (p / "site.yaml").exists()) if GOLDEN.exists() else []


def load(name: str) -> tuple[dict, dict | None, Path]:
    site = yaml.safe_load((GOLDEN / name / "site.yaml").read_text(encoding="utf-8"))
    expect_file = GOLDEN / name / "expect.yaml"
    expect = yaml.safe_load(expect_file.read_text(encoding="utf-8")) if expect_file.exists() else None
    return site, expect, GOLDEN / name / "fixtures"


def measure(site: dict, fixtures: Path, *, with_dem: bool = True) -> dict:
    """Полный замер на фикстурах: рельеф из DEM-файла, вода из сохранённого ответа."""
    from pyproj import Transformer

    lat, lon = float(site["lat"]), float(site["lon"])
    epsg = geo.utm_epsg(lat, lon)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    site_xy = np.array(tr.transform(lon, lat))

    if with_dem:
        grid_measured = terrain.measure_site(
            lat, lon, epsg=epsg, resolution_m=90.0, macro_km=float(site.get("radius_km", 20.0)),
            facing_deg=site.get("facing"), dem_path=fixtures / "dem.tif")
        grid = grid_measured["grid_macro"]
    else:
        # без DEM проверяем только воду: рельефные поля остаются None, а не выдуманными
        grid_measured = {"grid_macro": None}
        grid = None

    payload = yaml.safe_load("{}") or None
    import json

    payload = json.loads((fixtures / "osm-water.json").read_text(encoding="utf-8"))
    import gzip

    wide_file = fixtures / "osm-water-wide.json.gz"
    wide = None
    if wide_file.exists():
        with gzip.open(wide_file, "rt", encoding="utf-8") as fh:
            wide = json.load(fh)
    metrics = water.measure_water(lat, lon, epsg=epsg, grid=grid, site_xy=site_xy,
                                  payload=payload, radius_m=8000, wide_payload=wide,
                                  allow_network=False)
    # порядок реки и расход: HydroRIVERS, офлайн из фикстуры (иначе тест требовал бы 150 МБ шейп-файла)
    reach = None
    hydro_file = fixtures / "hydrorivers.json"
    if hydro_file.exists():
        from geosite import hydro

        saved = json.loads(hydro_file.read_text(encoding="utf-8"))
        reaches = [dict(r, _points=[tuple(pt) for pt in r.get("points", [])])
                   for r in saved.get("reaches", [])]
        reach = hydro.site_reach(lat, lon, reaches=reaches)
        hydro.assign(metrics, reach)
    # одна и та же река бывает склеена в несколько линий: берём ближайшую к участку
    by_name: dict[str, object] = {}
    for metric in metrics:
        if not metric.name:
            continue
        current = by_name.get(metric.name)
        if current is None or metric.nearest_m < current.nearest_m:
            by_name[metric.name] = metric
    return {"terrain": grid_measured, "water": by_name, "hydro": reach}


@pytest.mark.skipif(not SITES, reason="нет эталонов")
@pytest.mark.parametrize("name", SITES)
def test_golden_site(name: str):
    site, expect, fixtures = load(name)
    if expect is None:
        pytest.skip(f"{name}: нет expect.yaml — факты ещё не сверены")
    if not (fixtures / "osm-water.json").exists():
        pytest.skip(f"{name}: нет OSM-фикстуры, соберите её scripts/make_golden.py")
    has_dem = (fixtures / "dem.tif").exists()
    if not has_dem and (expect.get("water_mouth") or "closure_index_min" in str(expect)):
        pytest.skip(f"{name}: нужен DEM для проверки врат, а фикстуры нет")

    if (expect.get("status") or "").lower() == "blocked":
        pytest.skip(expect.get("reason") or "нет данных для эталона")
    if (expect.get("status") or "").lower() == "xfail":
        # факт верен, код пока не дотягивает; причина записана в expect.yaml и в errata
        pytest.xfail(expect.get("reason") or "известный дефект")

    result = measure(site, fixtures, with_dem=has_dem)
    facts = expect.get("water") or {}
    failures: list[str] = []

    hydro_facts = expect.get("hydrorivers") or {}
    if hydro_facts:
        reach = result.get("hydro") or {}
        if not reach:
            failures.append("hydrorivers: участок не выбран, а факты заданы")
        else:
            if "order" in hydro_facts and reach.get("order") != hydro_facts["order"]:
                failures.append(f"hydrorivers.order={reach.get('order')}, ждали {hydro_facts['order']}")
            if "discharge_m3s_range" in hydro_facts:
                low, high = hydro_facts["discharge_m3s_range"]
                got = reach.get("discharge_m3s")
                if got is None or not (low <= got <= high):
                    failures.append(f"hydrorivers.discharge_m3s={got} вне {hydro_facts['discharge_m3s_range']}")
            if "distance_max_m" in hydro_facts:
                got = reach.get("distance_m")
                if got is None or got > hydro_facts["distance_max_m"]:
                    failures.append(f"hydrorivers.distance_m={got} больше {hydro_facts['distance_max_m']}")

    for river_name, checks in facts.items():
        metric = result["water"].get(river_name)
        if metric is None:
            failures.append(f"{river_name}: не найден в пакете")
            continue
        for key, expected in (checks or {}).items():
            if key == "nearest_m_range":
                low, high = expected
                if not (low <= metric.nearest_m <= high):
                    failures.append(f"{river_name}.nearest_m={metric.nearest_m} вне {expected}")
            elif key == "bend":
                if metric.bend != expected:
                    failures.append(f"{river_name}.bend={metric.bend!r}, ждали {expected!r}")
            elif key == "bend_kind":
                if metric.bend_kind != expected:
                    failures.append(f"{river_name}.bend_kind={metric.bend_kind!r}, ждали {expected!r}")
            elif key == "loop":
                has_loop = metric.loop is not None
                if expected == "absent" and has_loop:
                    failures.append(f"{river_name}: петля найдена, а её быть не должно")
                if expected == "present" and not has_loop:
                    failures.append(f"{river_name}: петли нет, а она ожидалась")
            elif key == "site_position":
                if metric.site_position != expected:
                    failures.append(f"{river_name}.site_position={metric.site_position!r}, "
                                    f"ждали {expected!r}")
            elif key == "site_inside":
                inside = bool((metric.loop or {}).get("site_inside"))
                if inside is not expected:
                    failures.append(f"{river_name}.loop.site_inside={inside}, ждали {expected}")
            elif key == "tidal":
                if metric.tidal is not expected:
                    failures.append(f"{river_name}.tidal={metric.tidal}, ждали {expected}")
            elif key == "flow_chosen":
                if metric.flow_chosen != expected:
                    failures.append(f"{river_name}.flow_chosen={metric.flow_chosen!r}, ждали {expected!r}")
            elif key == "out_mountain_any":
                allowed = set(expected)
                if metric.out_mountain not in allowed:
                    failures.append(f"{river_name}.out_mountain={metric.out_mountain}, ждали из {sorted(allowed)}")
            elif key == "in_mountain_any":
                allowed = set(expected)
                if metric.in_mountain not in allowed:
                    failures.append(f"{river_name}.in_mountain={metric.in_mountain}, ждали из {sorted(allowed)}")
            elif key == "confluences_min":
                if len(metric.confluences) < int(expected):
                    failures.append(f"{river_name}: слияний {len(metric.confluences)}, ждали ≥ {expected}")
            elif key == "visible":
                if metric.visible is not expected:
                    failures.append(f"{river_name}.visible={metric.visible}, ждали {expected}")
            elif key == "hidden_share_min":
                share = metric.hidden_share
                if share is None or share < float(expected):
                    failures.append(f"{river_name}.hidden_share={share}, ждали ≥ {expected}")
            elif key == "order":
                if metric.order != expected:
                    failures.append(f"{river_name}.order={metric.order}, ждали {expected}")
            elif key == "discharge_m3s_range":
                low, high = expected
                got = metric.discharge_m3s
                if got is None or not (low <= got <= high):
                    failures.append(f"{river_name}.discharge_m3s={got} вне {expected}")
            elif key == "closure_index_min":
                mouth = metric.water_mouth or {}
                value = mouth.get("closure_index")
                if value is None or value < float(expected):
                    failures.append(f"{river_name}.closure_index={value}, ждали ≥ {expected}")
            else:
                failures.append(f"{river_name}: неизвестный факт {key!r}")

    for key, expected in (expect.get("water_mouth") or {}).items():
        rivers = list(result["water"].values())
        mouth = next((r.water_mouth for r in rivers if r.water_mouth), None) or {}
        if mouth.get(key) != expected:
            failures.append(f"water_mouth.{key}={mouth.get(key)!r}, ждали {expected!r}")

    assert not failures, "; ".join(failures)
