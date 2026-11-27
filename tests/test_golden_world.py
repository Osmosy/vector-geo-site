"""Мировые эталоны (G1): факты, которые не воспроизводятся в Европе.

Каждый эталон — каталог `tests/golden/<name>/`:

    site.yaml   — широта, долгота, радиус;
    expect.yaml — раздел `world:` с проверяемыми фактами и источником каждого;
    fixtures/   — ответ Overpass (8 км, с отношениями), берег, WorldCover 10 м, DEM 90 м.

Чем этот файл отличается от `test_golden.py`: тот проверяет воду и рельеф в метрах UTM, а тут
проверяется **выбор метода** — какая сетка взята и почему, на какой стороне полуденное солнце,
есть ли море в кольце, какая письменность в именах, работает ли прилив без тега. Это те места,
где ошибка не видна в числах: пакет собирается, `validate` чист, а метод при этом неверный.

Сеть запрещена: любой выход в сеть считается падением теста, а не «медленным прогоном».
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from geosite import geo, landcover, sea, sun, terrain, water

GOLDEN = Path(__file__).parent / "golden"

WORLD = sorted(p.name for p in GOLDEN.iterdir()
               if (p / "site.yaml").exists() and (p / "expect.yaml").exists()
               and "world" in (yaml.safe_load((p / "expect.yaml").read_text(encoding="utf-8")) or {}))


def _load_osm(fixtures: Path) -> dict | None:
    from geosite.cli import _load_osm_file

    for name in ("osm-water.json.gz", "osm-water.json"):
        path = fixtures / name
        if path.exists():
            return _load_osm_file(str(path))
    return None


def facts(name: str) -> dict:
    """Посчитать факты эталона на фикстурах. Сеть не трогается вообще."""
    site = yaml.safe_load((GOLDEN / name / "site.yaml").read_text(encoding="utf-8"))
    fixtures = GOLDEN / name / "fixtures"
    lat, lon = float(site["lat"]), float(site["lon"])
    radius_km = float(site.get("radius_km") or 8.0)
    # пакет выбирает сетку по масштабу места (20 км — адрес, 50 км — район/город), а рельеф
    # считает по вырезанному DEM; в эталоне обе величины записаны явно, чтобы проверять метод,
    # а не догадку о том, как место было распознано
    macro_km = float(site.get("macro_km") or 20.0)

    grid = geo.grid_crs(lat, lon, macro_km)
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(grid["crs"]), always_xy=True)
    site_xy = np.array(tr.transform(lon, lat))

    out: dict = {"grid": grid, "noon": sun.noon_side(lat, lon, year=2026)}

    dem_path = fixtures / "dem.tif"
    measured = {}
    raster = None
    if dem_path.exists():
        measured = terrain.measure_site(lat, lon, crs=grid["crs"], resolution_m=90.0,
                                        macro_km=radius_km, facing_deg=site.get("facing"),
                                        dem_path=dem_path)
        raster = measured.get("grid_macro")
    out["terrain"] = measured
    elev = (measured.get("point") or {}).get("elev_m")
    # NaN — это «не измерено», а не высота: сравнение nan > 100 молча проходит
    out["elev_m"] = None if elev is None or (isinstance(elev, float) and math.isnan(elev)) else elev

    payload = _load_osm(fixtures)
    coast = None
    coast_file = fixtures / "coast.json.gz"
    if coast_file.exists():
        from geosite.cli import _load_osm_file

        coast = _load_osm_file(str(coast_file))
    out["coast_m"] = water.nearest_coast_m(coast, lat, lon)
    out["sea_in_ring"] = bool(out["coast_m"] is not None and out["coast_m"] <= 2000.0)

    wide = None
    wide_file = fixtures / "osm-water-wide.json.gz"
    if wide_file.exists():
        from geosite.cli import _load_osm_file

        wide = _load_osm_file(str(wide_file))

    metrics = []
    if payload is not None:
        metrics = water.measure_water(lat, lon, crs=grid["crs"], grid=raster, site_xy=site_xy,
                                      payload=payload, radius_m=8000, wide_payload=wide,
                                      coast_payload=coast, allow_network=False)
    out["water_count"] = len(metrics)
    out["water_named"] = sum(1 for m in metrics if m.name)
    out["water_kinds"] = sorted({m.kind for m in metrics if m.kind})
    out["water_names"] = sorted({n for m in metrics for n in (getattr(m, "names", None) or [])})[:80]
    out["tidal"] = sum(1 for m in metrics if getattr(m, "tidal", None) is True)
    out["tidal_unseen"] = sum(1 for m in metrics if getattr(m, "tidal", None) is None)
    out["max_nearest_m"] = max((m.nearest_m or 0.0) for m in metrics) if metrics else None

    cover = None
    if raster is not None and (fixtures / "worldcover.tif").exists():
        cover = landcover.measure(lat, lon, crs=grid["crs"], site_xy=site_xy, radius_m=8000.0,
                                  path=fixtures / "worldcover.tif", offline=True)
    out["cover"] = cover
    out["water_wc"] = ((cover or {}).get("rings") or {}).get("0.5-2", {}).get("share")
    return out


def _check_world(name: str, expect: dict, got: dict) -> list[str]:
    """Проверить раздел `world:` эталона. Возвращает список расхождений, а не первый попавшийся."""
    fails: list[str] = []
    w = expect.get("world") or {}

    grid = w.get("grid") or {}
    if "kind" in grid and got["grid"]["kind"] != grid["kind"]:
        fails.append(f"сетка: {got['grid']['kind']} вместо {grid['kind']} "
                     f"(причина {got['grid'].get('reason')})")
    if "reason" in grid and got["grid"].get("reason") != grid["reason"]:
        fails.append(f"причина сетки: {got['grid'].get('reason')} вместо {grid['reason']}")

    noon = w.get("noon") or {}
    side = got["noon"]
    for key, bound in (("north_days_min", "days_noon_north"), ("south_days_min", "days_noon_south")):
        if key in noon and side[bound] < noon[key]:
            fails.append(f"полдень: {bound}={side[bound]} < {noon[key]}")
    for key, bound in (("north_days_max", "days_noon_north"), ("south_days_max", "days_noon_south")):
        if key in noon and side[bound] > noon[key]:
            fails.append(f"полдень: {bound}={side[bound]} > {noon[key]}")

    coast = w.get("coast") or {}
    if "sea_in_ring" in coast and got["sea_in_ring"] != bool(coast["sea_in_ring"]):
        fails.append(f"море в кольце: {got['sea_in_ring']}, берег {got['coast_m']} м")
    if "max_m" in coast:
        dist = got["coast_m"]
        if dist is None or dist > coast["max_m"]:
            fails.append(f"берег: {dist} м, ожидалось не дальше {coast['max_m']}")

    wat = w.get("water") or {}
    if "count_min" in wat and got["water_count"] < wat["count_min"]:
        fails.append(f"объектов воды: {got['water_count']} < {wat['count_min']}")
    if "named_min" in wat and got["water_named"] < wat["named_min"]:
        fails.append(f"имён воды: {got['water_named']} < {wat['named_min']}")
    if "kind_any" in wat:
        want = set(wat["kind_any"])
        if not want & set(got["water_kinds"]):
            fails.append(f"виды воды {got['water_kinds']} не пересекаются с {sorted(want)}")
    if "name_any" in wat:
        haystack = " | ".join(got["water_names"])
        if not any(part.lower() in haystack.lower() for part in wat["name_any"]):
            fails.append(f"в именах воды нет ни одного из {wat['name_any']}: {haystack[:160]}")
    if "tidal_min" in wat and got["tidal"] < wat["tidal_min"]:
        fails.append(f"приливных объектов: {got['tidal']} < {wat['tidal_min']}")

    lc = w.get("landcover") or {}
    if lc.get("absent") and got["water_wc"] is not None:
        fails.append(f"покров: ожидалось вне покрытия, а вода {got['water_wc']}")
    if "water_wc_min" in lc:
        share = got["water_wc"]
        if share is None or share < lc["water_wc_min"]:
            fails.append(f"покров: вода {share} < {lc['water_wc_min']}")

    dem = w.get("dem") or {}
    if "elev_max" in dem and (got["elev_m"] is None or got["elev_m"] > dem["elev_max"]):
        fails.append(f"высота точки: {got['elev_m']} > {dem['elev_max']}")
    if "elev_min" in dem and (got["elev_m"] is None or got["elev_m"] < dem["elev_min"]):
        fails.append(f"высота точки: {got['elev_m']} < {dem['elev_min']}")
    if dem.get("void") and got["elev_m"] is not None:
        fails.append(f"ожидалась пустота DEM, измерено {got['elev_m']} м")
    return fails


@pytest.mark.skipif(not WORLD, reason="нет мировых эталонов")
@pytest.mark.parametrize("name", WORLD)
def test_world_golden(name: str, monkeypatch):
    """Эталон проходит на сохранённых данных; сеть запрещена и падает как ошибка, а не как таймаут."""
    monkeypatch.setenv("GEOSITE_OFFLINE", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("эталонный прогон полез в сеть — фикстуры неполные")

    monkeypatch.setattr(water, "_ask", no_network)

    expect = yaml.safe_load((GOLDEN / name / "expect.yaml").read_text(encoding="utf-8")) or {}
    fixtures = GOLDEN / name / "fixtures"
    if not (fixtures / "osm-water.json.gz").exists() and not (fixtures / "osm-water.json").exists():
        pytest.skip(f"{name}: нет фикстур — соберите scripts/make_golden.py {name}")
    if (expect.get("status") or "").lower() == "blocked":
        pytest.skip(expect.get("reason") or "не хватает фикстур")
    got = facts(name)
    fails = _check_world(name, expect, got)
    assert not fails, f"{name}: " + "; ".join(fails)


@pytest.mark.skipif(not WORLD, reason="нет мировых эталонов")
@pytest.mark.parametrize("name", WORLD)
def _gzip_json(path: Path) -> dict | None:
    import gzip
    import json

    if not path.exists():
        return None
    return json.loads(gzip.open(path, "rt", encoding="utf-8").read())


@pytest.mark.parametrize("name", WORLD)
def test_world_sea_by_evidence(name: str):
    """Море (S2/E-038): ОСМ-признаки и снимок сверяются, а не берётся одна береговая линия.

    Ожидание в `expect.yaml` — с источником (карта, лоция). Проверка идёт по фикстурам:
    берег 8 км, берег 15 км и отдельный ответ Overpass с признаками моря.
    """
    expect = yaml.safe_load((GOLDEN / name / "expect.yaml").read_text(encoding="utf-8")) or {}
    exp = (expect.get("world") or {}).get("sea")
    if not exp:
        pytest.skip(f"{name}: ожидание моря не задано")
    fixtures = GOLDEN / name / "fixtures"
    site = yaml.safe_load((GOLDEN / name / "site.yaml").read_text(encoding="utf-8")) or {}
    lat, lon = float(site["lat"]), float(site["lon"])
    coast = _gzip_json(fixtures / "coast.json.gz")
    wide = _gzip_json(fixtures / "coast-wide.json.gz")
    payload = _gzip_json(fixtures / "sea.json.gz")
    polygons = sea.payload_polygons(payload, lat, lon) if payload else []
    got = sea.detect(water=[], coast_m=water.nearest_coast_m(coast, lat, lon),
                     coast_wide_m=water.nearest_coast_m(wide, lat, lon),
                     wc_share=exp.get("wc_share"), gsw_share=exp.get("gsw_share"),
                     osm_water_share=exp.get("osm_water_share"), sea_polygons=polygons)
    assert got["present"] is exp["present"], (
        f"{name}: море {got['present']}, ожидали {exp['present']}; признаки "
        f"{[e['kind'] for e in got['evidence']]}")
    if exp.get("kind"):
        assert got["kind"] == exp["kind"], f"{name}: вид моря {got['kind']}, ждали {exp['kind']}"
    kinds = {e["kind"] for e in got["evidence"]}
    assert len(kinds) >= int(exp.get("min_kinds", 1)), (
        f"{name}: признаков {sorted(kinds)}, ждали минимум {exp.get('min_kinds')}")
    if exp.get("nearest_max_m") is not None and got["nearest_m"] is not None:
        assert got["nearest_m"] <= float(exp["nearest_max_m"])


@pytest.mark.parametrize("name", WORLD)
def test_world_fixtures_are_offline(name: str):
    """Фикстура обязана быть: эталон без неё — обещание, а не проверка."""
    fixtures = GOLDEN / name / "fixtures"
    assert (fixtures / "osm-water.json.gz").exists() or (fixtures / "osm-water.json").exists(), \
        f"{name}: нет выгрузки OSM — соберите scripts/make_golden.py {name}"
    expect = yaml.safe_load((GOLDEN / name / "expect.yaml").read_text(encoding="utf-8")) or {}
    world = expect.get("world") or {}
    # «нет данных» — законный факт, если он записан в эталоне: у полярной точки вне покрытия
    # WorldCover покрова не будет никогда, и требовать файл значит требовать выдумку
    if not (world.get("landcover") or {}).get("absent"):
        assert (fixtures / "worldcover.tif").exists(), f"{name}: нет покрова WorldCover"
    if not (world.get("dem") or {}).get("void"):
        assert (fixtures / "dem.tif").exists(), f"{name}: нет вырезанного DEM"


@pytest.mark.skipif(not WORLD, reason="нет мировых эталонов")
def test_world_coverage_documented():
    """Список эталонов в docs/calc/world_coverage.md обязан совпадать с каталогами."""
    doc = Path(__file__).resolve().parents[1] / "docs" / "calc" / "world_coverage.md"
    if not doc.exists():
        pytest.fail("нет docs/calc/world_coverage.md — таблица покрытия мира")
    text = doc.read_text(encoding="utf-8")
    missing = [name for name in WORLD if f"`{name}`" not in text]
    assert not missing, f"в world_coverage.md нет эталонов: {missing}"
