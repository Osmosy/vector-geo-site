"""W1–W5: поля пакета, которые терялись между слоем метрик и файлом.

Проверка идёт через путь пакета (`cli.main(["pack", ...])` по фикстурам эталона), а не через
`facts()` эталона: W1 и W2 видны только на этом пути — эталон читает слой метрик, а пакет
собирает объекты и выбрасывает у контуров речные поля.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
import yaml

GOLDEN = Path(__file__).parent / "golden"
SITES = ("singapore", "sydney", "hong-kong", "buenos-aires", "suva")
PINNED_YEAR = 2026.5


def _no_network(self, address):  # noqa: ANN001
    raise AssertionError(f"офлайн-сборка полезла в сеть: {address}")


def _build(name: str, target: Path, *, with_sea_inputs: bool = True,
           skip: tuple[str, ...] = ()) -> Path:
    """Собрать пакет эталона из фикстур: тот же набор флагов, что у `geosite pack`."""
    from geosite import cli

    site = yaml.safe_load((GOLDEN / name / "site.yaml").read_text(encoding="utf-8"))
    fx = GOLDEN / name / "fixtures"
    args = ["pack", f"{site['lat']:g}, {site['lon']:g}", "--offline",
            "--osm-file", str(fx / "osm-water.json.gz"),
            "--dem-file", str(fx / "dem.tif"),
            "--worldcover-file", str(fx / "worldcover.tif"),
            "--out", str(target)]
    if with_sea_inputs:
        for flag, fname in (("--coast-file", "coast.json.gz"),
                            ("--coast-wide-file", "coast-wide.json.gz"),
                            ("--sea-file", "sea.json.gz"),
                            ("--osm-file-wide", "osm-water-wide.json.gz")):
            if flag in skip:
                continue
            if (fx / fname).exists():
                args += [flag, str(fx / fname)]
    original = socket.socket.connect
    socket.socket.connect = _no_network
    try:
        assert cli.main(args) == 0, f"{name}: пакет не собран"
    finally:
        socket.socket.connect = original
    out = target / "site_pack.json"
    assert out.exists(), f"{name}: пакет не записан"
    return out


@pytest.fixture(scope="module")
def packs(tmp_path_factory) -> dict[str, dict]:
    """Пакеты четырёх эталонов: сборка дорогая, поэтому один раз на модуль.

    Значение — `{"data": site_pack.json, "dir": каталог пакета}`: часть проверок смотрит
    `water.geojson`, который лежит рядом с пакетом.
    """
    cache = tmp_path_factory.mktemp("cache")
    old_cache = os.environ.get("GEOSITE_CACHE")
    old_year = os.environ.get("GEOSITE_DECLINATION_YEAR")
    os.environ["GEOSITE_CACHE"] = str(cache)
    os.environ["GEOSITE_DECLINATION_YEAR"] = str(PINNED_YEAR)
    try:
        out: dict[str, dict] = {}
        for name in SITES:
            target = tmp_path_factory.mktemp(f"pack-{name}")
            path = _build(name, target)
            out[name] = {"data": json.loads(path.read_text(encoding="utf-8")), "dir": target}
        return out
    finally:
        for key, old in (("GEOSITE_CACHE", old_cache),
                         ("GEOSITE_DECLINATION_YEAR", old_year)):
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@pytest.mark.parametrize("name", ("singapore", "sydney", "hong-kong"))
def test_sea_matches_expect(name: str, packs: dict) -> None:
    """W2/E-041: офлайн-пакет не расходится с `expect.yaml` того же эталона.

    Сингапур: океан в 9 км — признак `coastline_wide`; Сидней и Гонконг: залив задан
    полигонами `natural=bay`, без файла признаков вид падал до «океана».
    """
    expect = yaml.safe_load((GOLDEN / name / "expect.yaml").read_text(encoding="utf-8"))
    want = expect["world"]["sea"]
    got = packs[name]["data"]["sea"]
    assert got["present"] is want["present"], (name, got["present"], got.get("note"))
    assert got["kind"] == want["kind"], (name, got["kind"], want["kind"])
    kinds = {e["kind"] for e in got["evidence"]}
    assert len(kinds) >= int(want.get("min_kinds", 1)), (name, sorted(kinds))


def test_sea_unknown_without_evidence(tmp_path) -> None:
    """W2: без входов правила моря ответ «моря нет» был бы выдумкой — пишется null и причина."""
    pack = json.loads(_build("sydney", tmp_path / "pack", with_sea_inputs=False)
                      .read_text(encoding="utf-8"))
    assert pack["sea"]["present"] is None, pack["sea"]
    assert pack["sea"]["confidence"] == "low"
    assert "не получены" in (pack["sea"].get("note") or ""), pack["sea"].get("note")
    warnings = " ".join(pack["quality"]["warnings"])
    assert "признаки не получены" in warnings, warnings


def test_area_keeps_tidal_in_pack(packs: dict) -> None:
    """W1/E-040: прилив у контура водоёма доходит до пакета — Ла-Плата больше не молчит.

    У контура остаются только поля водоёма, но `tidal` — свойство воды: правило G3.4 делает
    приливной любую воду у берега, и море (S2) считает такие объекты своим признаком.
    """
    pack = packs["buenos-aires"]["data"]
    # Ла-Плата — площадь, и она приливная: раньше её `tidal` выбрасывался по дороге в файл
    plata = next(w for w in pack["water"] if "Plata" in (w.get("name") or ""))
    assert plata.get("geom") == "area" and plata.get("tidal") is True, plata
    assert plata.get("tidal_source"), plata

    # полный набор — в water.geojson: 48 контуров и 5 линий с приливом = 53 слоя метрик
    geojson = json.loads((packs["buenos-aires"]["dir"] / "water.geojson").read_text())
    tidal = [f["properties"] for f in geojson["features"] if f["properties"].get("tidal") is True]
    areas = [p for p in tidal if p.get("geom") == "area"]
    lines = [p for p in tidal if p.get("geom") == "line"]
    assert len(areas) >= 40 and len(lines) >= 3, (len(areas), len(lines))
    assert len(tidal) == len(areas) + len(lines) == 53, (len(tidal), len(areas), len(lines))
    # речные поля у контура по-прежнему отсутствуют, а не лежат нулями
    for item in areas:
        assert item.get("tidal_source"), item
        for field in ("bend", "bend_kind", "loop", "flow_rule", "water_mouth"):
            assert field not in item, (field, item.get("id"))



def test_near_ring_dsm_dtm_twin_is_not_published() -> None:
    """E-037: пара dsm/dtm в ближнем кольце не дублирует одно число под двумя именами.

    Когда DTM есть только в ближнем кольце (`scope: near_ring`), кольцо сектора посчитано по
    DSM; публиковать рядом «dtm» с тем же значением — значит выдать копию за второй слой.
    """
    from geosite import pack

    measured = {
        "dsm_near_ring": {"子": {"max_rel_elev_m": 5.0, "mean_rel_elev_m": 3.0, "horizon_deg": 1.0}},
        "dtm_info": {"source": "ea-lidar", "scope": "near_ring"},
        "sectors": [{"mountain": "子", "rings": [
            {"max_rel_elev_m": 5.0, "mean_rel_elev_m": 3.0, "horizon_deg": 1.0}]}],
    }
    assert pack._near_ring_layers(measured, "子", 0) == {}

    measured["dtm_info"]["scope"] = "all"
    pair = pack._near_ring_layers(measured, "子", 0)
    assert set(pair) == {"dsm", "dtm"}, pair
    assert pair["dsm"]["max_rel_elev_m"] == pair["dtm"]["max_rel_elev_m"] == 5.0


def test_sea_without_wide_coast_is_not_an_absence(tmp_path) -> None:
    """E-045: забытый `--coast-wide-file` не должен давать молчаливое «моря нет».

    У Сингапура океан в 8.97 км: в ближнем кольце берега нет, и без широкого входа правило
    видело «брега не было» там, где его не спрашивали.
    """
    path = _build("singapore", tmp_path, skip=("--coast-wide-file",))
    pack = json.loads(path.read_text(encoding="utf-8"))
    assert pack["sea"]["present"] is None, pack["sea"]
    assert pack["sea"]["confidence"] == "low"
    assert "--coast-wide-file" in (pack["sea"].get("note") or "")
    warnings = " ".join(pack["quality"]["warnings"])
    assert "--coast-wide-file" in warnings, warnings


def test_sea_without_polygons_does_not_guess_the_kind(tmp_path) -> None:
    """E-045: без `--sea-file` вид моря — догадка, а не измерение.

    Берег в 8 км есть, а признаков `natural=bay` нет: Сидней получал `kind: ocean` с
    уверенностью `high`, хотя залив Порт-Джексон нарисован именно полигоном.
    """
    path = _build("sydney", tmp_path, skip=("--sea-file",))
    pack = json.loads(path.read_text(encoding="utf-8"))
    sea = pack["sea"]
    assert sea["present"] is True, sea
    assert sea["confidence"] == "low", sea
    assert sea.get("kind") is None, sea
    assert "--sea-file" in (sea.get("note") or "")


def test_tidal_survives_without_near_coast_file(packs: dict, tmp_path) -> None:
    """E-046: прилив не пропадает, когда ближнего берега нет, а широкий есть.

    Восьмикилометровый берег после E-045 законно необязателен, но правило G3.4 читало только
    его: набор «широкий берег + --sea-file» давал пакет без единого приливного объекта, при
    здоровом блоке `sea`. Теперь правило получает объединение обоих кругов.
    """
    full = json.loads((packs["suva"]["dir"] / "water.geojson").read_text())
    full_tidal = [f["properties"] for f in full["features"]
                  if f["properties"].get("tidal") is True]
    assert len(full_tidal) >= 50, len(full_tidal)

    path = _build("suva", tmp_path, skip=("--coast-file",))
    partial_dir = path.parent
    geojson = json.loads((partial_dir / "water.geojson").read_text())
    tidal = [f["properties"] for f in geojson["features"] if f["properties"].get("tidal") is True]
    assert len(tidal) == len(full_tidal), (len(tidal), len(full_tidal))
    assert all(p.get("tidal_source") == "coastline" for p in tidal), tidal[0]
    pack = json.loads(path.read_text(encoding="utf-8"))
    assert pack["sea"]["present"] is True, pack["sea"]
