"""Поля примеров: то, что видно без сборки мировых пакетов.

Файл отделён от `test_pack_world_fields.py` намеренно: там подготовка модуля собирает пять
эталонов за ~2.5 минуты, и «быстрый прогон» через `--deselect` вместе с ней снимал бы и эти
проверки. Здесь — только чтение `examples/` и вызовы модулей: секунды, и они остаются в
быстром прогоне.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _pack(name: str) -> dict:
    return json.loads((EXAMPLES / name / "site_pack.json").read_text(encoding="utf-8"))


def test_bend_kind_reaches_example() -> None:
    """W3/E-042: вид изгиба в пакете примера, а не только в схеме."""
    pack = _pack("london-city")
    thames = next(w for w in pack["water"]
                  if w.get("name") == "River Thames" and w.get("geom") == "line")
    assert thames.get("bend_kind") == "loop", thames.get("bend_kind")
    geojson = json.loads((EXAMPLES / "london-city" / "water.geojson").read_text())
    assert any(f["properties"].get("bend_kind") == "loop" for f in geojson["features"])


def test_algo_lists_every_module_with_version() -> None:
    """W4/E-043: у каждого модуля с `ALGO_VERSION` есть ключ в `pack.ALGO`.

    Иначе данные модуля в пакете есть, а версию проверить нельзя: I-11 сверяет только то,
    что записано.
    """
    import geosite

    from geosite import pack as pack_mod

    missing = []
    for path in sorted(Path(geosite.__file__).parent.glob("*.py")):
        name = path.stem
        if name.startswith("__"):
            continue
        try:
            module = importlib.import_module(f"geosite.{name}")
        except Exception:                                   # модуль с необязательными зависимостями
            continue
        if getattr(module, "ALGO_VERSION", None) is None:
            continue
        key = "pack" if name == "pack" else name
        if key not in pack_mod.ALGO:
            missing.append(name)
    assert not missing, f"модули с версией вне pack.ALGO: {missing}"
    assert set(pack_mod.ALGO) <= set(_pack("london-city")["algo"])


def test_declination_date_is_recorded_and_pinnable() -> None:
    """W5/E-044: дата склонения в пакете, пин даёт одинаковое число.

    Без даты пересборка в другой день меняет `declination_deg` без следа в пакете;
    `GEOSITE_DECLINATION_YEAR` закрепляет дату, и две сборки равны.
    """
    got = _pack("london-city")["orientation"]
    assert got.get("declination_year") == pytest.approx(2026.91, abs=0.01), got

    from geosite import climate

    old = os.environ.get("GEOSITE_DECLINATION_YEAR")
    os.environ["GEOSITE_DECLINATION_YEAR"] = "2026.5"
    try:
        first = climate.declination(51.5138, -0.0984)
        second = climate.declination(51.5138, -0.0984)
    finally:
        if old is None:
            os.environ.pop("GEOSITE_DECLINATION_YEAR", None)
        else:
            os.environ["GEOSITE_DECLINATION_YEAR"] = old
    assert first == second, (first, second)
    assert first["year"] == pytest.approx(2026.5, abs=0.01)


def test_example_orientation_is_model_not_measurement() -> None:
    """E-006: ориентация примера не выдана за компасный замер.

    У `london-city` ось выведена моделью (`source: model`, `confidence: low`); у
    `london-city-compass` она задана вручную и в `examples/README.md` помечена демонстрацией —
    иначе «180° по компасу» читалось бы как измерение.
    """
    model = _pack("london-city")["orientation"]
    assert model["source"] == "model" and model["confidence"] == "low", model
    assert model["facing_true_deg"] == pytest.approx(135.0, abs=0.01)

    compass = _pack("london-city-compass")["orientation"]
    assert compass["source"] == "user_compass", compass
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    row = next(l for l in readme.splitlines() if "london-city-compass" in l)
    assert "демонстрац" in row.lower() or "демо" in row.lower(), row


def test_example_pack_is_one_water_object_per_line() -> None:
    """E-007: пакет не раздут — вода по объекту на строку, файл в пределах 3000 строк.

    Было 20 858 строк на 799 объектов: диффы нечитаемы, а «изменённый объект — изменённая
    строка» не выполнялось.
    """
    for name in ("london-city", "london-city-compass"):
        text = (EXAMPLES / name / "site_pack.json").read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) <= 3000, (name, len(lines))
        # `dem_layer` не первый ключ у объектов, где он пуст, — считаем по любому объекту
        water_lines = [l for l in lines if l.lstrip().startswith('{"')]
        assert len(water_lines) == len(_pack(name)["water"]), (name, len(water_lines))
        for line in water_lines[:5]:
            # в файле объекты идут через запятую — для проверки строки она снимается
            assert isinstance(json.loads(line.strip().rstrip(",")), dict)
