"""Сквозная проверка навыка (S1, PR 2): настоящий пакет → `landform` → две интерпретации.

Проверяется контракт между репозиториями, а не числа: пакет схемы `site_pack/2`,
собранный офлайн из фикстур эталона (`--osm-file`, `--dem-file`, `--coast-file`,
`--worldcover-file`), доходит до навыка `vector-fengshui-bazi`, и в отчёте:

- по южному участку — две интерпретации и непустой раздел «Где школы расходятся»;
- по северному — одна интерпретация и пустой раздел расхождений.

Версия навыка закреплена коммитом (`tests/skill_pin.txt`, он же в шаге CI). Проверка идёт
в существующей задаче `test` (`pytest -m "not network"`): в CI навык уже склонирован на
закреплённом коммите и передан через `LANDFORM_REPO`, локально берётся соседний каталог или
кэш. Клон — зависимость проверки, а не измерение: без него в CI тест падает, а не пропускается.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

GOLDEN = Path(__file__).parent / "golden"

SKILL_URL = "https://github.com/Osmosy/vector-fengshui-bazi.git"
# Закреплён коммитом (ТЗ S1): навык меняется отдельно от пакета, и «последний main»
# делал бы проверку невоспроизводимой. Один пин на тест и шаг CI — `tests/skill_pin.txt`;
# LANDFORM_SHA переопределяет его без правки файла.
PIN_FILE = Path(__file__).parent / "skill_pin.txt"


def _pin() -> str:
    env = os.environ.get("LANDFORM_SHA")
    if env:
        return env.strip()
    pin = PIN_FILE.read_text(encoding="utf-8").strip()
    assert len(pin) >= 7, "tests/skill_pin.txt пуст: впишите коммит навыка"
    return pin

SOUTH = ("sydney", "rio-de-janeiro", "buenos-aires", "suva", "taveuni")
NORTH = ("beijing",)
SITES = SOUTH + NORTH

pytestmark = pytest.mark.skill


def _looks_like_skill(path: Path) -> bool:
    return (path / "scripts" / "fengshui_bazi_calc.py").exists() and \
           (path / "scripts" / "landform.py").exists()


def _skill_root() -> Path:
    """Каталог навыка: env → соседний репозиторий → кэш закреплённого коммита."""
    env = os.environ.get("LANDFORM_REPO")
    if env:
        path = Path(env).expanduser().resolve()
        if _looks_like_skill(path):
            return path
        pytest.fail(f"LANDFORM_REPO={env}: нет scripts/fengshui_bazi_calc.py")
    # рядом лежит рабочая копия того же навыка — локальный прогон без сети
    sibling = Path(__file__).resolve().parents[2] / "vector-fengshui-bazi"
    if _looks_like_skill(sibling):
        return sibling

    sha = _pin()
    cache = Path(os.environ.get("GEOSITE_SKILL_CACHE",
                                Path.home() / ".cache" / "vector-geo-site" / "skill"))
    dest = cache / sha[:12]
    if _looks_like_skill(dest):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not (dest / ".git").exists():
            subprocess.run(["git", "clone", "--quiet", SKILL_URL, str(dest)],
                           check=True, capture_output=True, text=True, timeout=300)
        inside = subprocess.run(["git", "-C", str(dest), "cat-file", "-e", f"{sha}^{{commit}}"],
                                capture_output=True, text=True)
        if inside.returncode != 0:
            subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "origin", sha],
                           check=True, capture_output=True, text=True, timeout=300)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", sha],
                       check=True, capture_output=True, text=True, timeout=120)
    except Exception as exc:          # нет сети и нет рабочей копии навыка
        message = (f"навык не получен по закреплённому коммиту {sha}: {exc}. "
                   "Локально: LANDFORM_REPO=../vector-fengshui-bazi")
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    return dest


@pytest.fixture(scope="session")
def skill() -> Path:
    return _skill_root()


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> dict[str, Path]:
    """Пакеты офлайн из фикстур: тот же путь, что у `geosite pack`, без сети вообще."""
    from geosite import cli

    out: dict[str, Path] = {}
    real_connect = socket.socket.connect

    def no_network(self, address):
        raise AssertionError(f"офлайн-сборка пакета полезла в сеть: {address}")

    socket.socket.connect = no_network
    try:
        for name in SITES:
            site = yaml.safe_load((GOLDEN / name / "site.yaml").read_text(encoding="utf-8"))
            fixtures = GOLDEN / name / "fixtures"
            target = tmp_path_factory.mktemp(f"pack-{name}")
            # координаты с пробелом после запятой: строка без пробела и с ведущим «−»
            # разбирается argparse как опция, и `pack` теряет позиционный аргумент
            args = ["pack", f"{site['lat']:g}, {site['lon']:g}", "--offline",
                    "--osm-file", str(fixtures / "osm-water.json.gz"),
                    "--dem-file", str(fixtures / "dem.tif"),
                    "--worldcover-file", str(fixtures / "worldcover.tif"),
                    "--out", str(target)]
            if (fixtures / "coast.json.gz").exists():
                args += ["--coast-file", str(fixtures / "coast.json.gz")]
            # W2: у правила моря ещё два входа; без них офлайн-пакет расходился с expect.yaml
            # того же эталона (Сингапур «моря нет», Сидней и Гонконг теряли «залив»)
            if (fixtures / "coast-wide.json.gz").exists():
                args += ["--coast-wide-file", str(fixtures / "coast-wide.json.gz")]
            if (fixtures / "sea.json.gz").exists():
                args += ["--sea-file", str(fixtures / "sea.json.gz")]
            if (fixtures / "osm-water-wide.json.gz").exists():
                args += ["--osm-file-wide", str(fixtures / "osm-water-wide.json.gz")]
            if site.get("facing") is not None:
                args += ["--facing", str(site["facing"])]
            assert cli.main(args) == 0, f"{name}: пакет не собран"
            out[name] = target / "site_pack.json"
            assert out[name].exists(), f"{name}: пакет не записан"
    finally:
        socket.socket.connect = real_connect
    return out


def _landform(skill: Path, pack: Path, *extra: str) -> dict:
    script = skill / "scripts" / "fengshui_bazi_calc.py"
    proc = subprocess.run([sys.executable, str(script), "landform", "--pack", str(pack), *extra],
                          cwd=str(skill / "scripts"), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"landform упал: {proc.stderr[-2000:]}"
    return json.loads(proc.stdout)


@pytest.mark.parametrize("name", SOUTH)
def test_south_pack_gets_two_interpretations(name: str, skill: Path, built: dict):
    """Южный пакет: две интерпретации, расходятся по стихиям и фасаду, карта зеркала есть."""
    data = _landform(skill, built[name], "--json")

    assert data["schema"] == "site_pack/2", data["schema"]
    assert data["hemisphere"] == "south"
    assert data["school_mode"] == "both"
    assert len(data["interpretations"]) == 2, [
        i["school"] for i in data["interpretations"]]
    classic, mirror = data["interpretations"]
    assert classic["school"] == "classic" and mirror["school"] == "mirror"

    # стихии севера и юга у школ разные — и основание названо по школе
    assert classic["elements"]["north"] != mirror["elements"]["north"]
    assert classic["elements"]["south"] != mirror["elements"]["south"]
    elements_text = {c["code"]: c["text"] for c in mirror["checks"]}
    assert "стихии по солнцу" in elements_text["SCHOOL-ELEMENTS"]
    assert "Огонь" in elements_text["SCHOOL-ELEMENTS"]

    topics = {row["topic"] for row in data["divergences"]}
    assert {"Стихия севера", "Стихия юга", "Предпочтительный фасад"} <= topics, topics
    assert mirror["preferred_facing"] == "север"
    # зеркало поворачивает ось карты: без года или оси строки нет, но при пакете с высотой
    # точки ось считается моделью, и карта обязана быть
    assert mirror["chart"] is not None or mirror["chart_note"], mirror


@pytest.mark.parametrize("name", NORTH)
def test_north_pack_gets_one_interpretation(name: str, skill: Path, built: dict):
    """Северный пакет: одна классическая интерпретация, расхождений нет."""
    data = _landform(skill, built[name], "--json")

    assert data["hemisphere"] == "north"
    assert data["school_mode"] == "classic"
    assert len(data["interpretations"]) == 1
    assert data["interpretations"][0]["school"] == "classic"
    assert data["divergences"] == []


def test_south_report_has_school_sections(skill: Path, built: dict):
    """Отчёт по южному пакету целиком: общие данные, две интерпретации, расхождения, оговорки."""
    script = skill / "scripts" / "fengshui_bazi_calc.py"
    proc = subprocess.run([sys.executable, str(script), "landform", "--pack",
                           str(built["sydney"])],
                          cwd=str(skill / "scripts"), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    md = proc.stdout
    assert "## Общие данные (одинаковы для обеих школ)" in md
    assert "## Интерпретация 1 — классическая школа" in md
    assert "## Интерпретация 2 — зеркальная школа (Южное полушарие)" in md
    assert "## Где школы расходятся" in md
    assert "не называется правильной" in md
