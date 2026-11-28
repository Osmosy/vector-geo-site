#!/usr/bin/env python3
"""Таблица покрытия мира: воспроизведение `docs/calc/world_coverage.md` по фикстурам.

Считает по **слою метрик** — тем же путём, что `tests/test_golden_world.py::facts`, а не по
содержимому пакета: у Амстердама метрик 1940, а в пакете 205, и путать эти числа нельзя.

Запуск из корня репозитория:

    .venv/bin/python scripts/world_coverage_table.py            # markdown-таблица
    .venv/bin/python scripts/world_coverage_table.py --diff     # сверка с docs/calc/world_coverage.md

Сеть не нужна: всё читается из `tests/golden/<name>/fixtures`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

HEADER = ("| Эталон | Широта, долгота | Сетка | Полдень, дней С/Ю | Море в кольце | "
          "Вода OSM: объектов / имён / приливных |")
DIVIDER = "|---|---|---|---|---|---|"


def rows() -> list[dict]:
    import yaml

    import test_golden_world as tgw

    out = []
    for name in tgw.WORLD:
        facts = tgw.facts(name)
        site = yaml.safe_load((tgw.GOLDEN / name / "site.yaml").read_text(encoding="utf-8"))
        expect_path = tgw.GOLDEN / name / "expect.yaml"
        expect = yaml.safe_load(expect_path.read_text(encoding="utf-8")) or {}
        # «Море в кольце» — закреплённый факт эталона (expect.yaml), а не то, что код решил
        # при последнем прогоне: иначе таблица проверяла бы код кодом
        sea = ((expect.get("world") or {}).get("sea") or {}).get("present")
        out.append({
            "name": name,
            "lat": float(site["lat"]),
            "lon": float(site["lon"]),
            "grid": facts["grid"]["kind"],
            "north": facts["noon"].get("days_noon_north"),
            "south": facts["noon"].get("days_noon_south"),
            "sea": sea,
            "objects": facts["water_count"],
            "names": facts["water_named"],
            "tidal": facts["tidal"],
        })
    return out


def as_markdown() -> str:
    lines = [HEADER, DIVIDER]
    for r in rows():
        sea = r["sea"]
        sea_text = ("нет" if sea is False else "не знаю" if sea is None else "есть")
        lines.append(
            f"| `{r['name']}` | {r['lat']:.2f}, {r['lon']:.2f} | {r['grid']} | "
            f"{r['north']} / {r['south']} | {sea_text} | "
            f"{r['objects']} / {r['names']} / {r['tidal']} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff", action="store_true",
                        help="сверить строки с docs/calc/world_coverage.md и назвать расхождения")
    args = parser.parse_args()
    markdown = as_markdown()
    print(markdown)
    if not args.diff:
        return 0

    doc = (ROOT / "docs" / "calc" / "world_coverage.md").read_text(encoding="utf-8")
    missing = []
    for line in markdown.splitlines()[2:]:
        name = line.split("`")[1]
        doc_line = next((l for l in doc.splitlines() if l.startswith(f"| `{name}`")), None)
        if doc_line is None:
            missing.append(f"{name}: строки нет в документе")
            continue
        counts = line.split("|")[6].strip()
        if counts not in doc_line:
            missing.append(f"{name}: в документе не {counts!r}: {doc_line.strip()[:90]}")
    for item in missing:
        print(item, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
