"""Фикстуры моря для эталонов: ответ Overpass с признаками моря (S2).

Скрипт нужен редко — когда меняется правило определения моря или добавляется эталон.
Он делает один запрос на город и кладёт **сжатый** ответ в `tests/golden/<город>/fixtures/sea.json.gz`,
чтобы проверка моря в `test_golden_world.py` шла без сети.

Запуск:
    python scripts/make_sea_fixtures.py suva sydney singapore rio-de-janeiro hong-kong buenos-aires
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geosite import sea, water  # noqa: E402

GOLDEN = ROOT / "tests" / "golden"


def save(name: str, *, radius_m: int = 8000) -> Path | None:
    site_file = GOLDEN / name / "site.yaml"
    if not site_file.exists():
        print(f"{name}: нет site.yaml")
        return None
    site = {}
    for line in site_file.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.strip().startswith("#"):
            key, value = line.split(":", 1)
            site[key.strip()] = value.strip()
    lat, lon = float(site["lat"]), float(site["lon"])
    payload = sea.fetch_payload(lat, lon, radius_m)
    if payload is None:
        print(f"{name}: сеть недоступна, фикстура не обновлена")
        return None
    # берег 15 км нужен правилу «море есть, но берег дальше кольца» (Сингапур: 8.97 км)
    wide = water.fetch_coast(lat, lon, 15000)
    if wide is not None:
        out_wide = GOLDEN / name / "fixtures" / "coast-wide.json.gz"
        with gzip.open(out_wide, "wt", encoding="utf-8") as fh:
            json.dump(wide, fh, ensure_ascii=False)
    out = GOLDEN / name / "fixtures" / "sea.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    polygons = sea.payload_polygons(payload, lat, lon)
    print(f"{name}: элементов {len(payload.get('elements') or [])}, площадей моря {len(polygons)} → {out}")
    return out


def main(argv: list[str]) -> int:
    names = argv or ["suva", "sydney", "singapore", "rio-de-janeiro", "hong-kong",
                     "buenos-aires", "cairo", "beijing"]
    for name in names:
        save(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
