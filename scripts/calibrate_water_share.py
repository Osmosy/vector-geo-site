"""Калибровка доли воды: OSM против ESA WorldCover, по кольцам и клиньям (W2, W3).

Одна команда воспроизводит таблицу калибровки:

    python scripts/calibrate_water_share.py                 # все эталоны с фикстурами
    python scripts/calibrate_water_share.py london-city     # один эталон
    python scripts/calibrate_water_share.py --write         # обновить docs/calc/water_share_calibration.md

Считается по фикстурам, без сети: ответ Overpass и вырезка WorldCover лежат в эталоне.
Расхождение — не приговор коду: у него должна быть причина, и колонка «причина» подсказывает,
какая именно (осевая река без поверхности, мост, устаревший OSM, тень на снимке).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geosite import calibrate, geo, landcover, water, water_pack  # noqa: E402
from geosite.cli import _load_osm_file  # noqa: E402

GOLDEN = ROOT / "tests" / "golden"
RINGS = ((500.0, 2000.0), (2000.0, 8000.0))
RING_TOLERANCE_PP = calibrate.RING_TOLERANCE * 100.0
SECTOR_TOLERANCE_PP = calibrate.SECTOR_TOLERANCE * 100.0


def site_calibration(name: str) -> dict | None:
    site_dir = GOLDEN / name
    fixture = site_dir / "fixtures"
    site = yaml.safe_load((site_dir / "site.yaml").read_text(encoding="utf-8"))
    lat, lon = float(site["lat"]), float(site["lon"])
    # G2: та же сетка, что и в пакете, иначе калибровка меряет не то, что пакет
    epsg = geo.grid_crs(lat, lon, 50.0)["crs"]
    site_xy = water.to_utm([(lon, lat)], epsg)[0]

    osm_files = [p for p in (fixture / "osm-water.json.gz", fixture / "osm-water.json") if p.exists()]
    if not osm_files:
        return None
    payload = _load_osm_file(str(osm_files[0]))
    features = water.parse_features(payload, lat=lat, lon=lon, radius_m=8000.0)
    cover_path = fixture / "worldcover.tif"
    cover = (landcover.measure(lat, lon, crs=epsg, site_xy=site_xy, path=cover_path)
             if cover_path.exists() else None)

    tidal = any((f.tags.get("tidal") or "").lower() == "yes" for f in features)
    rows = []
    for r0, r1 in RINGS:
        osm_ring = water_pack.sector_water_share(features, crs=epsg, site_xy=site_xy,
                                                 ring_m=(r0, r1))
        osm_total = osm_ring["area_km2"] / max(osm_ring["ring_km2"], 1e-9)
        key = f"{r0 / 1000:g}-{r1 / 1000:g}"
        if cover is None:
            rows.append({"ring": key, "sector": "кольцо", "osm": osm_total, "wc": None,
                         "chosen": None, "diff_pp": None, "confidence": None,
                         "reason": "нет вырезки WorldCover"})
            continue
        wc_ring = cover["rings"].get(key) or {}
        wc_total = wc_ring.get("share")
        ring_diff = 0.0 if wc_total is None else osm_total - wc_total
        rows.append({"ring": key, "sector": "кольцо", "osm": osm_total, "wc": wc_total,
                     "chosen": None, "diff_pp": None if wc_total is None else ring_diff * 100.0,
                     "confidence": None if wc_total is None else
                     ("high" if abs(ring_diff) <= calibrate.RING_TOLERANCE else "low"),
                     "reason": "—" if wc_total is None else
                     ("в допуске" if abs(ring_diff) <= calibrate.RING_TOLERANCE
                      else "кольцо вне допуска: разбирать по клиньям")})
        for mountain in geo.MOUNTAINS:
            osm_share = osm_ring["by_mountain"].get(mountain, 0.0)
            wc_share = (wc_ring.get("by_mountain") or {}).get(mountain, 0.0)
            diff = (osm_share - wc_share) * 100.0
            if abs(diff) < 0.5:
                continue                      # клинья без воды и без расхождения не печатаем
            reason = calibrate.reason_for(
                osm_share, wc_share, linear_km=(osm_ring.get("linear_km") or {}).get(mountain),
                tidal=tidal, built_share=(cover["groups"] or {}).get("built"))
            rows.append({"ring": key, "sector": mountain, "osm": osm_share, "wc": wc_share,
                         "chosen": calibrate.choose(osm_share, wc_share, reason=reason),
                         "diff_pp": diff,
                         "confidence": calibrate.confidence_for(osm_share, wc_share,
                                                                ring_diff=ring_diff, reason=reason),
                         "reason": calibrate.explain(reason) or "—"})
    return {"name": name, "rows": rows, "has_cover": cover is not None,
            "tile": None if cover is None else cover["tile"]}


def render(results: list[dict]) -> str:
    lines = ["# Калибровка доли воды: OSM против WorldCover", "",
             "Считается `python scripts/calibrate_water_share.py --write` по фикстурам эталонов,",
             "без сети. Доли — по кольцам 0.5–2 и 2–8 км и по клиньям (24 горы).",
             "",
             f"Допуски W3: кольцо целиком ≤ {RING_TOLERANCE_PP:g} п.п., клин ≤ {SECTOR_TOLERANCE_PP:g} п.п.",
             "Расхождение сверх допуска — не ошибка сама по себе: у него должна быть причина.",
             "",
             "Правило выбора: " + calibrate.RULE + ".", ""]
    for result in results:
        lines.append(f"## {result['name']}")
        lines.append("")
        lines.append(f"Слой: ESA WorldCover 2021 v200, тайл {result['tile'] or '—'}."
                     if result["has_cover"] else "Слой: WorldCover не подключён для этого эталона.")
        lines.append("")
        lines.append("| Кольцо | Клин | OSM, % | WorldCover, % | Выбрано, % | Разница, п.п. "
                     "| Уверенность | Причина |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in result["rows"]:
            osm = "—" if row["osm"] is None else f"{row['osm'] * 100:.2f}"
            wc = "—" if row["wc"] is None else f"{row['wc'] * 100:.2f}"
            chosen = "—" if row.get("chosen") is None else f"{row['chosen'] * 100:.2f}"
            diff = "—" if row["diff_pp"] is None else f"{row['diff_pp']:+.2f}"
            conf = row.get("confidence") or "—"
            lines.append(f"| {row['ring']} | {row['sector']} | {osm} | {wc} | {chosen} | {diff} "
                         f"| {conf} | {row['reason']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sites", nargs="*", help="имена эталонов (по умолчанию все)")
    parser.add_argument("--write", action="store_true",
                        help="записать docs/calc/water_share_calibration.md")
    args = parser.parse_args(argv)

    names = args.sites or sorted(p.name for p in GOLDEN.iterdir() if (p / "site.yaml").exists())
    results = [r for r in (site_calibration(name) for name in names) if r]
    if not results:
        print("нет эталонов с фикстурами OSM", file=sys.stderr)
        return 1
    text = render(results)
    print(text)
    if args.write:
        target = ROOT / "docs" / "calc" / "water_share_calibration.md"
        target.write_text(text, encoding="utf-8")
        print(f"записано {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
