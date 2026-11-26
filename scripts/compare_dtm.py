"""Сверка двух слоёв рельефа в 1000 случайных точках (D4).

    python scripts/compare_dtm.py 52.3731 4.8922            # DTM (лидар) против DSM (Copernicus)
    python scripts/compare_dtm.py 52.3731 4.8922 --radius 8 --points 1000 --write

Считается отдельно для застройки и открытых мест (класс WorldCover 50 — застройка):
ME (средняя разница), RMSE, 95-й процентиль модуля разницы. Знак разницы: DSM минус DTM,
то есть положительное значение означает, что поверхность выше земли — дом, крона, мост.

Таблица воспроизводится одной командой; `--write` обновляет раздел в docs/calc/dtm.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geosite import dem, dtm, geo, landcover  # noqa: E402

BUILT_CLASS = 50


def _worldcover_classes(lat: float, lon: float, *, radius_m: float) -> dict | None:
    """Классы покрова в окне: нужны, чтобы разделить точки на застройку и открытое."""
    window = landcover.window(lat, lon, radius_m)
    if window is None:
        return None
    return window


def _class_at(window: dict, lat: float, lon: float) -> int | None:
    a, _, c, _, e, f = window["transform"]
    array = window["array"]
    col = int((lon - c) / a)
    row = int((lat - f) / e)
    if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
        return int(array[row, col])
    return None


def compare(lat: float, lon: float, *, radius_km: float, points: int, seed: int = 11) -> dict:
    epsg = geo.utm_epsg(lat, lon)
    ridge, info = dtm.grid(lat, lon, crs=epsg, radius_km=radius_km, resolution_m=30.0)
    if ridge is None:
        return {"error": "DTM недоступен: " + str(info.get("reason"))}
    surface = dem.load_dem(lat, lon, radius_km, crs=epsg, resolution_m=30.0)

    rng = np.random.default_rng(seed)
    r_max = radius_km * 1000.0 * 0.95
    radii = r_max * np.sqrt(rng.random(points * 2))
    angles = rng.random(points * 2) * 2 * np.pi
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x0, y0 = tr.transform(lon, lat)
    cover = _worldcover_classes(lat, lon, radius_m=radius_km * 1000.0)

    diffs, kinds = [], []
    for r, a in zip(radii, angles):
        if len(diffs) >= points:
            break
        x, y = x0 + r * np.sin(a), y0 + r * np.cos(a)
        z_dtm = ridge.point_value(x, y)
        z_dsm = surface.point_value(x, y)
        if not (np.isfinite(z_dtm) and np.isfinite(z_dsm)):
            continue
        kind = None
        if cover is not None:
            lon_p, lat_p = tr.transform(x, y, direction="INVERSE")
            cls = _class_at(cover, lat_p, lon_p)
            kind = "built" if cls == BUILT_CLASS else "open"
        diffs.append(float(z_dsm - z_dtm))
        kinds.append(kind)

    diffs = np.array(diffs)
    if diffs.size == 0:
        return {"error": "нет точек с данными в обоих слоях"}
    out = {"points": int(diffs.size), "source_dtm": info.get("source"),
           "resolution_dtm": info.get("resolution_m"), "vertical_datum": info.get("vertical_datum"),
           "all": {**_stats(diffs), "points": int(diffs.size)}}
    for name in ("built", "open"):
        sel = np.array([k == name for k in kinds])
        if sel.any():
            out[name] = _stats(diffs[sel])
            out[name]["points"] = int(sel.sum())
    return out


def _stats(values: np.ndarray) -> dict:
    absolute = np.abs(values)
    return {"me": round(float(values.mean()), 2),
            "rmse": round(float(np.sqrt((values ** 2).mean())), 2),
            "p95": round(float(np.percentile(absolute, 95)), 2)}


def render(result: dict, *, place: str, radius_km: float) -> str:
    if result.get("error"):
        return f"## {place}\n\nСверка не выполнена: {result['error']}\n"
    lines = [f"## {place}", "",
             f"DTM: {result['source_dtm']}, {result['resolution_dtm']} м, "
             f"система высот {result['vertical_datum']}; DSM: Copernicus GLO-30. "
             f"Точек: {result['points']}, радиус {radius_km:g} км.", "",
             "| Выборка | Точек | ME, м | RMSE, м | p95, м |",
             "|---|---|---|---|---|"]
    for key, label in (("all", "все"), ("built", "застройка"), ("open", "открытое")):
        block = result.get(key)
        if not block:
            continue
        lines.append(f"| {label} | {block['points']} | {block['me']:+.2f} | {block['rmse']:.2f} "
                     f"| {block['p95']:.2f} |")
    lines.append("")
    lines.append("Разница — DSM минус DTM: положительная означает, что поверхность выше земли "
                 "(дом, крона, мост), отрицательная — что грубее оказалась земля.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lat", type=float)
    parser.add_argument("lon", type=float)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--place", default="")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    result = compare(args.lat, args.lon, radius_km=args.radius, points=args.points)
    text = render(result, place=args.place or f"{args.lat:.4f}, {args.lon:.4f}", radius_km=args.radius)
    print(text)
    if args.write:
        target = ROOT / "docs" / "calc" / "dtm_comparison.md"
        target.write_text(text, encoding="utf-8")
        print(f"записано {target.relative_to(ROOT)}")
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
