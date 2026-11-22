"""Сборка пакета участка: site_pack.json + схемы."""

from __future__ import annotations

import json
from pathlib import Path

from . import figures, geo
from .models import SCHEMA_ID, SitePack, json_schema

ATTRIBUTION = [
    "© OpenStreetMap contributors",
    "produced using Copernicus WorldDEM-30 © DLR e.V.",
]


def build_pack(*, query: dict, site: dict, orientation: dict, chart_year: dict,
               measured: dict, warnings: list[str], notes: list[str],
               dem_resolution_m: float, out_dir: Path,
               site_label: str = "") -> SitePack:
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sectors = []
    for s in measured["sectors"]:
        rings = []
        for r in s["rings"]:
            rings.append({
                "r_km": tuple(r["r_km"]),
                "max_rel_elev_m": r["max_rel_elev_m"],
                "mean_rel_elev_m": r["mean_rel_elev_m"],
                "horizon_deg": r["horizon_deg"],
                "hill_share": r.get("hill_share"),
                "peaks": r.get("peaks", []),
            })
        sectors.append({"mountain": s["mountain"], "from_deg": geo.mountain_bounds(s["mountain"])[0],
                        "to_deg": geo.mountain_bounds(s["mountain"])[1], "rings": rings})

    # Схемы: роза, профиль горизонта, отмывка макро-охвата
    label = site_label or query.get("input", "")
    figures.rose24_svg(measured["sectors"], {"place": label, "facing_true_deg": orientation["facing_true_deg"]},
                       figures_dir / "rose24.svg")
    figures.horizon_svg(measured["sectors"], figures_dir / "horizon.svg")
    grid = measured["grid_macro"]
    col, row = ~grid.transform * (measured["x0"], measured["y0"])
    figures.hillshade_png(grid, figures_dir / "hillshade_macro.png", site_px=(int(col), int(row)))

    pack = SitePack(
        query=query,
        site=site,
        orientation=orientation,
        chart_year=chart_year,
        sectors=sectors,
        point=measured.get("point"),
        four_spirits=measured.get("four_spirits"),
        dragon_vein=measured.get("dragon_vein"),
        water=measured.get("water", []),
        roads=measured.get("roads", []),
        exposure=measured.get("exposure"),
        land_cover=measured.get("land_cover"),
        quality={"dem": "Copernicus DEM GLO-30 (DSM)", "dem_resolution_m": dem_resolution_m,
                 "warnings": warnings, "notes": notes},
        attribution=ATTRIBUTION,
    )
    (out_dir / "site_pack.json").write_text(
        json.dumps(pack.model_dump(by_alias=True, exclude_none=False), ensure_ascii=False, indent=2),
        encoding="utf-8")
    schema_dir = out_dir / "schema"
    schema_dir.mkdir(exist_ok=True)
    (schema_dir / "site_pack-1.json").write_text(
        json.dumps(json_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    return pack


def load_pack(path: Path) -> SitePack:
    return SitePack.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def schema_text() -> str:
    return json.dumps(json_schema(), ensure_ascii=False, indent=2)


__all__ = ["build_pack", "load_pack", "schema_text", "SCHEMA_ID", "ATTRIBUTION"]
