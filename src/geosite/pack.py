"""Сборка пакета участка: site_pack.json + схемы."""

from __future__ import annotations

import json
from pathlib import Path

import hashlib

from . import figures, geo
from . import validate as validate_mod
from .models import SCHEMA_ID, SitePack, json_schema

ALGO = {"water": "6.4.1", "terrain": "2.1.0", "geocode": "1.0.0", "pack": "1.3.1",
        "climate": "1.0.0"}

ALGO_VERSION = "1.1.0"

ATTRIBUTION = [
    "© OpenStreetMap contributors",
    "produced using Copernicus WorldDEM-30 © DLR e.V.",
]


RIVER_KEEP_M = 3000.0       # W7: реки до 3 км попадают в пакет
OTHER_KEEP_M = 1000.0       # водоёмы и скрытая вода — до 1 км


KEEP_NULL = {"closure_index", "closure_reason", "tidal", "dem_reason", "flow_dem_dir",
             "water_share", "outflow_visible", "confidence", "source", "exit"}


def _dump_pack(payload: dict) -> str:
    """Пакет в JSON: вода — по объекту на строку, остальное с отступом 1.

    Так диффы остаются осмысленными (изменённый объект — изменённая строка), а пакет
    не распухает на полторы тысячи строк из-за вложенности.
    """
    water = payload.get("water") or []
    payload = dict(payload)
    payload["water"] = "__WATER_PLACEHOLDER__"
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if water:
        rows = ",\n".join("   " + json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                          for item in water)
        text = text.replace('"__WATER_PLACEHOLDER__"', f"[\n{rows}\n  ]")
    else:
        text = text.replace('"__WATER_PLACEHOLDER__"', "[]")
    return text


def _prune(value):
    """Выбросить пустые поля, оставив явные null там, где null — это ответ (П1)."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if item is None and key not in KEEP_NULL:
                continue
            if isinstance(item, (list, dict)) and not item:
                continue
            out[key] = _prune(item)
        return out
    if isinstance(value, list):
        return [_prune(item) for item in value]
    return value


def trim_water(objects: list) -> list:
    """Оставить в пакете то, что читает интерпретатор: ближние реки, вода и все слияния."""
    kept = []
    for item in objects:
        nearest = item.nearest_m or 1e9
        if item.kind in {"river", "stream"} and nearest <= RIVER_KEEP_M:
            kept.append(item)
        elif item.kind not in {"river", "stream"} and nearest <= OTHER_KEEP_M:
            kept.append(item)
        elif item.confluences:
            kept.append(item)
    return kept


def water_summary(all_objects: list, kept: list) -> dict:
    """Сводка по типам и сторонам света: интерпретатор читает её, а не 800 объектов."""
    by_kind: dict[str, int] = {}
    by_direction: dict[str, int] = {}
    for item in all_objects:
        by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
        if item.in_mountain:
            by_direction[item.in_mountain] = by_direction.get(item.in_mountain, 0) + 1
    eight = {}
    for mountain, count in by_direction.items():
        eight[mountain] = eight.get(mountain, 0) + count
    return {"total": len(all_objects), "kept": len(kept), "by_kind": by_kind,
            "by_in_mountain": by_direction}


def write_water_geojson(path: Path, objects: list, geometry: dict) -> None:
    """Полный набор воды с геометрией: то, что не влезло в пакет, живёт здесь."""
    features = []
    for item in objects:
        coords = geometry.get(item.id)
        if not coords:
            continue
        props = item.model_dump(mode="json")
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in coords]},
            "properties": props,
        })
    payload = {"type": "FeatureCollection", "name": "water",
               "license": "ODbL © OpenStreetMap contributors", "features": features}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_pack(*, query: dict, site: dict, orientation: dict, chart_year: dict,
               measured: dict, warnings: list[str], notes: list[str],
               dem_resolution_m: float, out_dir: Path,
               dem_source: str = "copernicus", dem_label: str = "Copernicus DEM GLO-30 (DSM)",
               site_label: str = "") -> SitePack:
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    shares = measured.get("water_share") or {}

    def _share(ring: list, block: str, mountain: str) -> float | None:
        data = shares.get(block) or {}
        values = data.get("by_mountain") or {}
        return values.get(mountain)

    def _km(mountain: str) -> float | None:
        data = (shares.get("inner") or {}).get("linear_km") or {}
        value = data.get(mountain)
        return value if value else None

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
                "water_share": r.get("water_share") if r.get("water_share") is not None else
                (_share(r, "inner", s["mountain"]) if tuple(r.get("r_km") or ()) == (0.5, 2.0)
                 else _share(r, "middle", s["mountain"])
                 if tuple(r.get("r_km") or ()) == (2.0, 8.0) else None),
                "peaks": r.get("peaks", []),
            })
        wind_share = (measured.get("wind") or {}).get("by_mountain", {}).get(s["mountain"])
        sectors.append({"mountain": s["mountain"], "from_deg": geo.mountain_bounds(s["mountain"])[0],
                        "to_deg": geo.mountain_bounds(s["mountain"])[1], "rings": rings,
                        "water": (measured.get("water_names") or {}).get(s["mountain"], []),
                        "water_km": _km(s["mountain"]),
                        "wind_share": wind_share})

    # Схемы: роза, профиль горизонта, отмывка макро-охвата
    label = site_label or query.get("input", "")
    figures.rose24_svg(measured["sectors"], {"place": label, "facing_true_deg": orientation["facing_true_deg"]},
                       figures_dir / "rose24.svg")
    figures.horizon_svg(measured["sectors"], figures_dir / "horizon.svg")
    grid = measured["grid_macro"]
    col, row = ~grid.transform * (measured["x0"], measured["y0"])
    figures.hillshade_png(grid, figures_dir / "hillshade_macro.png", site_px=(int(col), int(row)))

    if measured.get("hydro") is not None:
        hydro_site = {k: v for k, v in measured["hydro"].items() if not k.startswith("_")}
        site = {**site, "hydro": hydro_site}
    warnings = list(warnings) + list(measured.get("hydro_warnings") or [])

    water_all = list(measured.get("water", []))
    geometry = measured.get("water_geometry") or {}
    kept = trim_water(water_all)
    summary = water_summary(water_all, kept)
    geojson_name = None
    sha = {}
    if water_all and geometry:
        geojson_name = "water.geojson"
        path = out_dir / geojson_name
        write_water_geojson(path, water_all, geometry)
        sha[geojson_name] = hashlib.sha256(path.read_bytes()).hexdigest()

    exposure = dict(measured.get("exposure") or {})
    wind = exposure.pop("wind", None)
    if wind:
        exposure.update({
            "wind_prevailing_deg": wind.get("prevailing_from_deg"),
            "wind_prevailing_mountain": wind.get("prevailing_from_mountain"),
            "wind_speed_ms": wind.get("mean_speed_ms"),
            "wind_calm_share": wind.get("calm_share"),
            "wind_days": wind.get("days"),
            "wind_seasonal": wind.get("seasonal"),
            "wind_source": wind.get("source"),
        })

    pack = SitePack(
        query=query,
        site=site,
        orientation=orientation,
        chart_year=chart_year,
        sectors=sectors,
        point=measured.get("point"),
        four_spirits=measured.get("four_spirits"),
        dragon_vein=measured.get("dragon_vein"),
        water=kept,
        roads=measured.get("roads", []),
        exposure=exposure or None,
        land_cover=measured.get("land_cover"),
        quality={"dem": dem_label, "dem_source": dem_source, "dem_resolution_m": dem_resolution_m,
                 "warnings": warnings, "notes": notes},
        summary=summary,
        files={"water_geojson": geojson_name, "sha256": sha},
        attribution=ATTRIBUTION,
    )
    payload = pack.model_dump(by_alias=True, exclude_none=False)
    payload = _prune(payload)
    payload["algo"] = dict(ALGO)
    violations = validate_mod.validate(payload)
    validate_mod.apply_to_pack(payload, violations)
    (out_dir / "site_pack.json").write_text(_dump_pack(payload), encoding="utf-8")
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
