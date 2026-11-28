"""Сборка пакета участка: site_pack.json + схемы."""

from __future__ import annotations

import json
from pathlib import Path

import hashlib

from . import calibrate, climate, dem, dtm, figures, geo, geocode, gsw, hydro, landcover, sea, sun
from . import terrain, water, worldnames
from . import validate as validate_mod
from .models import SCHEMA_ID, SitePack, json_schema

ALGO_VERSION = "2.14.0"      # версия сборки пакета

# P0: единственный источник версий — константы модулей. Ручные строки здесь разъезжались с
# кодом: ALGO_VERSION стоял 1.6.0, а в ALGO было "pack": "1.9.0".
ALGO = {
    "water": water.ALGO_VERSION,
    "terrain": terrain.ALGO_VERSION,
    "geocode": geocode.ALGO_VERSION,
    "geo": geo.ALGO_VERSION,
    "climate": climate.ALGO_VERSION,
    "sun": sun.ALGO_VERSION,
    "landcover": landcover.ALGO_VERSION,
    "calibrate": calibrate.ALGO_VERSION,
    "dtm": dtm.ALGO_VERSION,
    "worldnames": worldnames.ALGO_VERSION,
    # W4/E-043: модули, добавленные позже, в ALGO не попадали — их данные в пакете были,
    # а версии проверить было нельзя (I-11 сверяет только записанное)
    "sea": sea.ALGO_VERSION,
    "gsw": gsw.ALGO_VERSION,
    "hydro": hydro.ALGO_VERSION,
    "pack": ALGO_VERSION,
}

ATTRIBUTION = [
    "© OpenStreetMap contributors",
    "produced using Copernicus WorldDEM-30 © DLR e.V.",
]


RIVER_KEEP_M = 3000.0       # W7: реки до 3 км попадают в пакет
OTHER_KEEP_M = 1000.0       # водоёмы и скрытая вода — до 1 км
WATERBODY_KEEP_M = 2000.0   # док и бассейн: именованная вода у самого участка (E-018)


# `present` — ответ моря (W2/E-041): null значит «не смотрели или не знаю», и выбрасывать его
# из файла значит выдать отсутствие ключа за «моря нет» (П1)
KEEP_NULL = {"closure_index", "closure_reason", "tidal", "dem_reason", "flow_dem_dir",
             "water_share", "outflow_visible", "confidence", "source", "exit", "present"}


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
    from .water import is_watercourse

    kept = []
    for item in objects:
        nearest = item.nearest_m or 1e9
        # Порог реки — для русла. Водоём живёт по общему порогу в 1 км, но док и бассейн —
        # это именованная вода, которую читает толкователь, поэтому им оставлен порог реки:
        # иначе City Road Basin в 1,8 км исчезал из пакета вместе с речными полями (E-018).
        if is_watercourse(item) and nearest <= RIVER_KEEP_M:
            kept.append(item)
        elif item.kind in {"dock", "sea"} and nearest <= WATERBODY_KEEP_M:
            kept.append(item)
        elif nearest <= OTHER_KEEP_M:
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


# Речные поля, которых у контура водоёма нет по смыслу: бассейн не втекает и не вытекает,
# у него нет устья, изгиба и петли. Не null, а именно отсутствуют — null означал бы «считали,
# но не вышло» (E-018).
AREA_DROP = (
    # W1/E-040: прилив — свойство воды, а не русла. Береговая линия в километре делает приливной
    # любую воду, включая лагуну и залив, и море (S2) считает приливные объекты своим признаком.
    # Пока `tidal` выбрасывался у контуров, Ла-Плата в Буэнос-Айресе молчала о приливе.
    "in_mountain", "out_mountain", "bend", "bend_kind", "curvature_radius_m", "straight_at_site",
    "loop", "site_position", "flow_osm_dir", "flow_dem_dir", "flow_chosen", "flow_rule",
    "flow_confidence", "flow_agree", "flow_ways_share", "dem_slope_m_per_km", "dem_slope_snr",
    "dem_reason", "water_mouth", "outflow_visible",
    "crossings_2km", "crossing_angle_deg", "between_branches", "order", "discharge_m3s",
    "hydro_distance_m", "pieces", "confluences",
)


def _strip_area_fields(props: dict) -> dict:
    """Убрать речные поля у площадного объекта, оставив только данные водоёма (E-018)."""
    if props.get("geom") != "area":
        return props
    return {key: value for key, value in props.items() if key not in AREA_DROP}


def write_water_geojson(path: Path, objects: list, geometry: dict) -> None:
    """Полный набор воды с геометрией: то, что не влезло в пакет, живёт здесь."""
    features = []
    for item in objects:
        coords = geometry.get(item.id)
        if not coords:
            continue
        props = _strip_area_fields(item.model_dump(mode="json"))
        # dem_layer — свойство пакета, а не геометрии: в geojson он был бы null у всех объектов
        props.pop("dem_layer", None)
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
                "void_share": r.get("void_share"),
                **_near_ring_layers(measured, s["mountain"], len(rings)),
            })
        wind_share = (measured.get("wind") or {}).get("by_mountain", {}).get(s["mountain"])
        sectors.append({"mountain": s["mountain"], "from_deg": geo.mountain_bounds(s["mountain"])[0],
                        "to_deg": geo.mountain_bounds(s["mountain"])[1], "rings": rings,
                        "water": (measured.get("water_names") or {}).get(s["mountain"], []),
                        "water_km": _km(s["mountain"]),
                        "wind_share": wind_share,
                        **_calibration(measured, s["mountain"])})

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

    # D2/П7: слой рельефа подписывается у каждого поля, которое от него зависит
    dtm_info = measured.get("dtm_info") or {}
    layer = "dtm" if dtm_info.get("source") else dem.SOURCE_INFO.get(
        dem_source, {}).get("kind", "dsm_smoothed")
    built = (measured.get("land_cover") or {}).get("built")
    dem_confidence = dem.relief_confidence(layer, built)
    # G4.4: больше 5 % пустот в ближнем кольце — числам кольца верить нельзя, что бы ни говорил
    # паспорт слоя: пустоты не интерполируются, и «плоско» там означает «не измерено»
    voids = [r.get("void_share") for s_ in measured.get("sectors") or []
             for r in s_["rings"] if (r.get("r_km") or [0, 0])[1] <= 2.0 and r.get("void_share")]
    worst_void = max(voids) if voids else 0.0
    if worst_void > 0.05:
        dem_confidence = "low"
        measured.setdefault("void_warning", round(worst_void, 3))

    def _with_layer(block):
        """Поле по рельефу несёт `dem_layer`: иначе не видно, по земле оно или по крышам."""
        if block is None:
            return None
        data = dict(block)
        data.setdefault("dem_layer", layer)
        return data

    def _tag_water(items):
        tagged = []
        for item in items:
            update = {"dem_layer": layer}
            if item.water_mouth is not None:
                update["water_mouth"] = item.water_mouth.model_copy(update={"dem_layer": layer})
            tagged.append(item.model_copy(update=update))
        return tagged

    water_all = list(measured.get("water", []))
    geometry = measured.get("water_geometry") or {}
    kept = trim_water(water_all)
    kept = _tag_water(kept)
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
    if "cold_wind" in exposure and exposure["cold_wind"] is None:
        exposure.pop("cold_wind")
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
    if exposure:
        exposure.setdefault("dem_layer", layer)

    pack = SitePack(
        query=query,
        site=site,
        orientation=orientation,
        chart_year=chart_year,
        sectors=sectors,
        point=_with_layer(measured.get("point")),
        four_spirits=_with_layer(measured.get("four_spirits")),
        sun=measured.get("sun"),
        dragon_vein=_with_layer(measured.get("dragon_vein")),
        water=kept,
        roads=measured.get("roads", []),
        exposure=exposure or None,
        land_cover=measured.get("land_cover"),
        water_gsw=measured.get("water_gsw"),
        sea=measured.get("sea"),
        water_share_rule=calibrate.RULE if measured.get("water_calibration") else None,
        quality={"dem": dem_label, "dem_source": dem_source, "dem_kind": layer,
                 "dem_resolution_m": dem_resolution_m, "dem_confidence": dem_confidence,
                 "partial_overpass": True if measured.get("water_partial") else None,
                 "partial_osm": True if measured.get("partial_osm") else None,
                 "warnings": warnings, "notes": notes},
        summary=summary,
        files=_files_block(geojson_name, sha, measured.get("land_cover_layer"),
                           _dem_block(measured, dem_source, dem_resolution_m),
                           dtm.file_block(measured.get("dtm_info") or {})),
        attribution=_attribution(measured.get("land_cover_layer")),
    )
    payload = pack.model_dump(by_alias=True, exclude_none=False)
    payload["water"] = [_strip_area_fields(item) for item in payload.get("water") or []]
    payload = _prune(payload)
    payload["algo"] = dict(ALGO)
    violations = validate_mod.validate(payload)
    validate_mod.apply_to_pack(payload, violations)
    (out_dir / "site_pack.json").write_text(_dump_pack(payload), encoding="utf-8")
    schema_dir = out_dir / "schema"
    schema_dir.mkdir(exist_ok=True)
    # имя файла схемы следует за SCHEMA_ID: site_pack/2 → site_pack-2.json
    (schema_dir / (SCHEMA_ID.replace("/", "-") + ".json")).write_text(
        json.dumps(json_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    return pack


def load_pack(path: Path) -> SitePack:
    return SitePack.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


WORLDCOVER_ATTRIBUTION = ("ESA WorldCover 2021 / Contains modified Copernicus Sentinel data "
                          "(2021) processed by ESA WorldCover consortium (CC BY 4.0)")


def _attribution(cover: dict | None) -> list[str]:
    """Строки атрибуции: WorldCover добавляется, только если слой действительно считан."""
    return ATTRIBUTION if not cover else list(ATTRIBUTION) + [WORLDCOVER_ATTRIBUTION]


def _near_ring_layers(measured: dict, mountain: str, index: int) -> dict:
    """D3: в ближнем кольце оба вида рельефа — поверхность и земля рядом, а не вместо."""
    if index != 0:
        return {}
    dsm = (measured.get("dsm_near_ring") or {}).get(mountain)
    dtm_info = measured.get("dtm_info") or {}
    if not dsm or not dtm_info.get("source"):
        return {}
    # кольца секторов строятся по макро-сетке. Если земля есть только в ближнем кольце
    # (`scope: near_ring`), то кольцо посчитано по DSM — второй слой был бы тем же числом
    # под другим именем. Такой пары не публикуем: пусть лучше не будет, чем двойник.
    if (dtm_info.get("scope") or "all") != "all":
        return {}
    dtm_ring = next((s["rings"][0] for s in measured.get("sectors") or []
                     if s["mountain"] == mountain), None)
    if dtm_ring is None:
        return {}
    keys = ("max_rel_elev_m", "mean_rel_elev_m", "horizon_deg")
    return {"dsm": {k: dsm.get(k) for k in keys},
            "dtm": {k: dtm_ring.get(k) for k in keys}}


def _calibration(measured: dict, mountain: str) -> dict:
    """W4: оба значения доли воды, выбранное и цена выбора — на уровне клина."""
    data = (measured.get("water_calibration") or {}).get(mountain) or {}
    return {key: data.get(key) for key in
            ("water_share", "water_share_osm", "water_share_wc",
             "water_share_confidence", "water_share_reason")}


def _dem_block(measured: dict, dem_source: str, dem_resolution_m: float) -> dict:
    """`files.dem` описывает поверхность, даже когда числа в пакете сняты с земли (DTM)."""
    meta = measured.get("dem_dsm") or {}
    return dem.file_block(meta.get("source") or dem_source,
                          kind=meta.get("kind") or "dsm_smoothed",
                          resolution_m=meta.get("resolution_m") or dem_resolution_m)


def _files_block(geojson_name: str, sha: str, cover: dict | None,
                 dem_block: dict | None = None, dtm_block: dict | None = None) -> dict:
    """Откуда взяты данные: у недоступного источника это видно в `files`, а не в догадках."""
    block: dict = {"water_geojson": geojson_name, "sha256": sha}
    if dem_block:
        block["dem"] = dem_block
    if dtm_block:
        block["dtm"] = dtm_block
    if cover:
        block["land_cover"] = {"source": landcover.SOURCE, "license": landcover.LICENSE,
                               "resolution_m": cover.get("resolution_m"),
                               "tile": cover.get("tile"), "cached": cover.get("cached")}
    return block


def schema_text() -> str:
    return json.dumps(json_schema(), ensure_ascii=False, indent=2)


__all__ = ["build_pack", "load_pack", "schema_text", "SCHEMA_ID", "ATTRIBUTION"]
