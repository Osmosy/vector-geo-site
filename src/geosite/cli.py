"""CLI `geosite`: место → данные → пакет участка."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from pathlib import Path

from . import geo, terrain, water, water_pack
from .models import SitePack, json_schema

STAGES_DONE = {"resolve", "frame", "fetch", "terrain", "pack"}
STAGES_TODO = {"water": "этап 3: реки, изгибы, слияния, водные врата",
               "exposure": "этап 4: ветер, свет, склонение"}


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _resolve(text: str):
    from .geocode import Ambiguous, resolve
    try:
        return resolve(text)
    except Ambiguous as amb:
        print("Ввод неоднозначен, выберите вариант:", file=sys.stderr)
        for i, opt in enumerate(amb.options, 1):
            print(f"  {i}. {opt['name']}  ({opt['lat']:.4f}, {opt['lon']:.4f})", file=sys.stderr)
        raise SystemExit(2)


def _model_facing(measured: dict) -> tuple[float, str]:
    """Ось по модели: спина к главной возвышенности мезо-охвата, фасад — в противоположную сторону."""
    sectors = measured["sectors"]
    best = max(sectors, key=lambda s: s["rings"][2]["max_rel_elev_m"])
    idx = geo.MOUNTAINS.index(best["mountain"])
    behind = idx * 15.0
    return geo.normalize_deg(behind + 180.0), f"модель: спина к {best['mountain']} ({best['rings'][2]['max_rel_elev_m']} м)"


def _chart_year(year: int | None) -> dict:
    if year:
        return {"value": year, "source": "user"}
    return {"value": 2026, "source": "current_year"}


def cmd_resolve(args) -> int:
    r = _resolve(args.place)
    out = {"input": r.input, "lat": r.lat, "lon": r.lon, "kind": r.kind, "precision": r.precision,
           "display_name": r.display_name, "resolved_by": r.resolved_by,
           "utm_epsg": geo.utm_epsg(r.lat, r.lon),
           "alternatives": r.alternatives}
    _print(out)
    return 0


def cmd_frame(args) -> int:
    r = _resolve(args.place)
    epsg = geo.utm_epsg(r.lat, r.lon)
    macro_km = 50.0 if r.precision in {"district", "settlement"} else 20.0
    _print({"site": {"lat": r.lat, "lon": r.lon, "precision": r.precision},
            "utm_epsg": epsg,
            "extents": {"macro": {"radius_km": macro_km, "resolution_m": 90},
                        "meso": {"radius_km": 8.0, "resolution_m": 30},
                        "micro": {"radius_km": 0.5, "resolution_m": 10}}})
    return 0


def cmd_fetch(args) -> int:
    from .dem import bbox_for_radius, ensure_tile, tiles_for_bbox
    r = _resolve(args.place)
    macro_km = 50.0 if r.precision in {"district", "settlement"} else 20.0
    tiles = tiles_for_bbox(*bbox_for_radius(r.lat, r.lon, macro_km))
    done = []
    for la, lo in tiles:
        p = ensure_tile(la, lo)
        done.append({"tile": p.parent.name, "bytes": p.stat().st_size})
    _print({"place": r.display_name or r.input, "tiles": done})
    return 0


def _measure(args, r, *, with_water: bool = True) -> dict:
    epsg = geo.utm_epsg(r.lat, r.lon)
    macro_km = 50.0 if r.precision in {"district", "settlement"} else 20.0
    facing = args.facing
    # в городе микрорельеф — это крыши: для уклона и формы берём сглаживание 300 м
    point_smooth = 300.0 if r.precision in {"district", "settlement"} else 90.0
    out = terrain.measure_site(r.lat, r.lon, epsg=epsg, resolution_m=30.0, macro_km=macro_km,
                              meso_km=8.0, facing_deg=facing, point_smooth_m=point_smooth)
    if with_water:
        out.update(_measure_water(r, out, epsg))
    return out


def _measure_water(r, measured: dict, epsg: int) -> dict:
    """Вода: OSM + метрики. Сеть нужна один раз — дальше кеш."""
    site_xy = np.array([measured["x0"], measured["y0"]])
    grid = measured["grid_macro"]
    result: dict = {"water": [], "roads": []}
    try:
        payload = water.fetch_raw(r.lat, r.lon, 8000)
    except Exception as exc:
        result["water_error"] = str(exc)
        return result
    features = water.parse_features(payload)
    metrics = water.measure_water(r.lat, r.lon, epsg=epsg, grid=grid, site_xy=site_xy, radius_m=8000)
    result["water"] = water_pack.to_objects(metrics)
    # геометрия для water.geojson: тот же id, что в пакете (W7)
    result["water_geometry"] = {
        metrics_item.id: merged_item.coords
        for metrics_item, merged_item in zip(metrics, [m for m in water.merge_by_name(features) if m.coords])
    }
    # Долю воды по секторам пока не считаем: маска воды требует калибровки
    # (полигоны OSM чинятся buffer(0), линии расширяются — площадь систематически завышается).
    result["water_share_skipped"] = True
    # П6/П4: расхождения источников и низкая уверенность должны звучать в warnings
    conflicts = [m for m in metrics if m.flow_rule == 4]
    lows = [m for m in metrics if m.flow_confidence == "low"]
    if conflicts:
        result["flow_conflict"] = [m.name or m.id for m in conflicts][:5]
    if lows:
        result["flow_low"] = [m.name or m.id for m in lows][:5]
    result["water_count"] = len(metrics)
    return result


def cmd_terrain(args) -> int:
    r = _resolve(args.place)
    m = _measure(args, r)
    _print({"place": r.display_name or r.input, "elev_m": m["elev_m"], "point": m["point"],
            "four_spirits": m.get("four_spirits"),
            "dragon_vein": None if not m.get("dragon_vein") else {
                "length_km": m["dragon_vein"].length_km, "drop_m": m["dragon_vein"].drop_m,
                "bends": m["dragon_vein"].bends, "saddles": m["dragon_vein"].saddles},
            "sectors": [{"mountain": s["mountain"], "max_rel_elev_m": s["rings"][2]["max_rel_elev_m"],
                         "horizon_deg": s["rings"][2]["horizon_deg"]} for s in m["sectors"]]})
    return 0


def cmd_pack(args) -> int:
    from .pack import build_pack
    r = _resolve(args.place)
    m = _measure(args, r)
    facing = args.facing
    warnings, notes = [], []
    if facing is None:
        facing, why = _model_facing(m)
        orientation_source, confidence = "model", "low"
        notes.append(why)
    else:
        orientation_source, confidence = "user_compass", "high"
    if args.year is None:
        notes.append("год постройки не указан: карта считается для текущего года")
    if m.get("flow_conflict"):
        warnings.append("water.flow: low — конфликт OSM и рельефа у " + ", ".join(m["flow_conflict"]))
    if m.get("water_share_skipped"):
        warnings.append("доля воды по секторам не считается: маска воды не откалибрована по площади")
    if not m.get("water"):
        warnings.append("вода: слои OSM не получены, water и water_share пусты")
    if m.get("water_error"):
        warnings.append(f"вода: {m['water_error']}")
    warnings.append("склонение и ветер не рассчитаны (этап 4)")
    warnings.append("рельеф — модель поверхности (DSM): уклоны и форма точки сглажены на 90 м, "
                    "в плотной застройке возможны искажения")
    if r.precision in {"district", "settlement"}:
        warnings.append("точка — историческое ядро/центр по геокодеру, не контур здания")

    orientation = {
        "facing_true_deg": round(facing, 1),
        "declination_deg": None,
        "facing_mag_deg": None,
        "facing_mountain": geo.mountain_name(facing),
        "sitting_mountain": geo.mountain_name(facing + 180.0),
        "source": orientation_source,
        "confidence": confidence,
        "near_boundary": geo.near_boundary(facing),
    }
    site = {"lat": round(r.lat, 5), "lon": round(r.lon, 5), "elev_m": m["elev_m"],
            "precision": r.precision, "utm_epsg": geo.utm_epsg(r.lat, r.lon),
            "boundary": None}
    out_dir = Path(args.out or f"site-{abs(r.lat):.3f}-{abs(r.lon):.3f}")
    pack = build_pack(query={"input": r.input, "kind": r.kind, "resolved_by": r.resolved_by,
                             "alternatives": r.alternatives[:3]},
                      site=site, orientation=orientation, chart_year=_chart_year(args.year),
                      measured=m, warnings=warnings, notes=notes,
                      dem_resolution_m=30.0, out_dir=out_dir, site_label=r.display_name[:60])
    print(f"пакет: {out_dir / 'site_pack.json'} ({len(pack.sectors)} секторов, схемы в figures/)")
    return 0


def cmd_run(args) -> int:
    print("→ resolve", file=sys.stderr)
    r = _resolve(args.place)
    print(f"  {r.display_name or r.input} · {r.lat:.4f}, {r.lon:.4f} · {r.precision}", file=sys.stderr)
    print("→ terrain (DEM: Copernicus GLO-30)", file=sys.stderr)
    args.place = args.place
    rc = cmd_pack(args)
    print(f"→ interpret: следующий шаг — навык vector-fengshui-bazi "
          f"(landform 24 горы + feixing карта звёзд), этап 3/4 геомодуля не готовы", file=sys.stderr)
    return rc


def cmd_water(args) -> int:
    r = _resolve(args.place)
    epsg = geo.utm_epsg(r.lat, r.lon)
    macro_km = 50.0 if r.precision in {"district", "settlement"} else 20.0
    point_smooth = 300.0 if r.precision in {"district", "settlement"} else 90.0
    m = terrain.measure_site(r.lat, r.lon, epsg=epsg, resolution_m=30.0, macro_km=macro_km,
                            meso_km=8.0, facing_deg=None, point_smooth_m=point_smooth)
    w = _measure_water(r, m, epsg)
    rows = [{"id": o.id, "name": o.name, "kind": o.kind, "nearest_m": o.nearest_m,
             "in": o.in_mountain, "out": o.out_mountain, "bend": o.bend,
             "curvature_km": None if o.curvature_radius_m is None else round(o.curvature_radius_m / 1000, 2),
             "straight": o.straight_at_site, "flow_agree": o.flow_agree,
             "mouth": None if not o.water_mouth else {"mountain": o.water_mouth.mountain,
                                                      "closure": o.water_mouth.closure_index,
                                                      "width_m": o.water_mouth.width_m},
             "visible_out": o.outflow_visible, "confluences": len(o.confluences)}
            for o in w.get("water", [])[:20]]
    _print({"place": r.display_name or r.input, "features": w.get("water_count", 0),
            "error": w.get("water_error"), "nearest": rows})
    return 0


def cmd_validate(args) -> int:
    from .validate import validate_file

    violations = validate_file(args.pack)
    if args.baseline:
        # известные нарушения: новый код|путь — падение, исчезнувшие — сообщаем отдельно
        import json as _json
        known = set(_json.loads(Path(args.baseline).read_text(encoding="utf-8")))
        current = {f"{v['code']}|{v['path']}" for v in violations}
        fresh = sorted(current - known)
        gone = sorted(known - current)
        _print({"valid_not_worse": not fresh, "fresh": fresh, "fixed": gone,
                "count": len(violations)})
        return 1 if fresh else 0
    if violations:
        _print({"valid": False, "count": len(violations), "violations": violations})
        return 1
    _print({"valid": True, "count": 0, "violations": []})
    return 0


def cmd_schema(args) -> int:
    _print(json_schema())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="geosite", description="Геоданные местности для фэншуй-интерпретации")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("place", help="координаты, адрес или населённый пункт")
        p.add_argument("--facing", type=float, default=None, help="азимут фасада, истинный, градусы")
        p.add_argument("--year", type=int, default=None, help="год постройки / заселения")
        p.add_argument("--out", default=None, help="каталог пакета")
        p.add_argument("--offline", action="store_true", help="не ходить в сеть, брать из кеша")

    p_res = sub.add_parser("resolve", help="ввод → точка, тип, точность")
    p_res.add_argument("place")
    p_res.set_defaults(func=cmd_resolve)

    p_frame = sub.add_parser("frame", help="три охвата и зона UTM")
    p_frame.add_argument("place")
    p_frame.set_defaults(func=cmd_frame)

    p_fetch = sub.add_parser("fetch", help="скачать тайлы DEM в кеш")
    p_fetch.add_argument("place")
    p_fetch.set_defaults(func=cmd_fetch)

    p_ter = sub.add_parser("terrain", help="рельеф: 24 горы, четыре духа, жила")
    common(p_ter)
    p_ter.set_defaults(func=cmd_terrain)

    p_pack = sub.add_parser("pack", help="собрать site_pack.json и схемы")
    common(p_pack)
    p_pack.set_defaults(func=cmd_pack)

    p_run = sub.add_parser("run", help="весь путь: место → пакет")
    common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_water = sub.add_parser("water", help="вода: реки, изгибы, слияния, врата")
    p_water.add_argument("place")
    p_water.set_defaults(func=cmd_water)

    p_val = sub.add_parser("validate", help="проверить site_pack.json на инварианты")
    p_val.add_argument("pack", help="путь к site_pack.json")
    p_val.add_argument("--baseline", default=None, help="файл с известными нарушениями")
    p_val.set_defaults(func=cmd_validate)

    p_schema = sub.add_parser("schema", help="JSON Schema пакета")
    p_schema.set_defaults(func=cmd_schema)

    args = ap.parse_args(argv)
    if getattr(args, "offline", False):
        os.environ["GEOSITE_OFFLINE"] = "1"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
