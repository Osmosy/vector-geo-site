"""CLI `geosite`: место → данные → пакет участка."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from pathlib import Path

from . import calibrate, climate, dem, dtm, geo, hydro, landcover, sun, terrain, water
from . import water_pack
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


def _measure(args, r, *, with_water: bool = True, osm_payload: dict | None = None) -> dict:
    epsg = geo.utm_epsg(r.lat, r.lon)
    macro_km = 50.0 if r.precision in {"district", "settlement"} else 20.0
    facing = args.facing
    # в городе микрорельеф — это крыши: для уклона и формы берём сглаживание 300 м
    point_smooth = 300.0 if r.precision in {"district", "settlement"} else 90.0

    def _pass(macro_grid=None, meso_grid=None) -> dict:
        return terrain.measure_site(r.lat, r.lon, epsg=epsg, resolution_m=30.0,
                                    macro_km=macro_km, meso_km=8.0, facing_deg=facing,
                                    point_smooth_m=point_smooth,
                                    macro_grid=macro_grid, meso_grid=meso_grid)

    # D3: сначала поверхность (DSM) — она нужна всегда, в том числе для кольца 0–0.5 км,
    # где здания и есть городские 砂. Затем земля (DTM): по ней считаются все поля рельефа.
    dsm = _pass()
    out = dict(dsm)
    out["dem_dsm"] = {"source": dsm.get("dem_source"), "kind": dsm.get("dem_kind", "dsm_smoothed"),
                      "resolution_m": dsm.get("dem_resolution_m"),
                      "vertical_datum": dsm.get("dem_vertical_datum")}
    dtm_grid, dtm_info = (None, {"reason": "выключено флагом --no-dtm"})
    if not getattr(args, "no_dtm", False):
        macro_dtm, macro_info = dtm.grid(r.lat, r.lon, epsg=epsg, radius_km=macro_km,
                                         resolution_m=max(90.0, 30.0))
        if macro_dtm is not None:
            meso_dtm, _ = dtm.grid(r.lat, r.lon, epsg=epsg, radius_km=8.0, resolution_m=30.0)
            dtm_grid = macro_dtm
            dtm_info = macro_info
            dtm_out = _pass(macro_grid=macro_dtm, meso_grid=meso_dtm or macro_dtm)
            # ближнее кольцо остаётся в двух видах: поверхность (здания) и земля
            dtm_out["dsm_near_ring"] = {s["mountain"]: s["rings"][0] for s in dsm["sectors"]}
            dtm_out["dtm_info"] = dtm_info
            dtm_out["dem_dsm"] = out["dem_dsm"]
            out = dtm_out
            if dtm_info.get("vertical_datum") and dtm_info["vertical_datum"] != "EGM2008":
                out.setdefault("dtm_notes", []).append(
                    "DTM в системе %s (сдвиг к EGM2008: %s)"
                    % (dtm_info["vertical_datum"],
                       "не применён" if dtm_info.get("datum_shift_m") is None
                       else f"{dtm_info['datum_shift_m']:+.2f} м"))
        else:
            out["dtm_info"] = macro_info
    else:
        out["dtm_info"] = dtm_info

    if with_water:
        out.update(_measure_water(r, out, epsg, payload=osm_payload, args=args))
    return out


def _xy(point: tuple[float, float], epsg: int) -> np.ndarray:
    """lon/lat → UTM: одна точка, без батча."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(point[0], point[1])
    return np.array([x, y])


def _load_osm_file(path: str | None) -> dict | None:
    """Читать сохранённый ответ Overpass: JSON, в том числе сжатый gzip.

    Широкая выгрузка на 30 км весит десятки мегабайт и лежит в эталонах как `.json.gz`;
    распаковывать её руками перед запуском незачем — формат виден по магии файла, а не по
    расширению (переименованный `.gz` иначе читался бы как текст и падал на 0x8b).
    """
    if not path:
        return None
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        import gzip

        raw = gzip.decompress(raw)
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise SystemExit(f"{path}: не текст и не gzip ({exc}); ожидался JSON ответа Overpass") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: битый JSON ({exc})") from exc
    if "elements" not in data:
        raise SystemExit(f"{path}: в файле нет elements — это не ответ Overpass")
    return data


def _measure_water(r, measured: dict, epsg: int, payload: dict | None = None, args=None) -> dict:
    """Вода: OSM + метрики. Сеть нужна один раз — дальше кеш.

    `payload` — готовый ответ Overpass (файл): тогда сети не нужно вовсе, и пакет
    пересобирается офлайн на фикстуре эталона (`geosite pack --osm-file <fixture>`).
    """
    site_xy = np.array([measured["x0"], measured["y0"]])
    grid = measured["grid_macro"]
    result: dict = {"water": [], "roads": []}
    if payload is None:
        try:
            payload = water.fetch_raw(r.lat, r.lon, 8000)
        except Exception as exc:
            result["water_error"] = str(exc)
            return result
    wide_payload = _load_osm_file(getattr(args, "osm_file_wide", None))
    metrics = water.measure_water(r.lat, r.lon, epsg=epsg, grid=grid, site_xy=site_xy,
                                  radius_m=8000, payload=payload, wide_payload=wide_payload,
                                  allow_network=payload is None and not getattr(args, "offline", False))
    # Порядок реки и расход — HydroRIVERS: без них «ручей» и «Волга» в пакете неотличимы.
    # Источник необязателен: без shapefile пакет собирается, поле остаётся None с предупреждением.
    try:
        reach = hydro.site_reach(r.lat, r.lon)
        result["hydro"] = reach
        result.setdefault("hydro_warnings", []).extend(hydro.assign(metrics, reach))
    except Exception as exc:  # нет pyshp, нет архива, битый индекс
        result["hydro"] = None
        result.setdefault("hydro_warnings", []).append(f"hydro: источник недоступен ({exc})")
    result["water"] = water_pack.to_objects(metrics)
    # геометрия для water.geojson: берём её у самих метрик, а не сопоставляем два разных списка
    # по порядку — они фильтруются независимо (одна река = одна запись), и zip давал чужую линию
    result["water_geometry"] = {item.id: item.coords for item in metrics if item.coords}
    # Доля воды по секторам: по полигонам, аналитически в UTM, без растяжения линий.
    # Линии русла площадь не дают — река, нарисованная осевой, в долю не попадает, и это
    # оговорено в предупреждении; в городах реки обычно нарисованы полигонами.
    try:
        features = water.parse_features(payload, lat=r.lat, lon=r.lon, radius_m=8000.0)
        inner = water_pack.sector_water_share(features, epsg=epsg, site_xy=site_xy,
                                              ring_m=(500.0, 2000.0))
        middle = water_pack.sector_water_share(features, epsg=epsg, site_xy=site_xy,
                                               ring_m=(2000.0, 8000.0))
        result["water_share"] = {"inner": inner, "middle": middle}
        # вода по секторам: у какой горы ближайшая точка воды — имя туда. Только именованная
        # вода и только четыре ближайшие в секторе: иначе в сектор доков попадают 32 фонтана.
        pick: dict[str, list[tuple[float, str]]] = {}
        for metric in metrics:
            if not metric.coords or not metric.name:
                continue
            near = min(metric.coords, key=lambda c: float(np.hypot(
                *(_xy(c, epsg) - site_xy))))
            x, y = _xy(near, epsg) - site_xy
            dist = float(np.hypot(x, y))
            if dist > 2000.0:
                continue
            pick.setdefault(geo.mountain_name(math.degrees(math.atan2(x, y)) % 360.0),
                            []).append((dist, metric.name))
        names: dict[str, list[str]] = {}
        for mountain, items in pick.items():
            seen: list[str] = []
            for _, name in sorted(items):
                if name not in seen:
                    seen.append(name)
            names[mountain] = seen[:4]
        result["water_names"] = names
        osm_water = round(inner["area_km2"] / max(inner["ring_km2"], 1e-9), 4)
        result["land_cover"] = {"water": osm_water}
    except Exception as exc:
        result["water_share_skipped"] = True
        result["water_share_error"] = str(exc)
    # W2: независимая сверка по ESA WorldCover 10 м. Считается всегда, когда есть слой:
    # это не украшение, а второй источник по воде и первый — по застройке, деревьям и открытому.
    try:
        cover = landcover.measure(r.lat, r.lon, epsg=epsg, site_xy=site_xy,
                                  path=getattr(args, "worldcover_file", None),
                                  offline=bool(getattr(args, "offline", False)))
    except Exception as exc:                      # слой не должен ронять пакет
        cover = None
        result.setdefault("water_warnings", []).append(f"worldcover: слой не прочитан ({exc})")
    if cover is None:
        result.setdefault("water_warnings", []).append(
            "worldcover: слоя покрова нет — land_cover кроме воды пуст, "
            "расхождение OSM и снимков не проверено")
    else:
        share = result.get("land_cover") or {}
        share.update({"built": cover["groups"]["built"], "tree": cover["groups"]["tree"],
                      "open": cover["groups"]["open"], "water_wc": cover["groups"]["water"],
                      "source": cover["source"], "ring_km": cover["groups_ring_km"]})
        result["land_cover"] = share
        result["water_share_wc"] = cover["rings"]
        result["land_cover_layer"] = {"tile": cover["tile"], "cached": cover["cached"],
                                      "resolution_m": cover["resolution_m"],
                                      "license": cover["license"]}

        # W3/W4: калибровка по клиньям кольца 0.5–2 км — оба значения, выбор и причина
        osm_inner = (result.get("water_share") or {}).get("inner") or {}
        wc_inner = (cover["rings"].get("0.5-2") or {})
        osm_total = share.get("water") or 0.0
        wc_total = share.get("water_wc") or 0.0
        ring_diff = float(osm_total) - float(wc_total)
        # Приливность — свойство реки, а не клина: русло может лежать в одном секторе, а его
        # поверхность и осушка — в трёх. Берём кольцо целиком, вместе с берегами приливных рек.
        water_objs = result.get("water") or []
        tidal_rivers = {getattr(w, "id", None) for w in water_objs
                        if getattr(w, "tidal", None) is True}
        tidal_in_ring = bool(tidal_rivers) or any(
            getattr(w, "riverbank_of", None) in tidal_rivers for w in water_objs)
        linear = osm_inner.get("linear_km") or {}
        calibration = {}
        for mountain in geo.MOUNTAINS:
            osm_share = float((osm_inner.get("by_mountain") or {}).get(mountain) or 0.0)
            wc_share = float((wc_inner.get("by_mountain") or {}).get(mountain) or 0.0)
            reason = calibrate.reason_for(osm_share, wc_share,
                                          linear_km=linear.get(mountain),
                                          tidal=tidal_in_ring,
                                          built_share=share.get("built"))
            calibration[mountain] = {
                "water_share": calibrate.choose(osm_share, wc_share, reason=reason),
                "water_share_osm": round(osm_share, 4),
                "water_share_wc": round(wc_share, 4),
                "water_share_confidence": calibrate.confidence_for(
                    osm_share, wc_share, ring_diff=ring_diff, reason=reason),
                "water_share_reason": reason,
            }
        result["water_calibration"] = calibration
        low = [m for m, c in calibration.items() if c["water_share_reason"]]
        if low:
            codes = sorted({calibration[m]["water_share_reason"] for m in low})
            result.setdefault("water_warnings", []).append(
                "water_share: расхождение OSM и WorldCover выше допуска в %d клин(ах) из 24 (%s); "
                "причина записана, уверенность low"
                % (len(low), ", ".join(calibrate.explain(c) for c in codes)))
        if any(c["water_share"] != c["water_share_osm"] for c in calibration.values()):
            result.setdefault("water_notes", []).append(
                "water_share: в части клиньев выбрано значение по снимкам — OSM воды там не видит "
                "(осевая линия); правило: " + calibrate.RULE)

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
    m = _measure(args, r, osm_payload=_load_osm_file(getattr(args, "osm_file", None)))
    facing = args.facing
    warnings = list(m.get("water_warnings") or [])
    notes = list(m.get("water_notes") or [])
    if facing is None:
        facing, why = _model_facing(m)
        orientation_source, confidence = "model", "low"
        notes.append(why)
        # ось получена по модели — четыре духа надо посчитать от неё же,
        # иначе в пакете не окажется сравнения «спина / фасад» (R3)
        if not m.get("four_spirits") and m.get("grid_macro") is not None:
            from .terrain import four_spirits
            m["four_spirits"] = four_spirits(m["grid_macro"], m["frame"], m["elev_m"],
                                             facing, meso_km=8.0)
    else:
        orientation_source, confidence = "user_compass", "high"
    if args.year is None:
        notes.append("год постройки не указан: карта считается для текущего года")
    if m.get("flow_conflict"):
        warnings.append("water.flow: low — конфликт OSM и рельефа у " + ", ".join(m["flow_conflict"]))
    low_flow = [w for w in m.get("water") or [] if getattr(w, "flow_confidence", None) == "low"]
    if low_flow and not m.get("flow_conflict"):
        # E-020: низкая уверенность бывает и без конфликта — короткий канал не выходит
        # из своей горы, и направление нечем проверить. Молчать об этом нельзя (I-10).
        reasons = sorted({getattr(w, "same_mountain_reason", None) for w in low_flow} - {None})
        warnings.append("water.flow: low — у %d водотоков направление не проверено (%s)"
                        % (len(low_flow), ", ".join(reasons) or "вход и выход в одной горе"))
    if m.get("water_share_skipped"):
        warnings.append("доля воды по секторам не считается: полигоны OSM не получены")
    if m.get("water_share"):
        warnings.append("water_share — по полигонам OSM (water=*, natural=water); река, нарисованная "
                        "осевой, в долю не входит, её длина — sector.water_km")
    if not m.get("water"):
        warnings.append("вода: слои OSM не получены, water и water_share пусты")
    if m.get("water_error"):
        warnings.append(f"вода: {m['water_error']}")
    dtm_info = m.get("dtm_info") or {}
    if dtm_info.get("source"):
        notes.append("рельеф: земля (DTM) — %s, %s м, система высот %s; расчёт полей по земле, "
                     "кольцо 0–0.5 км — в двух видах (dsm и dtm)"
                     % (dtm_info.get("name"), dtm_info.get("resolution_m"),
                        dtm_info.get("vertical_datum")))
        for line in m.get("dtm_notes") or []:
            notes.append(line)
    else:
        warnings.append("no_dtm: " + str(dtm_info.get("reason") or "DTM не подключён")
                        + " — все поля рельефа посчитаны по поверхности (DSM)")
    dem_source = m.get("dem_source") or "copernicus"
    dem_layer = dem.SOURCE_INFO.get(dem_source, {}).get("kind", "dsm_smoothed")
    built_share = (m.get("land_cover") or {}).get("built")
    dem_conf = dem.relief_confidence(dem_layer, built_share)
    warnings.append("рельеф — модель поверхности (DSM): уклоны и форма точки сглажены на 90 м, "
                    "в плотной застройке возможны искажения")
    if dem_conf != "high":
        warnings.append(f"рельеф: уверенность полей по рельефу {dem_conf} (П7): слой "
                        f"{dem_layer}, застройка в кольце "
                        f"{'—' if built_share is None else format(built_share, '.0%')}; "
                        "высоты земли в городе по DSM завышены на высоту зданий")
    if r.precision in {"district", "settlement"}:
        warnings.append("точка — историческое ядро/центр по геокодеру, не контур здания")

    # этап 4: склонение по WMM (офлайн) и ветер по архиву ERA5 (кеш/сеть)
    decl = None
    try:
        decl = climate.declination(r.lat, r.lon)
    except Exception as exc:
        warnings.append(f"склонение не рассчитано: {exc}")
    wind = None
    try:
        wind = climate.summary(r.lat, r.lon, allow_network=not args.offline)
        m["wind"] = wind
    except Exception as exc:
        warnings.append(f"ветер не рассчитан: {exc}")
    if wind is not None:
        m["exposure"] = {**(m.get("exposure") or {}), "wind": wind,
                         "cold_wind": wind.get("cold_wind")}

    # H2: солнце. Горизонт берём по мезо-охвату (8 км) — тот же, что у сглаженного DSM.
    try:
        m["sun"] = sun.block(r.lat, r.lon, grid=m.get("grid_meso"), x0=m.get("x0"),
                             y0=m.get("y0"), z0=m.get("elev_m"))
    except Exception as exc:                       # солнце не должно ронять пакет
        warnings.append(f"солнце не рассчитано: {exc}")

    orientation = {
        "facing_true_deg": round(facing, 1),
        "declination_deg": None if decl is None else decl["deg"],
        "facing_mag_deg": None if decl is None else climate.magnetic_bearing(facing, decl["deg"]),
        "facing_mountain": geo.mountain_name(facing),
        "sitting_mountain": geo.mountain_name(facing + 180.0),
        "source": orientation_source,
        "confidence": confidence,
        "near_boundary": geo.near_boundary(facing),
    }
    site = {"lat": round(r.lat, 5), "lon": round(r.lon, 5), "elev_m": m["elev_m"],
            "precision": r.precision, "utm_epsg": geo.utm_epsg(r.lat, r.lon),
            "boundary": None,
            # H1: полушарие — факт; «около экватора» = солнце в зените в пределах 1°
            "hemisphere": "S" if r.lat < 0 else "N",
            "equatorial": abs(r.lat) <= 1.0}
    out_dir = Path(args.out or f"site-{abs(r.lat):.3f}-{abs(r.lon):.3f}")
    dem_source = m.get("dem_source") or "copernicus"
    dem_label = m.get("dem_label") or "Copernicus DEM GLO-30 (DSM)"
    if dem_source != "copernicus":
        # Не молчим: SRTM — не модель поверхности, он не видит крыши и кроны, и «горы» на нём ниже.
        warnings.append(f"рельеф — запасной источник: {dem_label}; числа по рельефу не сравнимы "
                        f"с пакетами на Copernicus")
    pack = build_pack(query={"input": r.input, "kind": r.kind, "resolved_by": r.resolved_by,
                             "alternatives": r.alternatives[:3]},
                      site=site, orientation=orientation, chart_year=_chart_year(args.year),
                      measured=m, warnings=warnings, notes=notes,
                      dem_resolution_m=30.0, dem_source=dem_source, dem_label=dem_label,
                      out_dir=out_dir, site_label=r.display_name[:60])
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
    w = _measure_water(r, m, epsg, payload=_load_osm_file(getattr(args, "osm_file", None)), args=args)
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
        p.add_argument("--osm-file", default=None,
                       help="готовый ответ Overpass (JSON) вместо загрузки: офлайн-пересборка")
        p.add_argument("--no-dtm", action="store_true",
                       help="не подключать DTM: считать всё по поверхности (DSM)")
        p.add_argument("--worldcover-file", default=None,
                       help="сохранённая вырезка ESA WorldCover (GeoTIFF) вместо сети")
        p.add_argument("--osm-file-wide", default=None,
                       help="ответ Overpass радиусом 30 км (приливность по руслу); подменяет загрузку")

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
    common(p_water)
    p_water.set_defaults(func=cmd_water)

    p_val = sub.add_parser("validate", help="проверить site_pack.json на инварианты")
    p_val.add_argument("pack", help="путь к site_pack.json")
    p_val.add_argument("--baseline", default=None, help="файл с известными нарушениями")
    p_val.set_defaults(func=cmd_validate)

    p_schema = sub.add_parser("schema", help="JSON Schema пакета")
    p_schema.set_defaults(func=cmd_schema)

    args = ap.parse_args(argv)
    # Флаг офлайна ставим на время команды и снимаем после: иначе один вызов `main()` в процессе
    # (тесты, ноутбук) оставляет сеть выключенной для всех следующих вызовов.
    prev_offline = os.environ.get("GEOSITE_OFFLINE")
    if getattr(args, "offline", False):
        os.environ["GEOSITE_OFFLINE"] = "1"
    try:
        return args.func(args)
    finally:
        if prev_offline is None:
            os.environ.pop("GEOSITE_OFFLINE", None)
        else:
            os.environ["GEOSITE_OFFLINE"] = prev_offline


if __name__ == "__main__":
    raise SystemExit(main())
