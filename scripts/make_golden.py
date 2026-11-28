"""Сборка эталонных фикстур для tests/golden: OSM-ответы и вырезанный DEM.

Запускается вручную при добавлении эталона; в CI ничего не скачивается.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
CACHE = Path.home() / ".cache" / "vector-geo-site"

# Мировые эталоны (G1). Каждый покрывает то, что не воспроизводится в Европе: полушарие,
# экватор, антимеридиан, граница зон UTM, полярная широта, море в кольце, прилив по берегу,
# письменность имени. `wide` — только там, где граница прилива дальше 8 км от участка.
WORLD_SITES = {
    "sydney": {"lat": -33.8688, "lon": 151.2093, "radius_km": 8.0, "facing": None, "wide": True},
    "rio-de-janeiro": {"lat": -22.9068, "lon": -43.1729, "radius_km": 8.0, "facing": None},
    "suva": {"lat": -18.1416, "lon": 178.4419, "radius_km": 8.0, "facing": None},
    "singapore": {"lat": 1.3521, "lon": 103.8198, "radius_km": 8.0, "facing": None},
    "cairo": {"lat": 30.0444, "lon": 31.2357, "radius_km": 8.0, "facing": None},
    "hong-kong": {"lat": 22.2819, "lon": 114.1582, "radius_km": 8.0, "facing": None,
                  "macro_km": 50.0},
    "reykjavik": {"lat": 64.1466, "lon": -21.9426, "radius_km": 8.0, "facing": None},
    "beijing": {"lat": 39.9042, "lon": 116.4074, "radius_km": 8.0, "facing": None},
    "istanbul": {"lat": 41.0082, "lon": 28.9784, "radius_km": 8.0, "facing": None},
    "buenos-aires": {"lat": -34.6037, "lon": -58.3816, "radius_km": 8.0, "facing": None},
    "dubai": {"lat": 25.2048, "lon": 55.2708, "radius_km": 8.0, "facing": None},
    "taveuni": {"lat": -16.79, "lon": -179.97, "radius_km": 8.0, "facing": None},
    "arctic-ocean": {"lat": 84.30, "lon": 15.00, "radius_km": 8.0, "facing": None},
    # S4: Амстердам обязателен — высоты ниже уровня моря (I-09), AHN как DTM, NAP → EGM2008;
    # классические места — на них проверяется смысл интерпретации, а не только геоданные
    "amsterdam": {"lat": 52.3731, "lon": 4.8922, "radius_km": 8.0, "facing": None},
    # Ханчжоу у границы зон UTM (120° в. д.): круг 20 км задевает её, и пакет считает в aeqd.
    # macro_km обязан стоять здесь, иначе DEM вырезается в UTM и центр участка попадает в пустоту
    "hangzhou-west-lake": {"lat": 30.2428, "lon": 120.1517, "radius_km": 8.0, "facing": 270.0,
                           "macro_km": 20.0},
    "kyoto-imperial-palace": {"lat": 35.0254, "lon": 135.7622, "radius_km": 8.0, "facing": 90.0},
    "chongqing-jiefangbei": {"lat": 29.5576, "lon": 106.5771, "radius_km": 8.0, "facing": None},
}

SITES = {
    "london-city": {"lat": 51.5138, "lon": -0.0984, "radius_km": 20.0, "facing": 180.0},
    "london-isle-of-dogs": {"lat": 51.4885, "lon": -0.0195, "radius_km": 20.0, "facing": None},
    "kingston-upon-thames": {"lat": 51.4100, "lon": -0.3000, "radius_km": 20.0, "facing": None},
    "moscow-kremlin": {"lat": 55.7520, "lon": 37.6175, "radius_km": 20.0, "facing": None},
    "nizhny-novgorod-strelka": {"lat": 56.3300, "lon": 44.0100, "radius_km": 20.0, "facing": None},
    "sankt-goar-rhine": {"lat": 50.1500, "lon": 7.7100, "radius_km": 20.0, "facing": None},
}
SITES.update(WORLD_SITES)


def dem_crop(lat: float, lon: float, radius_km: float, out: Path, resolution_m: float = 90.0,
             macro_km: float | None = None, crs: str | None = None) -> None:
    """Вырезать квадрат DEM из кеша тайлов Copernicus и сохранить плотным GeoTIFF.

    Сетка вырезки — та же, что у пакета (`geo.grid_crs`): у антимеридиана и границы зон это
    азимутальная, а не UTM. Иначе `measure_site` читает файл в чужой проекции и центр участка
    попадает в пустоту — высота точки приходит `None` там, где DEM на самом деле есть (E-034).
    """
    from pyproj import Transformer

    from geosite import geo
    from geosite.dem import bbox_parts, ensure_tile, tiles_for_bbox

    # сетка вырезки: явно заданная, иначе сетка пакета (macro_km), иначе UTM — так европейские
    # эталоны остаются в той же проекции, в которой их проверяет test_golden.py
    if crs is not None:
        epsg = crs
    elif macro_km is not None:
        epsg = geo.grid_crs(lat, lon, float(macro_km))["crs"]
    else:
        epsg = geo.utm_epsg(lat, lon)
    # у линии перемены дат один bbox выходит за 180°: берём части, как делает пакет (E-032)
    tiles = []
    seen = set()
    for lat_min, lat_max, lon_min, lon_max in bbox_parts(lat, lon, radius_km):
        for la, lo in tiles_for_bbox(lat_min, lat_max, lon_min, lon_max):
            if (la, lo) in seen:
                continue
            seen.add((la, lo))
            try:
                tiles.append(ensure_tile(la, lo))
            except Exception as exc:
                print(f"тайл {la},{lo}: {exc}")

    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(epsg), always_xy=True)
    x0, y0 = tr.transform(lon, lat)
    size = int(2 * radius_km * 1000 / resolution_m) + 1
    transform = rasterio.transform.from_origin(x0 - size * resolution_m / 2,
                                               y0 + size * resolution_m / 2,
                                               resolution_m, resolution_m)

    mosaic = np.full((size, size), np.nan, dtype="float32")
    from rasterio.warp import Resampling, reproject

    for tile in tiles:
        with rasterio.open(tile) as src:
            band = src.read(1).astype("float32")
            part = np.full_like(mosaic, np.nan)
            reproject(band, part, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=geo.crs_str(epsg),
                      resampling=Resampling.bilinear, dst_nodata=np.nan)
        # reproject заливает весь массив nodata'ой за пределами своего тайла: писать напрямую
        # в мозаику значит стирать уже собранное (из 4 тайлов выживал последний)
        mosaic = np.where(np.isnan(part), mosaic, part)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs=geo.crs_str(epsg), transform=transform,
                       compress="deflate", nodata=np.nan) as dst:
        dst.write(mosaic, 1)


def osm_payload(lat: float, lon: float, radius_m: int) -> dict:
    from geosite import water

    return water.fetch_raw(lat, lon, radius_m)


def main(names: list[str] | None = None, *, osm_only: bool = False) -> int:
    for name in names or list(SITES):
        site = SITES[name]
        target = GOLDEN / name
        (target / "fixtures").mkdir(parents=True, exist_ok=True)
        payload = osm_payload(site["lat"], site["lon"], 8000)
        # gzip: `out body geom` приносит геометрию членов отношений, и файл втрое меньше
        import gzip
        with gzip.open(target / "fixtures" / "osm-water.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        # берег — такой же вход, как вода (E-028): без него офлайн-пакет не знает про море.
        # W2/E-041: входов у правила моря три, и все три сохраняются, даже пустыми: пустой ответ
        # «берега нет» — это измерение, отсутствие файла — «не смотрели».
        from geosite import sea as _sea
        from geosite import water as _w
        for radius_m, fname in ((8000, "coast.json.gz"), (15000, "coast-wide.json.gz")):
            try:
                coast = _w.fetch_coast(site["lat"], site["lon"], radius_m, allow_network=True)
            except Exception as exc:
                coast = None
                print(f"{name}: берег {radius_m} м не получен ({exc})")
            if coast is None:
                print(f"{name}: берег {radius_m} м не запрошен")
                continue
            with gzip.open(target / "fixtures" / fname, "wt", encoding="utf-8") as fh:
                json.dump(coast, fh)
            if not coast.get("elements"):
                print(f"{name}: береговой линии в {radius_m / 1000:g} км нет")
        try:
            tokens = _sea.fetch_payload(site["lat"], site["lon"], 8000, allow_network=True)
        except Exception as exc:
            tokens = None
            print(f"{name}: признаки моря не получены ({exc})")
        if tokens is not None:
            with gzip.open(target / "fixtures" / "sea.json.gz", "wt", encoding="utf-8") as fh:
                json.dump(tokens, fh)
        # широкая выгрузка нужна только там, где граница прилива дальше 8 км (Темза, Парраматта);
        # для остальных это 2 МБ в репозитории без пользы
        if site.get("wide", name not in WORLD_SITES):
            try:
                wide = osm_payload(site["lat"], site["lon"], 30000)
                import gzip
                with gzip.open(target / "fixtures" / "osm-water-wide.json.gz", "wt",
                               encoding="utf-8") as fh:
                    json.dump(wide, fh)
            except Exception as exc:  # широкий запрос не обязателен для проверок без приливности
                print(f"{name}: широкая выгрузка не получена ({exc})")
        if not osm_only:
            try:
                dem_crop(site["lat"], site["lon"], site["radius_km"],
                         target / "fixtures" / "dem.tif", macro_km=site.get("macro_km"))
            except Exception as exc:      # вне покрытия Copernicus (84° с. ш.) рельефа не будет
                print(f"{name}: DEM не получен ({exc})")
            from geosite import landcover
            try:
                if not landcover.write_crop(site["lat"], site["lon"], 8000.0,
                                            target / "fixtures" / "worldcover.tif"):
                    print(f"{name}: слой покрова не получен")
            except Exception as exc:
                print(f"{name}: слой покрова не получен ({exc})")
        facing = "null" if site["facing"] is None else f"{site['facing']}"
        macro = f"macro_km: {site['macro_km']}\n" if site.get("macro_km") else ""
        (target / "site.yaml").write_text(
            f"name: {name}\nlat: {site['lat']}\nlon: {site['lon']}\n"
            f"radius_km: {site['radius_km']}\nfacing: {facing}\n{macro}",
            encoding="utf-8")
        print(f"{name}: {target}")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--osm-only"]
    raise SystemExit(main(args or None, osm_only="--osm-only" in sys.argv))
