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

SITES = {
    "london-city": {"lat": 51.5138, "lon": -0.0984, "radius_km": 20.0, "facing": 180.0},
    "london-isle-of-dogs": {"lat": 51.4885, "lon": -0.0195, "radius_km": 20.0, "facing": None},
    "kingston-upon-thames": {"lat": 51.4100, "lon": -0.3000, "radius_km": 20.0, "facing": None},
    "moscow-kremlin": {"lat": 55.7520, "lon": 37.6175, "radius_km": 20.0, "facing": None},
    "nizhny-novgorod-strelka": {"lat": 56.3300, "lon": 44.0100, "radius_km": 20.0, "facing": None},
    "sankt-goar-rhine": {"lat": 50.1500, "lon": 7.7100, "radius_km": 20.0, "facing": None},
}


def dem_crop(lat: float, lon: float, radius_km: float, out: Path, resolution_m: float = 90.0) -> None:
    """Вырезать квадрат DEM из кеша тайлов Copernicus в UTM и сохранить плотным GeoTIFF."""
    from pyproj import Transformer

    from geosite import geo
    from geosite.dem import bbox_for_radius, ensure_tile, tiles_for_bbox

    epsg = geo.utm_epsg(lat, lon)
    lat_min, lat_max, lon_min, lon_max = bbox_for_radius(lat, lon, radius_km)
    tiles = []
    for la, lo in tiles_for_bbox(lat_min, lat_max, lon_min, lon_max):
        tiles.append(ensure_tile(la, lo))

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
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
            reproject(band, mosaic, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=f"EPSG:{epsg}",
                      resampling=Resampling.bilinear, dst_nodata=np.nan)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs=f"EPSG:{epsg}", transform=transform,
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
        try:
            wide = osm_payload(site["lat"], site["lon"], 30000)
            import gzip
            with gzip.open(target / "fixtures" / "osm-water-wide.json.gz", "wt", encoding="utf-8") as fh:
                json.dump(wide, fh)
        except Exception as exc:  # широкий запрос не обязателен для проверок без приливности
            print(f"{name}: широкая выгрузка не получена ({exc})")
        if not osm_only:
            dem_crop(site["lat"], site["lon"], site["radius_km"], target / "fixtures" / "dem.tif")
            from geosite import landcover
            try:
                if not landcover.write_crop(site["lat"], site["lon"], 8000.0,
                                            target / "fixtures" / "worldcover.tif"):
                    print(f"{name}: слой покрова не получен")
            except Exception as exc:
                print(f"{name}: слой покрова не получен ({exc})")
        facing = "null" if site["facing"] is None else f"{site['facing']}"
        (target / "site.yaml").write_text(
            f"name: {name}\nlat: {site['lat']}\nlon: {site['lon']}\n"
            f"radius_km: {site['radius_km']}\nfacing: {facing}\n",
            encoding="utf-8")
        print(f"{name}: {target}")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--osm-only"]
    raise SystemExit(main(args or None, osm_only="--osm-only" in sys.argv))
