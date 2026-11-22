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


def main(names: list[str] | None = None) -> int:
    for name in names or list(SITES):
        site = SITES[name]
        target = GOLDEN / name
        (target / "fixtures").mkdir(parents=True, exist_ok=True)
        payload = osm_payload(site["lat"], site["lon"], 8000)
        (target / "fixtures" / "osm-water.json").write_text(
            json.dumps(payload), encoding="utf-8")
        dem_crop(site["lat"], site["lon"], site["radius_km"], target / "fixtures" / "dem.tif")
        facing = "null" if site["facing"] is None else f"{site['facing']}"
        (target / "site.yaml").write_text(
            f"name: {name}\nlat: {site['lat']}\nlon: {site['lon']}\n"
            f"radius_km: {site['radius_km']}\nfacing: {facing}\n",
            encoding="utf-8")
        print(f"{name}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
