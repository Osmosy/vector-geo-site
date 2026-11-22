"""Рельеф: тайлы Copernicus DEM GLO-30, мозаика, перевод в локальную UTM."""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

CACHE_DIR = Path(os.environ.get("GEOSITE_CACHE", Path.home() / ".cache" / "vector-geo-site"))
DEM_CACHE = CACHE_DIR / "dem30"
S3 = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_LABEL = "Copernicus DEM GLO-30 (DSM)"
ATTRIBUTION = "produced using Copernicus WorldDEM-30 © DLR e.V."
USER_AGENT = "vector-geo-site/0.1"


@dataclass
class DemGrid:
    data: np.ndarray          # высоты, м, NaN где нет данных
    transform: object         # affine.Affine, UTM
    crs: str
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def cell_area_m2(self) -> float:
        return abs(self.transform.a * self.transform.e)

    def point_value(self, x: float, y: float) -> float:
        col, row = ~self.transform @ (x, y)
        r, c = int(row), int(col)
        if 0 <= r < self.data.shape[0] and 0 <= c < self.data.shape[1]:
            return float(self.data[r, c])
        return float("nan")


def tile_name(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def tile_path(lat_floor: int, lon_floor: int) -> Path:
    name = tile_name(lat_floor, lon_floor)
    return DEM_CACHE / name / f"{name}.tif"


def offline_mode() -> bool:
    return os.environ.get("GEOSITE_OFFLINE") == "1"


def ensure_tile(lat_floor: int, lon_floor: int, *, timeout: int = 300) -> Path:
    path = tile_path(lat_floor, lon_floor)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    if offline_mode():
        raise FileNotFoundError(f"режим --offline: тайла нет в кеше: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.parent.name
    url = f"{S3}/{name}/{name}.tif"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(path)
    return path


def tiles_for_bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list[tuple[int, int]]:
    lats = range(math.floor(lat_min), math.floor(math.ceil(lat_max) if lat_max % 1 == 0 else lat_max) + 1)
    lons = range(math.floor(lon_min), math.floor(math.ceil(lon_max) if lon_max % 1 == 0 else lon_max) + 1)
    return [(la, lo) for la in lats for lo in lons]


def bbox_for_radius(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def load_dem(lat: float, lon: float, radius_km: float, *, epsg: int, resolution_m: float,
             resampling: Resampling = Resampling.bilinear) -> DemGrid:
    """Скачать тайлы в охвате, склеить, перевести в UTM и обрезать по радиусу."""
    lat_min, lat_max, lon_min, lon_max = bbox_for_radius(lat, lon, radius_km)
    datasets = []
    try:
        for la, lo in tiles_for_bbox(lat_min, lat_max, lon_min, lon_max):
            datasets.append(rasterio.open(ensure_tile(la, lo)))
        mosaic, mosaic_transform = merge(datasets, bounds=(lon_min, lat_min, lon_max, lat_max))
        src_crs = datasets[0].crs
        height, width = mosaic.shape[-2:]
        band = mosaic[0].astype("float32")

        dst_crs = f"EPSG:{epsg}"
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, width, height, left=lon_min, bottom=lat_min, right=lon_max, top=lat_max,
            resolution=resolution_m)
        out = np.full((dst_height, dst_width), np.nan, dtype="float32")
        reproject(
            source=band, destination=out,
            src_transform=mosaic_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=resampling, src_nodata=None, dst_nodata=np.nan,
        )
        # Обрезка кругом радиуса: за пределами — NaN, чтобы метрики не считали чужие километры.
        rows, cols = np.mgrid[0:dst_height, 0:dst_width]
        xs = dst_transform.c + (cols + 0.5) * dst_transform.a
        ys = dst_transform.f + (rows + 0.5) * dst_transform.e
        cx = dst_transform.c + dst_width * dst_transform.a / 2
        cy = dst_transform.f + dst_height * dst_transform.e / 2
        dist = np.hypot(xs - cx, ys - cy)
        out[dist > radius_km * 1000.0] = np.nan
        return DemGrid(out, dst_transform, dst_crs, resolution_m)
    finally:
        for d in datasets:
            d.close()


def smooth(grid: DemGrid, window_m: float) -> DemGrid:
    """Сглаживание макроформы: убирает крыши и лес из модели поверхности.

    Крыши и кроны в DSM дают ложные «горы» высотой 10–30 м; для чтения хребтов и долин
    нужен фильтр порядка сотен метров.
    """
    k = max(3, int(round(window_m / grid.resolution_m)) | 1)
    data = grid.data
    pad = k // 2
    padded = np.pad(data, pad, mode="edge")
    csum = np.cumsum(np.cumsum(np.nan_to_num(padded, nan=0.0), axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
    counts = np.cumsum(np.cumsum((~np.isnan(padded)).astype(np.int64), axis=0), axis=1)
    counts = np.pad(counts, ((1, 0), (1, 0)), mode="constant")
    h, w = data.shape
    total = (csum[k:k + h, k:k + w] - csum[0:h, k:k + w] - csum[k:k + h, 0:w] + csum[0:h, 0:w]).astype("float64")
    cnt = (counts[k:k + h, k:k + w] - counts[0:h, k:k + w] - counts[k:k + h, 0:w] + counts[0:h, 0:w]).astype("float64")
    out = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan).astype("float32")
    out[np.isnan(data)] = np.nan
    return DemGrid(out, grid.transform, grid.crs, grid.resolution_m)


def gradient(grid: DemGrid) -> tuple[np.ndarray, np.ndarray]:
    """Уклон (градусы) и экспозиция (азимут ската, от севера)."""
    z = grid.data.astype("float64")
    res = grid.resolution_m
    dz_dy, dz_dx = np.gradient(z, res, res)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = np.degrees(np.arctan2(-dz_dx, dz_dy)) % 360.0
    return slope.astype("float32"), aspect.astype("float32")
