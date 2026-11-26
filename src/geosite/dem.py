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
SRTM_CACHE = CACHE_DIR / "srtm30"
S3 = "https://copernicus-dem-30m.s3.amazonaws.com"
SRTM_S3 = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
DEM_LABEL = "Copernicus DEM GLO-30 (DSM)"
SRTM_LABEL = "SRTM 1 arc-second (AWS Terrain Tiles, skadi)"
ATTRIBUTION = "produced using Copernicus WorldDEM-30 © DLR e.V."
SRTM_ATTRIBUTION = "SRTM 1-arc-second, NASA/USGS, via AWS Open Data (elevation-tiles-prod)"
USER_AGENT = "vector-geo-site/0.1"

# Порядок источников. Первый — основной; остальные пробуются по очереди, если он недоступен.
# Порядок, а не «только один»: Copernicus местами отдаёт 503 на весь бакет, и тогда участок
# оставался без рельефа вовсе — при том что рядом лежит открытый SRTM того же разрешения.
DEM_SOURCES = ("copernicus", "srtm")

# D2: паспорт слоя. `kind` — природа данных, а не имя файла: и Copernicus, и SRTM — радарная
# поверхность, то есть DSM, и «землёй без крыш» не являются (E-023). `vertical_datum` нужен,
# потому что высоты GLO-30 и skadi посчитаны в разных системах.
SOURCE_INFO: dict[str, dict] = {
    "copernicus": {"kind": "dsm_smoothed", "resolution_m": 30.0,
                   "license": "Copernicus DEM free licence",
                   "vertical_datum": "EGM2008"},
    "srtm": {"kind": "dsm_smoothed", "resolution_m": 30.0,
             "license": "public domain (NASA/USGS), AWS Open Data",
             "vertical_datum": "EGM96"},
}

# П7: уверенность поля, посчитанного по рельефу. Правило по `kind` и застройке, а не по имени
# источника: любой DSM видит крыши и кроны, и в плотной застройке высота земли по нему завышена
# на высоту зданий. DTM — земля, но GEDTM30 в городе и на побережье даёт не выше medium (D3).
DSM_CITY_MEDIUM = 0.2      # с этой застройки DSM перестаёт быть надёжным
DSM_CITY_LOW = 0.5         # здесь он врёт уже на высоту дома


def relief_confidence(kind: str, built: float | None) -> str:
    """П7: `high` / `medium` / `low` для поля, зависящего от рельефа."""
    if kind == "dtm":
        return "medium" if (built or 0.0) >= DSM_CITY_LOW else "high"
    density = built or 0.0
    if density >= DSM_CITY_LOW:
        return "low"
    if density >= DSM_CITY_MEDIUM:
        return "medium"
    return "high"


def file_block(source: str, *, kind: str | None = None,
               resolution_m: float | None = None) -> dict:
    """Блок `files.dem`: источник, природа, разрешение, лицензия, система высот."""
    info = SOURCE_INFO.get(source) or {}
    return {"source": source, "kind": kind or info.get("kind") or "dsm_smoothed",
            "resolution_m": resolution_m or info.get("resolution_m"),
            "license": info.get("license"), "vertical_datum": info.get("vertical_datum")}


@dataclass
class DemGrid:
    data: np.ndarray          # высоты, м, NaN где нет данных
    transform: object         # affine.Affine, UTM
    crs: str
    resolution_m: float
    source: str = "copernicus"    # id источника: пишется в пакет
    label: str = DEM_LABEL        # человекочитаемое имя
    attribution: str = ATTRIBUTION

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


def _reproject_to_utm(band: np.ndarray, transform, src_crs: str, *,
                      lat: float, lon: float, radius_km: float, epsg: int,
                      resolution_m: float, resampling: Resampling) -> DemGrid:
    """Мозаика в градусах → локальная UTM, обрезка кругом радиуса."""
    lat_min, lat_max, lon_min, lon_max = bbox_for_radius(lat, lon, radius_km)
    dst_crs = f"EPSG:{epsg}"
    height, width = band.shape
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, left=lon_min, bottom=lat_min, right=lon_max, top=lat_max,
        resolution=resolution_m)
    out = np.full((dst_height, dst_width), np.nan, dtype="float32")
    reproject(
        source=band, destination=out,
        src_transform=transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=resampling, src_nodata=None, dst_nodata=np.nan,
    )
    rows, cols = np.mgrid[0:dst_height, 0:dst_width]
    xs = dst_transform.c + (cols + 0.5) * dst_transform.a
    ys = dst_transform.f + (rows + 0.5) * dst_transform.e
    cx = dst_transform.c + dst_width * dst_transform.a / 2
    cy = dst_transform.f + dst_height * dst_transform.e / 2
    dist = np.hypot(xs - cx, ys - cy)
    out[dist > radius_km * 1000.0] = np.nan
    return DemGrid(out, dst_transform, dst_crs, resolution_m)


def _load_copernicus(lat: float, lon: float, radius_km: float, *, epsg: int, resolution_m: float,
                     resampling: Resampling) -> DemGrid:
    lat_min, lat_max, lon_min, lon_max = bbox_for_radius(lat, lon, radius_km)
    datasets = []
    try:
        for la, lo in tiles_for_bbox(lat_min, lat_max, lon_min, lon_max):
            datasets.append(rasterio.open(ensure_tile(la, lo)))
        mosaic, mosaic_transform = merge(datasets, bounds=(lon_min, lat_min, lon_max, lat_max))
        band = mosaic[0].astype("float32")
        grid = _reproject_to_utm(band, mosaic_transform, str(datasets[0].crs), lat=lat, lon=lon,
                                 radius_km=radius_km, epsg=epsg, resolution_m=resolution_m,
                                 resampling=resampling)
        grid.source, grid.label, grid.attribution = "copernicus", DEM_LABEL, ATTRIBUTION
        return grid
    finally:
        for d in datasets:
            d.close()


def srtm_tile_name(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}"


def srtm_tile_path(lat_floor: int, lon_floor: int) -> Path:
    name = srtm_tile_name(lat_floor, lon_floor)
    return SRTM_CACHE / name[:3] / f"{name}.hgt.gz"


def ensure_srtm_tile(lat_floor: int, lon_floor: int, *, timeout: int = 300) -> Path:
    """Тайл SRTM 1×1° (gzip HGT) из открытого бакета AWS; кладётся в кеш как есть."""
    path = srtm_tile_path(lat_floor, lon_floor)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    if offline_mode():
        raise FileNotFoundError(f"режим --offline: тайла нет в кеше: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{SRTM_S3}/{path.name[:3]}/{path.name}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(path)
    return path


def read_hgt(path) -> tuple[np.ndarray, object]:
    """HGT → (высоты, transform). Файл: int16 big-endian, 3601×3601 (1 арк-секунда)."""
    import gzip

    from affine import Affine

    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    side = int(round(math.sqrt(len(raw) / 2)))
    if side * side * 2 != len(raw):
        raise ValueError(f"hgt: {path} не квадрат 2-байтовых отсчётов ({len(raw)} B)")
    data = np.frombuffer(raw, dtype=">i2").reshape(side, side).astype("float32")
    data[data <= -32000] = np.nan                  # -32768 = нет данных (океан, дыры)
    step = 1.0 / (side - 1)
    lat_floor = int(path.stem.split(".")[0][1:3]) * (1 if path.stem[0] == "N" else -1)
    lon_floor = int(path.stem.split(".")[0][4:7]) * (1 if path.stem[3] == "E" else -1)
    return data, Affine(step, 0.0, lon_floor, 0.0, -step, lat_floor + 1.0)


def _load_srtm(lat: float, lon: float, radius_km: float, *, epsg: int, resolution_m: float,
               resampling: Resampling) -> DemGrid:
    lat_min, lat_max, lon_min, lon_max = bbox_for_radius(lat, lon, radius_km)
    tiles = tiles_for_bbox(lat_min, lat_max, lon_min, lon_max)
    blocks = []
    for la, lo in tiles:
        data, transform = read_hgt(ensure_srtm_tile(la, lo))
        blocks.append((la, lo, data, transform))
    if len(blocks) == 1:
        band, transform = blocks[0][2], blocks[0][3]
    else:
        # мозаика вручную: все тайлы одного размера и одной сетки 1°, зазор не нужен
        side = blocks[0][2].shape[0]
        lat_lo = min(la for la, *_ in blocks)
        lat_hi = max(la for la, *_ in blocks)
        lon_lo = min(lo for _, lo, *_ in blocks)
        lon_hi = max(lo for _, lo, *_ in blocks)
        rows, cols = lat_hi - lat_lo + 1, lon_hi - lon_lo + 1
        band = np.full((rows * side, cols * side), np.nan, dtype="float32")
        for la, lo, data, _ in blocks:
            r = (lat_hi - la) * side
            c = (lo - lon_lo) * side
            band[r:r + side, c:c + side] = data
        from affine import Affine
        step = 1.0 / (side - 1)
        transform = Affine(step, 0.0, lon_lo, 0.0, -step, lat_hi + 1.0)
    grid = _reproject_to_utm(band, transform, "EPSG:4326", lat=lat, lon=lon, radius_km=radius_km,
                             epsg=epsg, resolution_m=resolution_m, resampling=resampling)
    grid.source, grid.label, grid.attribution = "srtm", SRTM_LABEL, SRTM_ATTRIBUTION
    return grid


def loaders() -> dict:
    """Источники по имени. Собираем на каждый вызов: словарь, собранный при импорте, держал бы
    старые функции, и подмена источника (в тестах и в `GEOSITE_DEM`) на него не действовала."""
    return {"copernicus": _load_copernicus, "srtm": _load_srtm}


def load_dem(lat: float, lon: float, radius_km: float, *, epsg: int, resolution_m: float,
             resampling: Resampling = Resampling.bilinear, source: str | None = None) -> DemGrid:
    """Рельеф участка: первый доступный источник из `DEM_SOURCES`.

    `GEOSITE_DEM=copernicus|srtm|auto` фиксирует источник руками (нужно для сравнения чисел).
    Молчаливой подмены нет: источник пишется в `DemGrid.source` и в пакет (`quality.dem`),
    потому что SRTM — не DSM: он не видит крыши и кроны, и «горы» на нём ниже.
    """
    available = loaders()
    order = list(DEM_SOURCES)
    requested = (source or os.environ.get("GEOSITE_DEM") or "auto").lower()
    if requested in available:
        order = [requested]
    errors: list[str] = []
    for name in order:
        try:
            return available[name](lat, lon, radius_km, epsg=epsg, resolution_m=resolution_m,
                                   resampling=resampling)
        except Exception as exc:                    # нет сети, нет кеша, битый тайл
            errors.append(f"{name}: {exc}")
    raise RuntimeError("рельеф недоступен ни из одного источника: " + "; ".join(errors))


def load_dem_file(path) -> DemGrid:
    """DEM из подготовленного файла — для офлайн-эталонов tests/golden."""
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        return DemGrid(data, src.transform, str(src.crs), float(abs(src.transform.a)))


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
