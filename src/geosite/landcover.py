"""Покров ESA WorldCover 10 м: независимая сверка воды и доли застройки, деревьев, открытого.

Зачем отдельный слой. Вода по OSM — это чужая разметка: где реку нарисовали осевой, где забыли
про пруд, где старая выгрузка. WorldCover даёт независимый ответ по снимкам 2021 года, и
расхождение двух источников — это не «ошибка», а повод записать причину (W3 ТЗ world).

Данные: ESA WorldCover 2021 v200, 10 м, тайлы 3° × 3° в EPSG:4326, лицензия CC BY 4.0.
Читаются окнами по HTTP (COG) и кешируются вырезанным окном: тайл целиком — 100 МБ, а расчёту
нужен квадрат 16 км вокруг участка.

Классы: 10 дерево, 20 кустарник, 30 трава, 40 посевы, 50 застройка, 60 голая земля, 70 снег,
80 вода, 90 болото, 95 мангры, 100 мох. Вода — только класс 80: болото и мангры это не открытая
вода, и записывать их в долю воды значит объяснять расхождение с OSM своей же ошибкой.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np

from . import geo

ALGO_VERSION = "1.0.0"
SOURCE = "ESA WorldCover 2021 v200"
LICENSE = "CC BY 4.0"
RESOLUTION_M = 10.0
TILE_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
            "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
CACHE_DIR = Path(os.environ.get("GEOSITE_WORLDCOVER_CACHE")
                 or Path.home() / ".cache" / "vector-geo-site" / "worldcover")

CLASS_NAMES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
               60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove",
               100: "moss"}
# Группы покрова: что считаем деревом, застройкой и «открытым» (трава, посевы, голая земля)
GROUPS = {"tree": {10}, "built": {50}, "open": {20, 30, 40, 60, 70, 100}, "water": {80}}
M_PER_DEG_LAT = 111132.0


def tile_name(lat: float, lon: float) -> str:
    """Имя тайла ESA: юго-западный угол, шаг 3°, широта 2 знака, долгота 3."""
    lat0 = int(math.floor(lat / 3.0) * 3)
    lon0 = int(math.floor(lon / 3.0) * 3)
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def tile_url(lat: float, lon: float) -> str:
    return TILE_URL.format(tile=tile_name(lat, lon))


def _bbox(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Рамка в градусах вокруг участка: широта по меридиану, долгота по параллели."""
    dlat = radius_m / M_PER_DEG_LAT
    dlon = radius_m / (M_PER_DEG_LAT * max(0.05, math.cos(math.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def _cache_path(lat: float, lon: float, radius_m: float) -> Path:
    key = hashlib.sha256(f"{tile_name(lat, lon)},{lat:.3f},{lon:.3f},{radius_m:.0f}".encode())
    return CACHE_DIR / f"wc-{key.hexdigest()[:16]}.npz"


def window(lat: float, lon: float, radius_m: float = 8000.0, *, path: str | Path | None = None,
           offline: bool = False) -> dict | None:
    """Окно покрова вокруг участка: классы и аффинное преобразование.

    Возвращает `{"array", "transform", "tile", "cached"}` либо None, если данных нет:
    офлайн без кеша, тайл не существует, сеть недоступна. Молчаливой подстановки нулей нет —
    ноль в покрове означал бы «здесь ничего нет», а не «мы не смотрели».
    """
    if path:
        # имя тайла берём из координат, а не из имени файла: вырезка в фикстуре названа
        # worldcover.tif, и в пакете должно стоять, какой тайл ESA за ней стоит
        return _window_from_file(Path(path), tile_name(lat, lon))
    cached = _cache_path(lat, lon, radius_m)
    if cached.exists():
        with np.load(cached) as data:
            return {"array": data["array"], "transform": tuple(float(v) for v in data["transform"]),
                    "tile": str(data["tile"]), "cached": True}
    if offline:
        return None
    got = _window_remote(lat, lon, radius_m)
    if got is None:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, array=got["array"],
                        transform=np.array(got["transform"], dtype="float64"),
                        tile=np.array(got["tile"]))
    return got


def _window_from_file(path: Path, tile: str = "") -> dict | None:
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as ds:
        array = ds.read(1)
        transform = tuple(ds.transform)[:6]
        tile = tile or path.stem
    if not array.size:
        return None
    return {"array": array, "transform": transform, "tile": tile, "cached": True}


def _window_remote(lat: float, lon: float, radius_m: float) -> dict | None:
    import rasterio
    from rasterio.windows import from_bounds

    url = tile_url(lat, lon)
    left, bottom, right, top = _bbox(lat, lon, radius_m)
    try:
        with rasterio.open(url) as ds:
            window = from_bounds(left, bottom, right, top, ds.transform)
            window = window.round_offsets().round_lengths()
            array = ds.read(1, window=window)
            transform = tuple(ds.window_transform(window))[:6]
    except Exception:
        return None
    if not array.size:
        return None
    return {"array": array, "transform": transform, "tile": tile_name(lat, lon), "cached": False}


def _pixel_coords(data: dict, crs: int | str, site_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Координаты центров пикселей относительно участка в UTM.

    Пересчёт идёт через градусы: WorldCover в EPSG:4326, а сектора и кольца считаются в UTM.
    Считаем оси отдельно (широта × долгота) и складываем сеткой — на 8 км ошибка раздельного
    пересчёта меньше метра, а 2.5 млн точек одним вызовом pyproj стоят секунд.
    """
    from pyproj import Transformer

    a, b, c, d, e, f = data["transform"]
    array = data["array"]
    rows, cols = array.shape
    lons = c + (np.arange(cols) + 0.5) * a
    lats = f + (np.arange(rows) + 0.5) * e
    to_utm = Transformer.from_crs("EPSG:4326", geo.crs_str(crs), always_xy=True)
    xs, _ = to_utm.transform(lons, np.full_like(lons, float(lats.mean())))
    _, ys = to_utm.transform(np.full_like(lats, float(lons.mean())), lats)
    return xs - site_xy[0], ys - site_xy[1]


def write_crop(lat: float, lon: float, radius_m: float, path: str | Path, *,
               offline: bool = False) -> bool:
    """Сохранить вырезку покрова в фикстуру эталона: GeoTIFF uint8, EPSG:4326.

    Хранить тайл целиком (100 МБ) в репозитории нельзя, а окно 16 км сжимается до сотен
    килобайт и воспроизводит расчёт офлайн бит в бит.
    """
    import rasterio
    from rasterio.transform import Affine

    data = window(lat, lon, radius_m, offline=offline)
    if data is None:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = Affine(*data["transform"])
    with rasterio.open(path, "w", driver="GTiff", height=data["array"].shape[0],
                       width=data["array"].shape[1], count=1, dtype="uint8",
                       crs="EPSG:4326", transform=transform, compress="deflate") as dst:
        dst.write(data["array"], 1)
    return True


def measure(lat: float, lon: float, *, crs: int | str, site_xy: np.ndarray,
            rings: tuple[tuple[float, float], ...] = ((500.0, 2000.0), (2000.0, 8000.0)),
            radius_m: float = 8000.0, path: str | Path | None = None,
            offline: bool = False) -> dict | None:
    """Доли покрова и воды по секторам — независимо от OSM.

    `groups` считается по первому кольцу (0.5–2 км), как `land_cover.water` в пакете:
    сравнивать застройку в кольце 2–8 км с водой в 0.5–2 км бессмысленно.
    """
    from . import geo

    data = window(lat, lon, radius_m, path=path, offline=offline)
    if data is None:
        return None
    array = data["array"]
    xs, ys = _pixel_coords(data, crs, site_xy)
    xg, yg = np.meshgrid(xs, ys)
    radius = np.hypot(xg, yg)
    inside = radius <= radius_m
    bearing = (np.degrees(np.arctan2(xg, yg)) % 360.0)
    sector = ((bearing + 7.5) // 15.0).astype(np.int32) % 24
    total = int(inside.sum())
    if total == 0:
        return None

    by_ring: dict[str, dict] = {}
    for r0, r1 in rings:
        band = inside & (radius >= r0) & (radius < r1)
        count = int(band.sum())
        if count == 0:
            continue
        by_mountain: dict[str, float] = {}
        for index, name in enumerate(geo.MOUNTAINS):
            cell = band & (sector == index)
            cell_count = int(cell.sum())
            if cell_count == 0:
                by_mountain[name] = 0.0
                continue
            water_cell = int(np.isin(array[cell], list(GROUPS["water"])).sum())
            by_mountain[name] = round(water_cell / cell_count, 4)
        water_total = int(np.isin(array[band], list(GROUPS["water"])).sum())
        by_ring[f"{r0 / 1000:g}-{r1 / 1000:g}"] = {
            "share": round(water_total / count, 4),
            "by_mountain": by_mountain,
            "pixels": count,
        }

    r0, r1 = rings[0] if rings else (500.0, 2000.0)
    band = inside & (radius >= r0) & (radius < r1)
    group_share: dict[str, float] = {}
    count = int(band.sum())
    for group, classes in GROUPS.items():
        hits = int(np.isin(array[band], list(classes)).sum()) if count else 0
        group_share[group] = round(hits / count, 4) if count else 0.0
    return {"source": SOURCE, "license": LICENSE, "resolution_m": RESOLUTION_M,
            "tile": data["tile"], "cached": bool(data["cached"]),
            "groups": group_share, "groups_ring_km": [r0 / 1000, r1 / 1000],
            "rings": by_ring, "pixels": total}


__all__ = ["ALGO_VERSION", "SOURCE", "LICENSE", "RESOLUTION_M", "CLASS_NAMES", "GROUPS",
           "tile_name", "tile_url", "window", "write_crop", "measure"]
