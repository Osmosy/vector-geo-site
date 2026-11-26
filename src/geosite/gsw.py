"""JRC Global Surface Water (GSW): второй голос по воде, а не замена OSM.

Зачем: OSM рисует воду по контуру, но не говорит, стоит ли она круглый год. Приливную реку
и постоянное озеро там видно одинаково, а от этого зависит, что считать водой в кольце.
Снимок даёт третий факт: сколько месяцев в году пиксель был под водой (сезонность) и как часто
вода там была вообще (occurrence).

Правила те же, что с WorldCover (W3/W4): **основа — OSM**, снимок не заменяет его числа,
а даёт второе мнение и повод понизить уверенность с причиной. Подгонять значение под снимок
запрещено.

Данные: JRC Global Surface Water v1.4 (2021), плитки 10°×10°, 30 м (0.00025°), EPSG:4326,
uint8 в процентах. Скачиваются прямыми range-запросами с `storage.googleapis.com` — без
Earth Engine, аккаунта и асинхронного экспорта.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from . import dem

ALGO_VERSION = "1.0.0"
BASE = "https://storage.googleapis.com/global-surface-water/downloads2021"
VERSION_TAG = "v1_4_2021"
RES_DEG = 0.00025
CACHE_DIR = dem.CACHE_DIR / "gsw"
LICENSE = "JRC Global Surface Water v1.4 — Copernicus, свободно с указанием (Pekel et al., 2016)"
SOURCE = "jrc-gsw"
# Слои: occurrence — % месяцев с водой? нет: % наблюдений с водой; seasonality — месяцев в году
LAYERS = ("occurrence", "seasonality")


def offline() -> bool:
    return os.environ.get("GEOSITE_OFFLINE") == "1"


def tile_name(lat: float, lon: float) -> str:
    """Имя плитки 10°×10°: западная и северная границы (`10W_60N` = lon −10…0, lat 50…60).

    Северная граница — верх плитки, поэтому ровно 50° попадает в `60N`, а не в `50N`:
    плитка `50N` покрывает 40…50 и точки 50.0 уже не содержит.
    """
    west = math.floor(lon / 10.0) * 10
    north = (math.floor(lat / 10.0) + 1) * 10
    lon_label = f"{abs(int(west))}W" if west < 0 else f"{int(west)}E"
    lat_label = f"{abs(int(north))}N" if north > 0 else f"{abs(int(north))}S"
    return f"{lon_label}_{lat_label}"


def tile_url(layer: str, name: str) -> str:
    return f"{BASE}/{layer}/{layer}_{name}{VERSION_TAG}.tif"


def _window(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def _read(layer: str, lat: float, lon: float, radius_km: float, *,
          allow_network: bool = True, timeout: int = 120) -> dict | None:
    """Окно слоя: сначала кеш (`.npz`), затем range-запрос к COG-плитке.

    Возвращает массив **и** аффинное преобразование: без него нельзя отсечь круг, а квадрат
    и круг дают разные доли — сравнение с OSM было бы нечестным (площадь разная).
    """
    name = tile_name(lat, lon)
    cache = CACHE_DIR / f"{layer}_{name}_{lat:.4f}_{lon:.4f}_{radius_km:g}.npz"
    if cache.exists():
        try:
            data = np.load(cache, allow_pickle=False)
            return {"array": data["array"], "transform": tuple(float(v) for v in data["transform"])}
        except Exception:
            pass
    if offline() or not allow_network:
        return None
    import rasterio
    from rasterio.windows import from_bounds

    left, bottom, right, top = _window(lat, lon, radius_km)
    url = tile_url(layer, name)
    try:
        with rasterio.open(f"/vsicurl/{url}?timeout={timeout}") as ds:
            # окно могло выйти за плитку — прижимаем к её границам
            b = (max(left, ds.bounds.left), max(bottom, ds.bounds.bottom),
                 min(right, ds.bounds.right), min(top, ds.bounds.top))
            if b[0] >= b[2] or b[1] >= b[3]:
                return None
            win = from_bounds(*b, transform=ds.transform)
            array = ds.read(1, window=win, boundless=True, fill_value=255)
            transform = tuple(ds.window_transform(win))[:6]
    except Exception:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, array=array, transform=np.array(transform))
    return {"array": array, "transform": transform}


def metrics(lat: float, lon: float, radius_km: float, *, allow_network: bool = True) -> dict | None:
    """Доли воды по снимку для круга радиуса `radius_km`.

    `255` — вне данных (океанская часть плиток и края), такие пиксели не считаются вовсе:
    «не измерено» ≠ «сухо».
    """
    occ_data = _read("occurrence", lat, lon, radius_km, allow_network=allow_network)
    if occ_data is None:
        return None
    occ = occ_data["array"]
    # круг, а не квадрат: сравнивать с кольцом OSM надо одну и ту же площадь
    rows, cols = np.mgrid[0:occ.shape[0], 0:occ.shape[1]]
    a, b, c, d, e, f = occ_data["transform"]
    xs = a * (cols + 0.5) + b * (rows + 0.5) + c
    ys = d * (cols + 0.5) + e * (rows + 0.5) + f
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    inside = np.hypot((xs - lon) * 111320.0 * cos_lat, (ys - lat) * 110574.0) <= radius_km * 1000.0
    valid = (occ != 255) & inside
    if not valid.any():
        return None
    season_data = _read("seasonality", lat, lon, radius_km, allow_network=allow_network)
    season = season_data["array"] if season_data else None
    any_water = (occ > 0) & valid
    permanent = (occ >= 50) & valid
    result = {
        "source": SOURCE,
        "version": VERSION_TAG,
        "license": LICENSE,
        "tile": tile_name(lat, lon),
        "resolution_m": 30.0,
        "radius_km": radius_km,
        "pixels": int(valid.sum()),
        "share_any": round(float(any_water.sum()) / float(valid.sum()), 4),
        "share_permanent": round(float(permanent.sum()) / float(valid.sum()), 4),
    }
    if season is not None and season.shape == occ.shape:
        seasonal = permanent & (season >= 1) & (season < 12)
        result["share_seasonal"] = round(float(seasonal.sum()) / float(valid.sum()), 4)
        months = season[permanent & (season > 0) & (season < 255)]
        result["months_water_median"] = float(np.median(months)) if months.size else None
    return result


def compare(osm_share: float | None, gsw_metrics: dict | None, *, tolerance: float = 0.03) -> dict:
    """Расхождение OSM и снимка: причина и цена выбора, без подгонки чисел.

    `osm_share` — доля воды в кольце по полигонам OSM (0…1). Сравнение идёт с `share_permanent`:
    осевые реки OSM в долю не входят, а снимок их видит, поэтому расхождение в сторону снимка
    — нормальный случай, а не ошибка.
    """
    if gsw_metrics is None or osm_share is None:
        return {"status": "unknown", "note": "снимок или OSM недоступны — сверки нет"}
    delta = float(gsw_metrics["share_permanent"]) - float(osm_share)
    status = "ok" if abs(delta) <= tolerance else "differs"
    return {
        "status": status,
        "osm_share": round(float(osm_share), 4),
        "gsw_share": gsw_metrics["share_permanent"],
        "delta": round(delta, 4),
        "tolerance": tolerance,
        "note": ("OSM и снимок сходятся в пределах допуска" if status == "ok"
                 else "OSM и снимок расходятся: вода в OSM нарисована контуром, снимок считает "
                      "по пикселям; причина остаётся за OSM, снимок — второе мнение"),
    }


__all__ = ["ALGO_VERSION", "BASE", "LAYERS", "LICENSE", "SOURCE", "VERSION_TAG", "compare",
           "metrics", "offline", "tile_name", "tile_url"]
