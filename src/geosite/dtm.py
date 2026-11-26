"""DTM: земля вместо поверхности (D3).

Цепочка одинакова для любой точки: национальный лидар (если точка внутри покрытия и лицензия
открытая) → GEDTM30 → нет DTM (расчёт по DSM, понижение уверенности по П7, предупреждение
`no_dtm`). Режим фиксируется `GEOSITE_DTM=auto|lidar|gedtm30|none`.

Почему это отдельный модуль, а не флаг в `dem`: DSM — поверхность (крыши, кроны), DTM — земля.
Смешать их в одном гриде просто, а различить потом невозможно: числа выглядят одинаково.
Поэтому у каждого поля в пакете есть `dem_layer`, а здесь — паспорт слоя.
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from . import dem

ALGO_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data" / "dtm_sources.json"
CACHE_DIR = dem.CACHE_DIR / "dtm"
USER_AGENT = "vector-geo-site/0.1"
MODES = ("auto", "lidar", "gedtm30", "none")
# Верхняя граница высот: у AHN nodata — 3.4e38, и без отсечки среднее становится 2.4e37
MAX_ELEV_M = 9000.0
MIN_ELEV_M = -500.0


def mode() -> str:
    value = (os.environ.get("GEOSITE_DTM") or "auto").strip().lower()
    return value if value in MODES else "auto"


def offline() -> bool:
    return os.environ.get("GEOSITE_OFFLINE") == "1"


def registry() -> list[dict]:
    try:
        return json.loads(REGISTRY.read_text(encoding="utf-8")).get("sources") or []
    except Exception:
        return []


def lidar_for(lat: float, lon: float, *, only_active: bool = True) -> dict | None:
    """Первый лидар из реестра, покрывающий точку: страна, полигон, открытая лицензия."""
    for item in registry():
        if item.get("kind") == "cog-tiles":
            continue
        if item.get("license_class") != "open":
            continue
        if only_active and item.get("active") is False:
            continue
        south, west, north, east = item["bbox"]
        if south <= lat <= north and west <= lon <= east:
            return item
    return None


def _utm_size(radius_km: float, resolution_m: float) -> int:
    return max(16, int(round(2 * radius_km * 1000.0 / resolution_m)))


def _mask_heights(array: np.ndarray) -> np.ndarray:
    """Nodata и абсурд — в NaN: иначе одна ячейка 3.4e38 делает среднее бессмысленным."""
    out = array.astype("float32", copy=True)
    bad = ~np.isfinite(out) | (out > MAX_ELEV_M) | (out < MIN_ELEV_M)
    out[bad] = np.nan
    return out


def _wcs_window(source: dict, lat: float, lon: float, radius_km: float,
                resolution_m: float, *, timeout: int = 120) -> np.ndarray | None:
    """Окно из WCS 1.0.0 в EPSG:4326: сервер сам пересчитывает сетку под WIDTH/HEIGHT."""
    from rasterio.io import MemoryFile

    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    size = _utm_size(radius_km, resolution_m)
    params = {
        "service": "WCS", "request": "GetCoverage", "version": "1.0.0",
        "coverage": source["coverage"], "CRS": "EPSG:4326",
        "BBOX": f"{lon - dlon:.6f},{lat - dlat:.6f},{lon + dlon:.6f},{lat + dlat:.6f}",
        "WIDTH": str(size), "HEIGHT": str(size), "FORMAT": "GeoTIFF",
    }
    url = f"{source['url']}?{urllib.parse.urlencode(params)}"
    cache = CACHE_DIR / f"{source['id']}_{lat:.4f}_{lon:.4f}_{radius_km:g}_{resolution_m:g}.npz"
    cached = _read_npz(cache)
    if cached is not None:
        return cached
    if offline():
        return None
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
    with MemoryFile(payload) as mem, mem.open() as ds:
        band = ds.read(1)
        transform = tuple(ds.transform)[:6]
        crs = str(ds.crs)
    array = _mask_heights(band)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, array=array, transform=np.array(transform), crs=crs)
    return {"array": array, "transform": transform, "crs": crs}


def _read_npz(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return {"array": data["array"], "transform": tuple(float(v) for v in data["transform"]),
                "crs": str(data["crs"])}
    except Exception:
        return None


def _tile_name(lat: float, lon: float) -> str:
    """GEDTM30: имя тайла — 1° юго-западный угол (N51E000), как в репозитории OpenGeoHub."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(int(math.floor(lat))):02d}{ew}{abs(int(math.floor(lon))):03d}"


def _gedtm30_window(source: dict, lat: float, lon: float, radius_km: float,
                    resolution_m: float, *, timeout: int = 180) -> dict | None:
    """COG-тайл GEDTM30 читается окном 16 км; тайл целиком (десятки МБ) не тянем."""
    import rasterio
    from rasterio.windows import from_bounds

    if offline():
        raise RuntimeError("offline: сеть запрещена")

    base = os.environ.get("GEOSITE_GEDTM30_URL") or source["url"]
    pattern = source.get("tile_pattern") or "GEDTM30_v1.2.0/{name}.tif"
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    lats = range(int(math.floor(lat - dlat)), int(math.floor(lat + dlat)) + 1)
    lons = range(int(math.floor(lon - dlon)), int(math.floor(lon + dlon)) + 1)
    parts = []
    for la in lats:
        for lo in lons:
            name = _tile_name(la + 0.5, lo + 0.5)
            url = f"{base}/{pattern.format(name=name)}"
            try:
                with rasterio.open(url) as ds:
                    window = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, ds.transform)
                    band = ds.read(1, window=window, boundless=True)
                    transform = ds.window_transform(window)
                    parts.append((band, tuple(transform)[:6], str(ds.crs)))
            except Exception:
                continue
    if not parts:
        return None
    band, transform, crs = parts[0]
    return {"array": _mask_heights(band), "transform": transform, "crs": crs}


def grid(lat: float, lon: float, *, epsg: int, radius_km: float, resolution_m: float = 30.0
         ) -> tuple[dem.DemGrid | None, dict]:
    """DTM для точки: (грид, паспорт). Если DTM нет — `(None, {reason, tried})`."""
    chosen = mode()
    tried: list[str] = []
    info: dict = {"reason": None, "tried": tried}

    def _build(source: dict, reader) -> dem.DemGrid | None:
        try:
            window = reader(source, lat, lon, radius_km, resolution_m)
        except Exception as exc:                   # сеть, формат, отказ сервера — не падаем
            tried.append(f"{source['id']}: {type(exc).__name__}")
            return None
        if not window:
            tried.append(f"{source['id']}: пусто")
            return None
        grid_obj = dem._reproject_to_utm(window["array"], window["transform"], window["crs"],
                                         lat=lat, lon=lon, radius_km=radius_km, epsg=epsg,
                                         resolution_m=resolution_m, resampling=dem.Resampling.bilinear)
        grid_obj.source = source["id"]
        grid_obj.label = source["name"]
        grid_obj.attribution = source.get("license") or ""
        grid_obj.kind = "dtm"
        grid_obj.vertical_datum = source.get("vertical_datum")
        grid_obj.datum_shift_m = source.get("datum_shift_m")
        grid_obj.license = source.get("license")
        grid_obj.resolution_m = float(source.get("resolution_m") or resolution_m)
        info.update({"source": source["id"], "name": source["name"], "kind": "dtm",
                     "license": source.get("license"), "vertical_datum": source.get("vertical_datum"),
                     "datum_shift_m": source.get("datum_shift_m"),
                     "datum_note": source.get("datum_note"),
                     "resolution_m": grid_obj.resolution_m, "tried": tried})
        return grid_obj

    if chosen == "none":
        info["reason"] = "режим GEOSITE_DTM=none"
        return None, info

    if chosen in ("auto", "lidar"):
        source = lidar_for(lat, lon, only_active=chosen == "auto")
        if source is None:
            tried.append("лидар: точка вне покрытия или источник не активен")
        else:
            result = _build(source, _wcs_window)
            if result is not None:
                return result, info
    if chosen in ("auto", "gedtm30"):
        global_source = next((s for s in registry() if s.get("id") == "gedtm30"), None)
        if global_source is not None and chosen == "auto" and global_source.get("active") is False:
            tried.append("gedtm30 помечен недоступным при последней проверке")
            global_source = None
        if global_source is None:
            if not any("gedtm30" in item for item in tried):
                tried.append("gedtm30 нет в реестре")
        else:
            result = _build(global_source, _gedtm30_window)
            if result is not None:
                return result, info
    info["reason"] = "; ".join(tried) or "источники не ответили"
    return None, info


def file_block(info: dict) -> dict | None:
    """`files.dtm`: тот же паспорт, что у `files.dem` (D2)."""
    if not info.get("source"):
        return None
    return {"source": info["source"], "kind": "dtm", "resolution_m": info.get("resolution_m"),
            "license": info.get("license"), "vertical_datum": info.get("vertical_datum"),
            "datum_shift_m": info.get("datum_shift_m")}


__all__ = ["ALGO_VERSION", "MODES", "REGISTRY", "mode", "registry", "lidar_for", "grid",
           "file_block"]
