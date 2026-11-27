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
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from . import dem, geo

ALGO_VERSION = "1.2.1"
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
    """Первый **местный** лидар, покрывающий точку: страна, полигон, открытая лицензия.

    Глобальные источники (`fallback: true`, GEDTM30) сюда не попадают: у них покрытие —
    весь мир, и они вернулись бы на любую точку, выдав себя за национальный лидар.
    """
    for item in registry():
        if item.get("kind") == "cog-tiles" or item.get("fallback"):
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


def _wcs2_window(source: dict, lat: float, lon: float, radius_km: float,
                 resolution_m: float, *, timeout: int = 300) -> dict | None:
    """Окно из WCS 2.0.1: подмножества `subset=E(…)&subset=N(…)` в нативной сетке источника.

    У EA LIDAR (Англия) другого способа нет: сервис отдаёт только 2.0.1, осей «широта/долгота»
    не признаёт (`InvalidAxisLabel`), а `scaleFactor` масштабирует всё покрытие целиком — на
    Англию это 30.9 ГБ против лимита сервера 10 ГБ. Ответ при этом ограничен 50 МБ, поэтому
    окно режется на куски и собирается в один массив: один запрос «всё кольцо 8 км» сервер
    отвергает (`too much data … 244.14MB`).
    """
    from pyproj import Transformer
    from rasterio.io import MemoryFile

    if offline():
        raise RuntimeError("offline: сеть запрещена")

    native = source.get("crs") or "EPSG:27700"
    axes = source.get("axes") or ["E", "N"]
    native_res = float(source.get("resolution_m") or 1.0)
    cache = CACHE_DIR / f"{source['id']}_{lat:.4f}_{lon:.4f}_{radius_km:g}_{resolution_m:g}.npz"
    cached = _read_npz(cache)
    if cached is not None:
        return cached

    # окно шире радиуса: квадрат в OSGB, повёрнутый в UTM, иначе оставляет пустые углы
    # (1 − π/4 ≈ 21 % площади) — это дырка от проекции, а не отсутствие данных
    radius_km = radius_km * float(source.get("window_factor") or 1.45)
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    tr = Transformer.from_crs("EPSG:4326", geo.crs_str(native), always_xy=True)
    xs, ys = [], []
    for f_lat in (lat - dlat, lat, lat + dlat):
        for f_lon in (lon - dlon, lon, lon + dlon):
            x, y = tr.transform(f_lon, f_lat)
            xs.append(x)
            ys.append(y)
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)

    # сервер пишет ответ целиком, лимит по умолчанию 50 МБ → кусок ≈ 3.3×3.3 тыс. px
    max_bytes = float(source.get("max_bytes") or 45_000_000)
    side_limit = max(256, int(math.floor(math.sqrt(max_bytes / 4.0))))
    cols = max(1, int(math.ceil((x_max - x_min) / native_res / side_limit)))
    rows = max(1, int(math.ceil((y_max - y_min) / native_res / side_limit)))
    width = int(round((x_max - x_min) / native_res))
    height = int(round((y_max - y_min) / native_res))
    total_limit = float(source.get("max_total_pixels") or 80_000_000)
    if width * height > total_limit:
        raise RuntimeError(
            f"окно {width}x{height} px ({width * height / 1e6:.0f} Мпикс) больше лимита источника "
            f"{total_limit / 1e6:.0f} Мпикс: радиус {radius_km:g} км при сетке {native_res:g} м — "
            f"это {width * height * 4 / 1e6:.0f} МБ перекачки")
    mosaic = np.full((height, width), np.nan, dtype="float32")
    got = 0
    errors: list[str] = []
    for row in range(rows):
        for col in range(cols):
            cx0 = x_min + (x_max - x_min) * col / cols
            cx1 = x_min + (x_max - x_min) * (col + 1) / cols
            cy1 = y_max - (y_max - y_min) * row / rows
            cy0 = y_max - (y_max - y_min) * (row + 1) / rows
            query = [
                ("service", "WCS"), ("request", "GetCoverage"), ("version", "2.0.1"),
                ("CoverageId", source["coverage"]),
                ("subset", f"{axes[0]}({cx0:.2f},{cx1:.2f})"),
                ("subset", f"{axes[1]}({cy0:.2f},{cy1:.2f})"),
                ("format", "image/tiff"),
            ]
            url = f"{source['url']}?{urllib.parse.urlencode(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = resp.read()
            except Exception as exc:                     # кусок мог не отдаться — остальное берём
                errors.append(f"{type(exc).__name__}")
                continue
            if payload[:5] == b"<?xml" or payload[:4] == b"<ows":
                text = payload.decode("utf-8", "replace")
                reason = text.split("<ows:ExceptionText>")[1].split("<")[0] if "<ows:ExceptionText>" in text else text[:120]
                errors.append(reason)
                continue
            with MemoryFile(payload) as mem, mem.open() as ds:
                band = _mask_heights(ds.read(1))
                ctransform = ds.transform
                ccrs = str(ds.crs)
            # кусок кладём по его собственному углу: сервер вправе сдвинуть сетку на пиксель
            left = int(round((ctransform.c - x_min) / native_res))
            top = int(round((y_max - ctransform.f) / native_res))
            h, w = band.shape
            r0, c0 = max(0, top), max(0, left)
            r1, c1 = min(height, top + h), min(width, left + w)
            if r1 <= r0 or c1 <= c0:
                errors.append("кусок вне окна")
                continue
            mosaic[r0:r1, c0:c1] = band[r0 - top:r1 - top, c0 - left:c1 - left]
            got += 1
    if got == 0:
        raise RuntimeError("WCS 2.0.1: ни один кусок не отдан — " + "; ".join(sorted(set(errors))[:2]))
    from rasterio.transform import from_origin
    transform = from_origin(x_min, y_max, native_res, native_res)
    result = {"array": mosaic, "transform": tuple(transform)[:6], "crs": ccrs if got else geo.crs_str(native)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, array=mosaic, transform=np.array(result["transform"]), crs=result["crs"])
    return result


def _wcs_window(source: dict, lat: float, lon: float, radius_km: float,
                resolution_m: float, *, timeout: int = 120) -> np.ndarray | None:
    """Окно из WCS 1.0.0 в EPSG:4326: сервер сам пересчитывает сетку под WIDTH/HEIGHT."""
    from rasterio.io import MemoryFile

    if (source.get("protocol") or "").startswith("wcs-2"):
        return _wcs2_window(source, lat, lon, radius_km, resolution_m)

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


KEY_FILE = Path.home() / ".config" / "vector-geo-site" / "opentopography.key"


def _gedtm30_key(source: dict) -> str | None:
    """Ключ OpenTopography: переменная окружения, иначе файл `~/.config/.../opentopography.key`.

    Файл нужен, чтобы пакет собирался без `export` в каждой оболочке; права 600, в репозиторий
    он не попадает (секрет живёт вне дерева).
    """
    key = os.environ.get(source.get("key_env") or "OPENTOPOGRAPHY_API_KEY")
    if key:
        return key.strip()
    try:
        return KEY_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _ot_window(source: dict, lat: float, lon: float, radius_km: float,
               resolution_m: float, *, timeout: int = 300) -> dict | None:
    """GEDTM30 через OpenTopography GlobalDEM API: один запрос по bbox, ответ — GeoTIFF 4326.

    Почему API, а не тайлы: домен `data.opengeohub.org` с тайлами GEDTM30 не резолвится
    (DNS), на Zenodo лежит только 240-метровая версия. Тот же 30-метровый продукт отдаёт
    OpenTopography по бесплатному ключу; проверено 26.11.2026: окно 0.17°×0.15° — 1.0 МБ
    за 12.5 с, 0.41°×0.37° (20 км) — 6.4 МБ за 14.4 с, Сува (Фиджи) — 1.9 МБ за 12.4 с.

    Предел площади на стороне сервиса не мешает: 2°×2° отдаются (153 МБ за 42.7 с), но нам
    столько не нужно — радиус ограничен 32 км, иначе это уже не ближний рельеф.
    """
    import urllib.parse
    import urllib.request

    from rasterio.io import MemoryFile

    if offline():
        raise RuntimeError("offline: сеть запрещена")
    if radius_km > 32.0:
        raise RuntimeError(f"gedtm30: радиус {radius_km:.0f} км больше предела 32 км — "
                           "рельеф ближнего кольца, не региональный")
    key = _gedtm30_key(source)
    if not key:
        raise RuntimeError(f"gedtm30: нет ключа OpenTopography "
                           f"({source.get('key_env') or 'OPENTOPOGRAPHY_API_KEY'})")
    # окно берём по диагонали квадрата пакета: сетка в UTM, а bbox в 4326, и углы 16×16 км
    # иначе выходят за запрошенное (у Лондона 24.5 % пустот — это наши углы, а не пробелы
    # источника). Пустота обязана значить «не измерено», а не «не спросили».
    half = radius_km * float(source.get("window_factor") or 1.45)
    dlat = half / 111.32
    dlon = half / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    south, north = round(lat - dlat, 4), round(lat + dlat, 4)
    west, east = round(lon - dlon, 4), round(lon + dlon, 4)
    cache = CACHE_DIR / f"gedtm30_{south}_{north}_{west}_{east}.tif"
    if cache.exists():
        payload = cache.read_bytes()
    else:
        query = urllib.parse.urlencode({"demtype": source.get("demtype") or "GEDTM30",
                                        "south": south, "north": north,
                                        "west": west, "east": east,
                                        "outputFormat": "GTiff", "API_Key": key})
        url = f"{source['url']}?{query}"
        # S5: бесплатный ключ ограничен 50 запросами в сутки (200 для академических).
        # Один 429 не должен ронять пакет: пауза, повтор, и только потом отказ с причиной,
        # по которой вызывающий код перейдёт на DSM с предупреждением `dtm_quota`.
        payload = b""
        for attempt in range(2):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    payload = resp.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429:
                    raise
                if attempt == 0:
                    time.sleep(5.0)
                    continue
                raise RuntimeError(
                    "dtm_quota: OpenTopography ограничил частоту запросов (бесплатный ключ — "
                    "50 запросов в сутки, академический — 200). Окно уже в кеше? "
                    "См. docs/calc/dtm.md, раздел «Квоты и условия»") from exc
        if payload[:2] == b"<?" or payload[:1] == b"<":       # сервис вернул XML с ошибкой
            raise RuntimeError("gedtm30: сервер вернул текст ошибки вместо GeoTIFF")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
    with MemoryFile(payload) as mem, mem.open() as ds:
        band = ds.read(1)
        transform = tuple(ds.transform)[:6]
        crs = str(ds.crs)
    if not np.isfinite(_mask_heights(band)).any():
        return None
    return {"array": _mask_heights(band), "transform": transform, "crs": crs}


def _gedtm30_cog(source: dict, lat: float, lon: float, radius_km: float,
                 resolution_m: float, *, timeout: int = 180) -> dict | None:
    """Резервный путь: COG-тайлы OpenGeoHub, если `GEOSITE_GEDTM30_URL` всё-таки отвечает."""
    import rasterio
    from rasterio.windows import from_bounds

    if offline():
        raise RuntimeError("offline: сеть запрещена")
    base = os.environ["GEOSITE_GEDTM30_URL"]
    pattern = source.get("tile_pattern") or "GEDTM30_v1.2.0/{name}.tif"
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    parts = []
    for la in range(int(math.floor(lat - dlat)), int(math.floor(lat + dlat)) + 1):
        for lo in range(int(math.floor(lon - dlon)), int(math.floor(lon + dlon)) + 1):
            url = f"{base}/{pattern.format(name=_tile_name(la + 0.5, lo + 0.5))}"
            try:
                with rasterio.open(url) as ds:
                    window = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat,
                                         ds.transform)
                    parts.append((ds.read(1, window=window, boundless=True),
                                  tuple(ds.window_transform(window))[:6], str(ds.crs)))
            except Exception:
                continue
    if not parts:
        return None
    band, transform, crs = parts[0]
    return {"array": _mask_heights(band), "transform": transform, "crs": crs}


def _gedtm30_window(source: dict, lat: float, lon: float, radius_km: float,
                    resolution_m: float, *, timeout: int = 300) -> dict | None:
    """GEDTM30: API OpenTopography по умолчанию, COG-тайлы — только по явному адресу."""
    if os.environ.get("GEOSITE_GEDTM30_URL"):
        return _gedtm30_cog(source, lat, lon, radius_km, resolution_m, timeout=timeout)
    return _ot_window(source, lat, lon, radius_km, resolution_m, timeout=timeout)


def grid(lat: float, lon: float, *, crs: int | str, radius_km: float, resolution_m: float = 30.0
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
                                         lat=lat, lon=lon, radius_km=radius_km, crs=crs,
                                         resolution_m=resolution_m, resampling=dem.Resampling.bilinear)
        grid_obj.source = source["id"]
        grid_obj.label = source["name"]
        grid_obj.attribution = source.get("license") or ""
        grid_obj.kind = "dtm"
        grid_obj.vertical_datum = source.get("vertical_datum")
        grid_obj.datum_shift_m = source.get("datum_shift_m")
        grid_obj.license = source.get("license")
        grid_obj.resolution_m = float(resolution_m)          # сетка пакета, не родная сетка источника
        grid_obj.source_resolution_m = float(source.get("resolution_m") or resolution_m)
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
            "source_resolution_m": info.get("source_resolution_m"),
            "scope": info.get("scope") or "all",
            "license": info.get("license"), "vertical_datum": info.get("vertical_datum"),
            "datum_shift_m": info.get("datum_shift_m")}


__all__ = ["ALGO_VERSION", "MODES", "REGISTRY", "mode", "registry", "lidar_for", "grid",
           "file_block"]
