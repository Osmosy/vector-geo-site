"""Геометрия гор и азимутов: 24 горы по 15°, сетка координат, зоны UTM.

Сетка выбирается по точке и радиусу (`grid_crs`): UTM, пока участок мал и не пересекает границу
зоны; азимутальная равнопромежуточная проекция с центром в участке (`aeqd`) — для дальних
охватов, приграничных и полярных точек. В `aeqd` азимут луча из центра равен истинному, и 24
горы считаются по нему без поправок.
"""

from __future__ import annotations

import math

# Порядок 24 гор по часовой стрелке от севера; 子 центрирована на 0°.
MOUNTAINS: list[str] = [
    "子", "癸", "丑", "艮", "寅", "甲",
    "卯", "乙", "辰", "巽", "巳", "丙",
    "午", "丁", "未", "坤", "申", "庚",
    "酉", "辛", "戌", "乾", "亥", "壬",
]

SECTOR_DEG = 15.0
HALF_SECTOR = SECTOR_DEG / 2


def normalize_deg(deg: float) -> float:
    return deg % 360.0


def mountain_index(bearing: float) -> int:
    """Индекс горы, в сектор которой попадает азимут (истинный, от севера)."""
    return int(((normalize_deg(bearing) + HALF_SECTOR) % 360.0) // SECTOR_DEG)


def mountain_name(bearing: float) -> str:
    return MOUNTAINS[mountain_index(bearing)]


def mountain_from_range_deg(lo: float, hi: float) -> str:
    return mountain_name((lo + hi) / 2.0)


def mountain_bounds(name: str) -> tuple[float, float]:
    """Границы сектора горы: (от, до) в градусах; для 子 — 352.5 … 7.5."""
    idx = MOUNTAINS.index(name)
    center = idx * SECTOR_DEG
    lo = normalize_deg(center - HALF_SECTOR)
    hi = normalize_deg(center + HALF_SECTOR)
    return lo, hi


def angle_distance(a: float, b: float) -> float:
    """Минимальная угловая разница 0…180°."""
    d = abs(normalize_deg(a) - normalize_deg(b)) % 360.0
    return d if d <= 180.0 else 360.0 - d


def near_boundary(bearing: float, margin_deg: float = 3.0) -> bool:
    """Близко ли азимут к границе сектора (риск разных школ)."""
    idx = mountain_index(bearing)
    center = idx * SECTOR_DEG
    return abs(normalize_deg(bearing) - center) > (HALF_SECTOR - margin_deg)


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Начальный азимут из точки 1 в точку 2, градусы от севера."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return normalize_deg(math.degrees(math.atan2(y, x)))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


ALGO_VERSION = "1.1.0"

AEQD = "+proj=aeqd +lat_0={lat:.6f} +lon_0={lon:.6f} +datum=WGS84 +units=m +no_defs"

POLAR_NORTH = 84.0
POLAR_SOUTH = -80.0
RADIUS_AEQD_KM = 50.0
ZONE_WIDTH_DEG = 6.0


def crs_str(value) -> str:
    """CRS в виде строки для pyproj/rasterio.

    Код UTM — «EPSG:32630»; PROJ-строка (`+proj=aeqd …`) и WKT (`PROJCS["…"]`, каким его отдаёт
    rasterio) идут **как есть**: приписав к WKT «EPSG:», получаем `CRSError: crs not found`,
    и пакет падает на пустом месте (E-035).
    """
    if isinstance(value, int):
        return f"EPSG:{value}"
    text = str(value).strip()
    if text.upper().startswith("EPSG:") or text.startswith("+"):
        return text
    if text.split("[", 1)[0].upper() in {"PROJCS", "GEOGCS", "PROJCRS", "GEOGCRS", "BOUNDCRS",
                                          "COMPOUNDCRS", "COMPD_CS", "LOCAL_CS", "VERT_CS"}:
        return text
    return f"EPSG:{text}"


def utm_zone(lon: float) -> int:
    return min(max(int((normalize_deg(lon + 180.0)) // ZONE_WIDTH_DEG) + 1, 1), 60)


def zone_bounds(zone: int) -> tuple[float, float]:
    """Долготы зоны UTM: (западная, восточная)."""
    west = normalize_deg((zone - 1) * ZONE_WIDTH_DEG - 180.0)
    return west, normalize_deg(west + ZONE_WIDTH_DEG)


def polar(lat: float) -> bool:
    """Выше 84° с. ш. и ниже 80° ю. ш. — только азимутальная проекция."""
    return lat > POLAR_NORTH or lat < POLAR_SOUTH


def span_lon(lat: float, radius_km: float) -> float:
    """Сколько градусов долготы занимает круг радиуса: у полюса — почти вся параллель."""
    cos = max(math.cos(math.radians(min(abs(lat), 89.9))), 1e-6)
    return 2.0 * radius_km / (111.32 * cos)


def crosses_antimeridian(lon: float, radius_km: float, lat: float = 0.0) -> bool:
    """Круг переходит через 180-й меридиан: запросы и bbox придётся делить на две части."""
    half = span_lon(lat, radius_km) / 2.0
    if half >= 180.0:
        return True
    west, east = lon - half, lon + half
    return west < -180.0 or east > 180.0


def crosses_zone_boundary(lon: float, radius_km: float, lat: float = 0.0) -> bool:
    """Круг задевает границу зоны UTM: тогда UTM даёт разрыв, и нужна азимутальная."""
    half = span_lon(lat, radius_km) / 2.0
    if half >= ZONE_WIDTH_DEG:
        return True
    west = normalize_deg(lon - half)
    east = normalize_deg(lon + half)
    zone = utm_zone(lon)
    bounds = zone_bounds(zone)
    for edge in (bounds[0], bounds[1]):
        if (normalize_deg(edge - west) % 360.0) <= (normalize_deg(east - west) % 360.0):
            return True
    return False


def grid_crs(lat: float, lon: float, radius_km: float) -> dict:
    """Какую сетку брать для участка: UTM или азимутальную с центром в участке.

    Правило из ТЗ: азимутальная — при радиусе больше 50 км, при пересечении границы зоны UTM,
    на полюсах и на линии перемены дат. Причина пишется в пакет: сетка координат — часть метода,
    а не деталь реализации.
    """
    reason = None
    if polar(lat):
        reason = "polar"
    elif radius_km > RADIUS_AEQD_KM:
        reason = "radius>50km"
    elif crosses_antimeridian(lon, radius_km, lat):
        reason = "antimeridian"
    elif crosses_zone_boundary(lon, radius_km, lat):
        reason = "zone_boundary"
    if reason is None:
        return {"kind": "utm", "crs": crs_str(utm_epsg(lat, lon)), "reason": None,
                "utm_epsg": utm_epsg(lat, lon)}
    return {"kind": "aeqd", "crs": AEQD.format(lat=lat, lon=lon), "reason": reason,
            "utm_epsg": utm_epsg(lat, lon)}


def convergence_deg(lat: float, lon: float, crs) -> float:
    """Сходимость меридианов: насколько «сеточный север» повёрнут от истинного.

    В UTM оси сетки повёрнуты к центральному меридиану зоны: у Лондона (3° от меридиана зоны,
    51.5° с. ш.) — на 2.27°. Без этой поправки все секторы и стороны повёрнуты на два градуса,
    то есть на седьмую часть горы. В азимутальной проекции с центром в участке север истинный,
    и поправка равна нулю.
    """
    text = crs_str(crs)
    if text.startswith("+"):
        return 0.0
    zone = utm_zone(lon)
    central = (zone - 1) * ZONE_WIDTH_DEG - 180.0 + ZONE_WIDTH_DEG / 2.0
    dlon = ((lon - central + 180.0) % 360.0) - 180.0
    return dlon * math.sin(math.radians(lat))


def azimuth_xy(x: float, y: float, offset_deg: float = 0.0) -> float:
    """Истинный азимут по координатам сетки: плоскостной угол плюс сходимость меридианов."""
    return normalize_deg(math.degrees(math.atan2(x, y)) + offset_deg)


def offset_for_grid(grid, x0: float, y0: float) -> float:
    """Поправка для готовой сетки: по её CRS и точке участка."""
    from pyproj import Transformer

    text = crs_str(getattr(grid, "crs", "EPSG:4326"))
    if text.startswith("+"):
        return 0.0
    inv = Transformer.from_crs(text, "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(x0, y0)
    return convergence_deg(lat, lon, text)


def split_bbox(west: float, east: float) -> list[tuple[float, float]]:
    """bbox, пересекающий 180-й меридиан, делится на две части — иначе запрос пуст."""
    if west <= east and west >= -180.0 and east <= 180.0:
        return [(west, east)]
    if west < -180.0:
        return [(-180.0, east), (west + 360.0, 180.0)]
    if east > 180.0:
        return [(west, 180.0), (-180.0, east - 360.0)]
    return [(west, east)]


def azimuth_geodesic(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Начальный азимут по геодезической линии (эллипсоид WGS84), градусы от севера."""
    from pyproj import Geod

    forward, _, _ = Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
    return normalize_deg(forward)


def utm_epsg(lat: float, lon: float) -> int:
    return (32600 if lat >= 0 else 32700) + utm_zone(lon)
