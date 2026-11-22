"""Геометрия гор и азимутов: 24 горы по 15°, склонение, зоны UTM."""

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


def utm_epsg(lat: float, lon: float) -> int:
    zone = int((normalize_deg(lon + 180.0)) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone
