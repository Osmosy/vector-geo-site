"""Солнце: нейтральные факты для пакета (H2).

Никакой школы: полуденная высота и азимут в дни солнцестояний и равноденствия, сторона, где
солнце стоит в полдень большую часть года, сколько дней в году оно с севера и сколько с юга,
и часы прямого солнца по восьми секторам с учётом рельефа.

Формула NOAA (Spencer 1971 для склонения и уравнения времени) считается без сети и без
зависимостей. Южное полушарие ничего не меняет в формулах — меняется только ответ.

`days_noon_north` и `days_noon_south` — замена привязки к «±1° от экватора»: у Сингапура (1.35°
с. ш.) солнце в полдень с севера примерно 192 дня, с юга 173, у Сиднея (−33.9°) — 0 и 365.
Порог «различие школ слабее» толкователь ставит по этим числам, а не по условной широте.
"""

from __future__ import annotations

import datetime as dt
import math

from . import geo

ALGO_VERSION = "1.0.0"
SOURCE = "NOAA Solar Calculator (формула, без сети)"

SIDES8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
SIDE8_RU = {"N": "север", "NE": "северо-восток", "E": "восток", "SE": "юго-восток",
            "S": "юг", "SW": "юго-запад", "W": "запад", "NW": "северо-запад"}
ILLUM_STEP_MIN = 20      # шаг перебора суток: 20 минут — 26 тыс. точек на год
HORIZON_BINS = 72        # профиль горизонта: шаг 5°


def _declination_eot(day_of_year: int, hour_utc: float = 12.0) -> tuple[float, float]:
    """Склонение (градусы) и уравнение времени (минуты) по NOAA."""
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour_utc - 12.0) / 24.0)
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    eot = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                    - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    return math.degrees(decl), eot


def position(lat: float, lon: float, day_of_year: int, hour_utc: float) -> tuple[float, float]:
    """Высота и азимут солнца (градусы; азимут от истинного севера по часовой)."""
    decl, eot = _declination_eot(day_of_year, hour_utc)
    true_solar_min = hour_utc * 60.0 + eot + 4.0 * lon
    hour_angle = true_solar_min / 4.0 - 180.0
    phi = math.radians(lat)
    delta = math.radians(decl)
    ha = math.radians(hour_angle)
    cos_zenith = (math.sin(phi) * math.sin(delta)
                  + math.cos(phi) * math.cos(delta) * math.cos(ha))
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    altitude = 90.0 - math.degrees(math.acos(cos_zenith))
    # азимут по NOAA: от севера, по часовой
    azimuth = math.degrees(math.atan2(math.sin(ha),
                                      math.cos(ha) * math.sin(phi) - math.tan(delta) * math.cos(phi)))
    azimuth = (azimuth + 180.0) % 360.0
    return altitude, azimuth


def solar_noon_hour(lat: float, lon: float, day_of_year: int) -> float:
    """Момент истинного полудня в часах UTC (широта в формуле не участвует)."""
    _, eot = _declination_eot(day_of_year, 12.0)
    return (720.0 - 4.0 * lon - eot) / 60.0


def side8(azimuth_deg: float) -> str:
    index = int(((azimuth_deg % 360.0) + 22.5) // 45.0) % 8
    return SIDES8[index]


def _event_days(year: int) -> list[tuple[str, str, int]]:
    """Дни солнцестояний и равноденствия: устойчивые даты, год нужен для подписи."""
    return [("june_solstice", f"{year}-06-21", dt.date(year, 6, 21).timetuple().tm_yday),
            ("december_solstice", f"{year}-12-21", dt.date(year, 12, 21).timetuple().tm_yday),
            ("equinox", f"{year}-03-20", dt.date(year, 3, 20).timetuple().tm_yday)]


def noons(lat: float, lon: float, *, year: int = 2026) -> list[dict]:
    rows = []
    for event, date_text, doy in _event_days(year):
        hour = solar_noon_hour(lat, lon, doy)
        altitude, azimuth = position(lat, lon, doy, hour)
        rows.append({"event": event, "date": date_text,
                     "altitude_deg": round(altitude, 1), "azimuth_deg": round(azimuth, 1),
                     "mountain": geo.mountain_name(azimuth), "side8": side8(azimuth),
                     "utc_hour": round(hour % 24.0, 2)})
    return rows


def noon_side(lat: float, lon: float, *, year: int = 2026) -> dict:
    """Сторона полуденного солнца за год: по 24 горам и по 8 сторонам, плюс счёт дней."""
    north_days = south_days = 0
    by_mountain: dict[str, int] = {name: 0 for name in geo.MOUNTAINS}
    by_side: dict[str, int] = {name: 0 for name in SIDES8}
    for doy in range(1, 366):
        hour = solar_noon_hour(lat, lon, doy)
        _, azimuth = position(lat, lon, doy, hour)
        if math.cos(math.radians(azimuth)) >= 0.0:      # азимут в северной половине
            north_days += 1
        else:
            south_days += 1
        by_mountain[geo.mountain_name(azimuth)] += 1
        by_side[side8(azimuth)] += 1
    mountain = max(by_mountain, key=lambda key: by_mountain[key])
    side = max(by_side, key=lambda key: by_side[key])
    return {"mountain": mountain, "side8": side, "side8_ru": SIDE8_RU[side],
            "days_noon_north": north_days, "days_noon_south": south_days,
            "by_side8": by_side}


def horizon_profile(grid, x0: float, y0: float, z0: float, *,
                    max_km: float = 8.0, bins: int = HORIZON_BINS,
                    step_m: float = 60.0) -> list[float]:
    """Угол горизонта по азимутальным бинам: максимум (высота − высота участка) / расстояние.

    Пустоты DEM (море, край тайла) пропускаются: молчаливая подстановка нуля занизила бы
    горизонт и подарила бы участку лишние часы солнца.
    """
    steps = int(max_km * 1000.0 / step_m)
    out: list[float] = []
    for index in range(bins):
        azimuth = index * 360.0 / bins
        sin_az = math.sin(math.radians(azimuth))
        cos_az = math.cos(math.radians(azimuth))
        best = 0.0
        for k in range(1, steps + 1):
            dist = k * step_m
            z = grid.point_value(x0 + sin_az * dist, y0 + cos_az * dist)
            if z != z:            # NaN: нет данных
                continue
            angle = math.degrees(math.atan2(z - z0, dist))
            if angle > best:
                best = angle
        out.append(round(best, 2))
    return out


def illumination(lat: float, lon: float, profile: list[float] | None, *,
                 step_min: int = ILLUM_STEP_MIN, year: int = 2026) -> dict:
    """Часы прямого солнца за год по 8 секторам с учётом горизонта.

    Считается, в каком секторе стоит солнце, когда оно выше горизонта. Без профиля (нет DEM)
    получается астрономическая освещённость: рельеф не мешает.
    """
    hours = {name: 0.0 for name in SIDES8}
    blocked = 0.0
    total = 0.0
    bins = len(profile) if profile else 0
    step_h = step_min / 60.0
    for doy in range(1, 366):
        for minute in range(0, 1440, step_min):
            altitude, azimuth = position(lat, lon, doy, minute / 60.0)
            if altitude <= 0.0:
                continue
            total += step_h
            if profile:
                angle = profile[int(azimuth % 360.0 * bins / 360.0) % bins]
                if altitude <= angle:
                    blocked += step_h
                    continue
            hours[side8(azimuth)] += step_h
    return {"dem_layer": "dsm_smoothed" if profile else None,
            "hours_by_side": {name: round(value, 1) for name, value in hours.items()},
            "total_hours": round(total, 1), "blocked_hours": round(blocked, 1),
            "step_min": step_min,
            "source": SOURCE}


def block(lat: float, lon: float, *, grid=None, x0: float | None = None,
          y0: float | None = None, z0: float | None = None, max_km: float = 8.0,
          year: int = 2026) -> dict:
    """Блок `sun` пакета: полудни, сторона полудня, счёт дней, освещённость."""
    profile = None
    if grid is not None and x0 is not None and y0 is not None and z0 is not None:
        try:
            profile = horizon_profile(grid, x0, y0, z0, max_km=max_km)
        except Exception:
            profile = None
    side = noon_side(lat, lon, year=year)
    return {"source": SOURCE,
            "latitude_abs": round(abs(lat), 3),
            "noons": noons(lat, lon, year=year),
            "noon_side": {key: side[key] for key in ("mountain", "side8", "side8_ru")},
            "days_noon_north": side["days_noon_north"],
            "days_noon_south": side["days_noon_south"],
            "by_side8": side["by_side8"],
            "illumination": illumination(lat, lon, profile)}
