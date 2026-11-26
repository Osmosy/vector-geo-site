"""Этап 4: магнитное склонение и ветер.

Склонение нужно, чтобы перевести компасный азимут в истинный и обратно: карта звёзд строится
по истинному северу, а человек на месте меряет компасом. Считается по модели WMM (pygeomag,
коэффициенты NOAA), без сети.

Ветер — не «роза» из головы, а измеренная статистика: Open-Meteo отдаёт ERA5-архив
(10 м, суточные доминирующее направление и средняя скорость) без ключа. Из суточных векторов
считается средний перенос, повторяемость по 24 горам и сезонное сравнение — зима против лета.
Предупреждение «ветер не рассчитан» остаётся, если архива нет: догадка о ветре хуже пустого поля.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ALGO_VERSION = "1.1.0"

SOURCE_MAG = "WMM (NOAA), pygeomag"
SOURCE_WIND = "ERA5 через Open-Meteo archive API (суточные, 10 м)"

WIND_URL = "https://archive-api.open-meteo.com/v1/archive"
WIND_START = "2016-01-01"
WIND_END = "2025-12-31"


def cache_dir() -> Path:
    path = Path.home() / ".cache" / "vector-geo-site" / "climate"
    path.mkdir(parents=True, exist_ok=True)
    return path


def decimal_year(day: dt.date | None = None) -> float:
    day = day or dt.date.today()
    start = dt.date(day.year, 1, 1)
    end = dt.date(day.year + 1, 1, 1)
    return day.year + (day - start).days / (end - start).days


def declination(lat: float, lon: float, year: float | None = None) -> dict:
    """Магнитное склонение: градусы, восточное положительное."""
    import pygeomag

    when = decimal_year() if year is None else float(year)
    model = pygeomag.GeoMag()
    result = model.calculate(glat=lat, glon=lon, alt=0.0, time=when)
    return {"deg": round(float(result.d), 3), "year": round(when, 2), "source": SOURCE_MAG}


def magnetic_bearing(true_bearing: float, declination_deg: float) -> float:
    """Истинный азимут → компасный: на восточном склонении компас показывает меньше."""
    return round((float(true_bearing) - float(declination_deg)) % 360.0, 1)


def _wind_cache_path(lat: float, lon: float) -> Path:
    key = f"{lat:.3f},{lon:.3f},{WIND_START},{WIND_END}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cache_dir() / f"wind-{digest}.json"


def fetch_wind(lat: float, lon: float, *, allow_network: bool = True) -> dict:
    """Суточный архив ветра: направление (откуда) и средняя скорость за 10 лет."""
    path = _wind_cache_path(lat, lon)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not allow_network:
        raise FileNotFoundError(f"нет кеша ветра: {path}")
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode({
        "latitude": f"{lat:.3f}", "longitude": f"{lon:.3f}",
        "start_date": WIND_START, "end_date": WIND_END,
        "daily": "wind_direction_10m_dominant,wind_speed_10m_mean",
        "timezone": "UTC",
    })
    url = f"{WIND_URL}?{query}"
    request = urllib.request.Request(url, headers={
        "User-Agent": "vector-geo-site/1.0 (fengshui climate; https://github.com/Osmosy/vector-geo-site)"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    daily = payload.get("daily") or {}
    data = {"time": daily.get("time") or [],
            "direction": daily.get("wind_direction_10m_dominant") or [],
            "speed_kmh": daily.get("wind_speed_10m_mean") or [],
            "source": SOURCE_WIND, "elevation_m": payload.get("elevation")}
    if data["time"]:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


COLD_MONTHS = {"N": (12, 1, 2), "S": (6, 7, 8)}
COLD_LABEL = {"N": "декабрь–февраль", "S": "июнь–август"}


def summarize(data: dict, *, calm_kmh: float = 3.0, hemisphere: str | None = None) -> dict:
    """Средний перенос, повторяемость по 24 горам, зима против лета, холодный сезон (H3).

    Холодный сезон выбирается по полушарию — это физика, а не толкование: в Северном это
    декабрь–февраль, в Южном июнь–август. Поле `cold_wind` и есть ответ на «откуда дует
    холодный ветер».
    """
    from . import geo

    days = len(data.get("time") or [])
    if not days:
        raise ValueError("ветер: пустой архив")
    u = v = 0.0
    speeds: list[float] = []
    calms = 0
    by_mountain: dict[str, int] = {name: 0 for name in geo.MOUNTAINS}
    seasonal: dict[str, list[tuple[float, float]]] = {"winter": [], "summer": []}
    cold: list[tuple[float, float]] = []
    for i, day in enumerate(data["time"]):
        direction = data["direction"][i]
        speed = data["speed_kmh"][i]
        if direction is None or speed is None:
            continue
        rad = math.radians(float(direction))
        ms = float(speed) / 3.6
        speeds.append(ms)
        if float(speed) < calm_kmh:
            calms += 1
        # вектор переноса: направление «откуда» → воздух идёт в противоположную сторону
        u += -ms * math.sin(rad)
        v += -ms * math.cos(rad)
        by_mountain[geo.mountain_name(float(direction))] += 1
        month = int(str(day)[5:7])
        if month in (12, 1, 2):
            seasonal["winter"].append((ms, float(direction)))
        elif month in (6, 7, 8):
            seasonal["summer"].append((ms, float(direction)))
        if hemisphere and month in COLD_MONTHS[hemisphere]:
            cold.append((ms, float(direction)))

    total = sum(by_mountain.values()) or 1
    prevailing = math.degrees(math.atan2(-u, -v)) % 360.0 if (u or v) else None
    shares = {k: round(count / total, 3) for k, count in by_mountain.items()}
    return {
        "prevailing_from_deg": None if prevailing is None else round(prevailing, 1),
        "prevailing_from_mountain": None if prevailing is None else geo.mountain_name(prevailing),
        "mean_speed_ms": round(sum(speeds) / len(speeds), 2) if speeds else None,
        "calm_share": round(calms / total, 3),
        "by_mountain": shares,
        "seasonal": {k: _season(v) for k, v in seasonal.items()},
        "cold_wind": _cold(cold, hemisphere, total),
        "days": total,
        "source": data.get("source") or SOURCE_WIND,
    }


def _season(items: list[tuple[float, float]]) -> dict:
    if not items:
        return {"prevailing_from_deg": None, "mean_speed_ms": None, "days": 0}
    u = sum(-ms * math.sin(math.radians(d)) for ms, d in items)
    v = sum(-ms * math.cos(math.radians(d)) for ms, d in items)
    prevailing = math.degrees(math.atan2(-u, -v)) % 360.0
    return {"prevailing_from_deg": round(prevailing, 1),
            "mean_speed_ms": round(sum(ms for ms, _ in items) / len(items), 2),
            "days": len(items)}


def _cold(items: list[tuple[float, float]], hemisphere: str | None, total: int) -> dict | None:
    """Преобладающее направление и скорость в холодный сезон полушария (H3)."""
    if not hemisphere:
        return None
    from . import geo

    season = _season(items)
    return {"season": COLD_LABEL[hemisphere], "months": list(COLD_MONTHS[hemisphere]),
            "prevailing_from_deg": season["prevailing_from_deg"],
            "prevailing_from_mountain": (None if season["prevailing_from_deg"] is None
                                         else geo.mountain_name(season["prevailing_from_deg"])),
            "mean_speed_ms": season["mean_speed_ms"],
            "days": season["days"],
            "share_of_days": round(season["days"] / total, 3) if total else None}


def summary(lat: float, lon: float, *, allow_network: bool = True,
            payload: dict | None = None) -> dict:
    data = payload if payload is not None else fetch_wind(lat, lon, allow_network=allow_network)
    hemisphere = "S" if lat < 0 else "N"
    return summarize(data, hemisphere=hemisphere)
