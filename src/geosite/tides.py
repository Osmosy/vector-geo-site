"""Приливность: тег OSM, иначе таблица известных границ прилива.

Таблица пополняется вручную и обязана ссылаться на источник. Если ни тега, ни записи нет —
`tidal: null`: предполагать нельзя (П1).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "tidal_limits.json"

# Известные границы прилива: ниже по течению от точки — приливное влияние.
DEFAULTS: list[dict] = [
    {
        "river_name_contains": "Thames",
        "limit_name": "Teddington Lock",
        "limit_lat": 51.4310,
        "limit_lon": -0.3080,
        "tidal_side": "east",
        "source": "Port of London Authority: приливная граница Темзы — шлюз Теддингтон",
    }
]


def load_limits() -> list[dict]:
    if DATA.exists():
        try:
            data = json.loads(DATA.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except json.JSONDecodeError:
            pass
    return DEFAULTS


def from_tag(tags: dict) -> tuple[bool | None, str | None]:
    """Приливность из тега OSM: tidal=yes|no."""
    raw = (tags.get("tidal") or "").lower()
    if raw in {"yes", "true", "1"}:
        return True, "osm_tag"
    if raw in {"no", "false", "0"}:
        return False, "osm_tag"
    # tidal=... бывает и у приливных водотоков в других формах — не гадаем
    return None, None


def limit_entry(name: str | None) -> dict | None:
    """Запись таблицы для реки (по вхождению имени), без решения о приливности."""
    if not name:
        return None
    for entry in load_limits():
        needle = entry.get("river_name_contains", "")
        if needle and needle.lower() in name.lower():
            return entry
    return None


def status_by_distance(entry: dict, along_site_m: float | None, along_limit_m: float | None,
                       flow_toward_end: bool) -> dict:
    """Приливность по положению вдоль русла относительно границы прилива.

    Сравнение по долготе ошибается на изгибах (Кингстон западнее Теддингтона, но выше него).
    Поэтому берём расстояние вдоль русла: ниже по течению от границы — прилив.
    """
    if along_site_m is None or along_limit_m is None:
        return {"tidal": None, "tidal_source": None,
                "note": "граница прилива известна, но точку не удалось поставить на русло"}
    downstream = along_site_m > along_limit_m if flow_toward_end else along_site_m < along_limit_m
    return {"tidal": bool(downstream), "tidal_source": "known_limit",
            "note": f"граница прилива: {entry.get('limit_name')} ({entry.get('source', '')})".strip()}


def from_limits(name: str | None, lon: float, lat: float) -> tuple[bool | None, str | None, str | None]:
    """Приливность по таблице: сравнение с известной границей по долготе/широте."""
    if not name:
        return None, None, None
    for entry in load_limits():
        needle = entry.get("river_name_contains", "")
        if needle and needle.lower() not in name.lower():
            continue
        limit_lon = float(entry["limit_lon"])
        side = entry.get("tidal_side", "east")
        if side == "east":
            tidal = lon > limit_lon
        elif side == "west":
            tidal = lon < limit_lon
        elif side == "north":
            tidal = lat > float(entry["limit_lat"])
        else:
            tidal = lat < float(entry["limit_lat"])
        note = f"граница прилива: {entry.get('limit_name')} ({entry.get('source', '')})".strip()
        return bool(tidal), "known_limit", note
    return None, None, None


def status(tags: dict, name: str | None, lon: float, lat: float) -> dict:
    """Итоговый статус приливности с источником и причиной."""
    tidal, source = from_tag(tags)
    if tidal is not None:
        return {"tidal": tidal, "tidal_source": source, "note": None}
    # положение относительно границы решается по руслу (water.py), а не по координате:
    # сравнение долгот ошибается на изгибах (Кингстон западнее Теддингтона, но выше него)
    return {"tidal": None, "tidal_source": None, "note": "нет тега, проверяем по таблице вдоль русла"}
