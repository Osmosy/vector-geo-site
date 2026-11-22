"""Инварианты пакета участка: `site_pack/1` должен быть внутренне согласован.

Правило модуля: проверки не чинят данные, а называют нарушение кодом. `pack` пишет
результат в `quality.valid` / `quality.violations`, CLI `geosite validate` падает с кодом 1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import geo

MOUNTAINS = set(geo.MOUNTAINS)

# Коды нарушений из ТЗ (часть Б1). Расширять только добавлением.
CODES = {
    "I-01": "горы и азимуты",
    "I-02": "расстояния",
    "I-03": "индексы и доли",
    "I-04": "closure_index при упёртом замере",
    "I-05": "петля без критерия отношения",
    "I-06": "дубли слияний",
    "I-07": "вход и выход воды в одной горе",
    "I-08": "выбор направления течения без правила",
    "I-09": "высоты вне диапазона",
    "I-10": "низкая уверенность без предупреждения",
    "I-11": "версии алгоритмов",
    "I-12": "ноль при флаге предела",
}

MIN_ELEV, MAX_ELEV = -430.0, 8850.0
LOOP_MIN_RATIO = 3.0


def _violation(code: str, path: str, message: str, **extra: Any) -> dict:
    item = {"code": code, "path": path, "message": message}
    item.update({k: v for k, v in extra.items() if v is not None})
    return item


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)
    return None


def validate(pack: dict, *, dem_point: float | None = None, radius_m: float = 20000.0) -> list[dict]:
    """Полный прогон инвариантов. Возвращает список нарушений (пусто — пакет годен)."""
    out: list[dict] = []

    # I-01 … I-03 по секторам
    for i, sector in enumerate(pack.get("sectors") or []):
        base = f"sectors[{i}]"
        name = sector.get("mountain")
        if name not in MOUNTAINS:
            out.append(_violation("I-01", f"{base}.mountain", f"недопустимая гора: {name!r}"))
        for key in ("from_deg", "to_deg"):
            deg = _num(sector.get(key))
            if deg is None or not (0.0 <= deg < 360.0):
                out.append(_violation("I-01", f"{base}.{key}", f"азимут вне [0,360): {sector.get(key)!r}"))
        for j, ring in enumerate(sector.get("rings") or []):
            rbase = f"{base}.rings[{j}]"
            for key in ("max_rel_elev_m", "mean_rel_elev_m"):
                val = _num(ring.get(key))
                if val is not None and not (MIN_ELEV <= val <= MAX_ELEV):
                    out.append(_violation("I-09", f"{rbase}.{key}", f"высота вне диапазона: {val}"))
            for key in ("water_share", "hill_share"):
                val = _num(ring.get(key))
                if val is not None and not (0.0 <= val <= 1.0):
                    out.append(_violation("I-03", f"{rbase}.{key}", f"доля вне [0,1]: {val}"))

    # высота участка
    elev = _num((pack.get("site") or {}).get("elev_m"))
    if elev is not None and not (MIN_ELEV <= elev <= MAX_ELEV):
        out.append(_violation("I-09", "site.elev_m", f"высота вне диапазона: {elev}"))
    if elev is not None and dem_point is not None and abs(elev - dem_point) > 1.0:
        out.append(_violation("I-09", "site.elev_m", f"высота {elev} не совпадает с DEM {dem_point:.1f}"))

    # I-03 по духам
    spirits = pack.get("four_spirits") or {}
    for key in ("tortoise", "bird", "dragon", "tiger"):
        item = spirits.get(key) or {}
        val = _num(item.get("continuity"))
        if val is not None and not (0.0 <= val <= 1.0):
            out.append(_violation("I-03", f"four_spirits.{key}.continuity", f"доля вне [0,1]: {val}"))

    # вода
    for i, water in enumerate(pack.get("water") or []):
        base = f"water[{i}]"
        wid = water.get("id", "?")
        for key in ("in_mountain", "out_mountain"):
            name = water.get(key)
            if name is not None and name not in MOUNTAINS:
                out.append(_violation("I-01", f"{base}.{key}", f"недопустимая гора: {name!r}", id=wid))
        nearest = _num(water.get("nearest_m"))
        if nearest is not None and nearest < 0:
            out.append(_violation("I-02", f"{base}.nearest_m", f"отрицательное расстояние: {nearest}", id=wid))
        if nearest is not None and nearest > radius_m:
            out.append(_violation("I-02", f"{base}.nearest_m", f"дальше радиуса выборки: {nearest}", id=wid))

        # I-05: петля обязана иметь отношение и порог
        loop = water.get("loop")
        if loop:
            ratio = _num(loop.get("ratio"))
            if ratio is None:
                out.append(_violation("I-05", f"{base}.loop.ratio", "петля без отношения", id=wid))
            elif ratio < LOOP_MIN_RATIO:
                out.append(_violation("I-05", f"{base}.loop.ratio", f"отношение {ratio} < {LOOP_MIN_RATIO}", id=wid))

        # I-06: дубли слияний одной пары
        seen: list[tuple[str, float]] = []
        for ci, confluence in enumerate(water.get("confluences") or []):
            other = str(confluence.get("with"))
            dist = _num(confluence.get("dist_m")) or 0.0
            for prev_other, prev_dist in seen:
                if other == prev_other and abs(dist - prev_dist) < 50.0:
                    out.append(_violation("I-06", f"{base}.confluences[{ci}]",
                                          f"дубль слияния с {other} на {dist} м", id=wid))
            seen.append((other, dist))

        # I-07: река мимо участка не может входить и выходить из одной горы
        if (water.get("kind") == "river" and nearest is not None and nearest < 3000
                and (water.get("crossings_2km") or 0) >= 2
                and (water.get("crossing_angle_deg") or 0) >= 30.0):
            if water.get("in_mountain") and water.get("in_mountain") == water.get("out_mountain"):
                out.append(_violation("I-07", f"{base}.in_mountain",
                                      "вход и выход в одной горе у проходящей реки", id=wid))

        # I-08: выбор направления и правило
        if water.get("flow_chosen") and water.get("flow_rule") is None:
            out.append(_violation("I-08", f"{base}.flow_rule", "flow_chosen без flow_rule", id=wid))

        # I-10: низкая уверенность должна звучать в quality.warnings
        warnings = " ".join((pack.get("quality") or {}).get("warnings") or [])
        if water.get("flow_confidence") == "low" and "water.flow" not in warnings and "flow_conflict" not in warnings:
            out.append(_violation("I-10", f"{base}.flow_confidence",
                                  "low без предупреждения в quality.warnings", id=wid))

        # врата: I-04 и I-12
        mouth = water.get("water_mouth") or {}
        if mouth:
            closure = _num(mouth.get("closure_index"))
            capped = mouth.get("width_capped")
            capped_sides = mouth.get("capped_sides")
            if closure is not None and capped is True and (capped_sides or 0) >= 2:
                out.append(_violation("I-04", f"{base}.water_mouth.closure_index",
                                      "индекс задан, хотя обе стороны упёрлись в предел", id=wid))
            for key in ("closure_index", "width_m", "left_hill_m", "right_hill_m"):
                val = _num(mouth.get(key))
                if val == 0.0 and (capped is True or (capped_sides or 0) > 0):
                    out.append(_violation("I-12", f"{base}.water_mouth.{key}",
                                          f"{key} = 0 при флаге предела", id=wid))
            if closure is not None and not (0.0 <= closure <= 1.0):
                out.append(_violation("I-03", f"{base}.water_mouth.closure_index",
                                      f"индекс вне [0,1]: {closure}", id=wid))

    # I-11: версии алгоритмов
    algo = pack.get("algo") or {}
    if not algo:
        out.append(_violation("I-11", "algo", "версии алгоритмов не записаны"))
    for module in ("water", "terrain"):
        if (pack.get("water") or pack.get("sectors")) and not algo.get(module):
            out.append(_violation("I-11", f"algo.{module}", "версия модуля не записана"))

    return out


def validate_file(path: str | Path, **kwargs) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate(data, **kwargs)


def apply_to_pack(pack: dict, violations: list[dict]) -> dict:
    """Записать результат проверки в quality пакета (используется в geosite pack)."""
    quality = pack.setdefault("quality", {})
    quality["valid"] = not violations
    quality["violations"] = violations
    return pack
