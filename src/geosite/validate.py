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

# P0 (I-11): версии модулей, с которыми сверяется пакет.
from . import climate as _climate, geocode as _geocode, sun as _sun, terrain as _terrain
from . import water as _water
CODE_VERSIONS = {
    "water": _water.ALGO_VERSION,
    "terrain": _terrain.ALGO_VERSION,
    "geocode": _geocode.ALGO_VERSION,
    "climate": _climate.ALGO_VERSION,
    "sun": _sun.ALGO_VERSION,
}

# I-10: поля с эвристикой обязаны нести парную уверенность. Список пополняется на каждом этапе.
CONFIDENCE_PAIRS = {
    "flow_chosen": "flow_confidence",
    "water_mouth.closure_index": "water_mouth.closure_confidence",
    "water_mouth.enclosure": "water_mouth.closure_confidence",
}
LOOP_MIN_RATIO = 3.0
LOOP_MIN_TURN_DEG = 180.0
LOOP_MIN_AREA_K = 0.05
# I-07a/I-07b (E-020): нарушение — только длинный кусок с пологим входом; короче или
# касательнее — предупреждение с причиной.
I07_MIN_LENGTH_M = 500.0
I07_MIN_ANGLE_DEG = 30.0


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
    # H4: новые имена; старые читаются — пакеты до 2.0.0 не должны «ломаться» валидатором
    for key in ("back", "front", "left", "right", "tortoise", "bird", "dragon", "tiger"):
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
            elif (loop.get("turn_deg") is not None and loop["turn_deg"] < LOOP_MIN_TURN_DEG
                  and (loop.get("area_k") or 0) < LOOP_MIN_AREA_K):
                out.append(_violation("I-05", f"{base}.loop",
                                      f"петля без поворота {LOOP_MIN_TURN_DEG}° и площади "
                                      f"{LOOP_MIN_AREA_K}·gap²", id=wid))
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

        # I-07a: видимая проходящая река не может входить и выходить из одной горы.
        # Разделение на I-07a/I-07b (E-020): нарушение — только длинный кусок с пологим
        # входом; короткий кусок, касательная и петля получают причину и предупреждение,
        # а не нарушение. Старые пакеты без `length_2km_m` проверяются по числу пересечений.
        length = _num(water.get("length_2km_m"))
        if length is not None:
            long_enough = length >= I07_MIN_LENGTH_M
        else:
            long_enough = (water.get("crossings_2km") or 0) >= 2
        if (nearest is not None and nearest < 3000
                and water.get("geom", "line") == "line"
                and water.get("visible") is not False
                and not water.get("hidden_share")
                and not water.get("loop")
                and long_enough
                and (water.get("crossing_angle_deg") or 0) >= I07_MIN_ANGLE_DEG):
            if water.get("in_mountain") and water.get("in_mountain") == water.get("out_mountain"):
                out.append(_violation("I-07", f"{base}.in_mountain",
                                      "вход и выход в одной горе у видимой проходящей реки", id=wid))

        # I-08: выбор направления и правило
        if water.get("flow_chosen") and water.get("flow_rule") is None:
            out.append(_violation("I-08", f"{base}.flow_rule", "flow_chosen без flow_rule", id=wid))

        # I-10 (R1): эвристическое поле без парного confidence — нарушение
        for field, conf in CONFIDENCE_PAIRS.items():
            if "." in field:
                head, tail = field.split(".", 1)
                block = water.get(head) or {}
                present = tail in block
                has_conf = conf.split(".", 1)[1] in block
            else:
                present = field in water
                has_conf = conf in water
            if present and not has_conf:
                out.append(_violation("I-10", f"{base}.{field}",
                                      f"нет парного поля уверенности {conf.split('.')[-1]}", id=wid))

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
    # P0: версия в пакете обязана совпадать с версией кода, иначе пакет собран старым расчётом.
    # Сравниваем только то, что пакет записал: у пакетов до 1.7.0 нет climate и geocode.
    for module, version in CODE_VERSIONS.items():
        got = algo.get(module)
        if got and got != version:
            out.append(_violation("I-11", f"algo.{module}",
                                  f"в пакете {got}, в коде {version} — пакет собран другим расчётом"))

    return out


def validate_file(path: str | Path, **kwargs) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate(data, **kwargs)


def warnings(pack: dict) -> list[str]:
    """I-07b (E-020): объяснимые совпадения гор входа и выхода — предупреждения, не нарушения.

    Причина пишется в самом объекте (`same_mountain_reason`), а здесь она превращается в строку
    `quality.warnings`, чтобы читатель отчёта видел исключение и его основание, а не молчание.
    """
    out: list[str] = []
    for water in pack.get("water") or []:
        reason = water.get("same_mountain_reason")
        if not reason or water.get("geom", "line") != "line":
            continue
        if reason == "вне круга выборки":
            # Это не исключение, а граница замера: русла в круге 2 км нет вовсе. Пишем в
            # объекте, но в предупреждения не тащим — иначе десяток далёких каналов
            # заполняет список тем, что про них ничего не считалось.
            continue
        nearest = _num(water.get("nearest_m"))
        if nearest is None or nearest >= 3000:
            continue
        out.append(f"I-07b: {water.get('id') or '?'} — вход и выход в одной горе ({reason})")
    return out


def apply_to_pack(pack: dict, violations: list[dict]) -> dict:
    """Записать результат проверки в quality пакета (используется в geosite pack)."""
    quality = pack.setdefault("quality", {})
    quality["valid"] = not violations
    quality["violations"] = violations
    warns = warnings(pack)
    if warns:
        existing = quality.setdefault("warnings", [])
        for item in warns:
            if item not in existing:
                existing.append(item)
    return pack
