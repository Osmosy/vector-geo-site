"""Калибровка доли воды: OSM против ESA WorldCover (W3, W4).

Правило выбора значения (в ТЗ названо П4, но не расшифровано — расшифровка ниже и в
`docs/calc/water_share.md`): **основа — OSM**. WorldCover заменяет её только в одном случае —
когда OSM воды не видит вовсе (`osm < 0.5 %`, а по снимкам `≥ 1 %`): так выглядит река,
нарисованная осевой линией. Во всех остальных расхождениях остаётся OSM, а уверенность
понижается и пишется причина: снимок прилива показывает отливное дно, тень дома — воду,
мост режет контур на куски.

Подгонять значение под снимок запрещено: расхождение — факт о данных, а не ошибка, которую
надо замазать.
"""

from __future__ import annotations

ALGO_VERSION = "1.0.0"

# Допуски W3, в долях (1 п.п. кольца, 3 п.п. сектора)
RING_TOLERANCE = 0.01
SECTOR_TOLERANCE = 0.03

# Коды причин — закрытый список W3, значения в пакете пишутся кодом, расшифровка рядом
REASONS: dict[str, str] = {
    "axis_line": "осевая река без поверхности",
    "bridge": "мост",
    "dock_lock": "док за шлюзом",
    "tidal_flat": "осушка",
    "stale_osm": "устаревший OSM",
    "shadow": "тень или тёмная крыша в WorldCover",
}

RULE = ("основа — OSM; WorldCover заменяет её, когда OSM воды почти не видит "
        "(osm < 0.5 % или меньше четверти показанного снимком при wc ≥ 1 %); "
        "при прочих расхождениях остаётся OSM с понижением уверенности и причиной")

# Порог «OSM воды не видит вовсе» и «видит в разы меньше снимка»
BLIND_OSM = 0.005
BLIND_WC = 0.01
BLIND_RATIO = 0.25
# Застройка, при которой тень может читаться как вода. Тень — эффект на единицы процентов:
# разрыв в 15 п.п. в городе объясняется не тенью, а тем, чего в OSM нет вовсе.
URBAN_SHADOW_BUILT = 0.6
URBAN_SHADOW_MAX_GAP = 0.10
# Длина линейной воды в секторе, подтверждающая «река нарисована осевой», км
AXIS_LINEAR_KM = 0.3


def _diff(osm: float, wc: float) -> float:
    return float(osm) - float(wc)


def reason_for(osm: float, wc: float, *, linear_km: float | None = None,
               tidal: bool = False, built_share: float | None = None) -> str | None:
    """Причина расхождения в клине. `None` — расхождение в допуске, объяснять нечего.

    `tidal` — в кольце есть приливная река (свойство места, а не клина). Коды `bridge` и
    `dock_lock` зарезервированы: чтобы отличить мост от дока, нужны объекты `bridge=yes`,
    `lock_gate`, `dock`, а такой выгрузки нет — придумывать по косвенным признакам нельзя.
    """
    diff = _diff(osm, wc)
    if abs(diff) <= SECTOR_TOLERANCE:
        return None
    if osm < BLIND_OSM and wc >= BLIND_WC:
        # река осевой линией: длина линии есть, площади нет
        return "axis_line" if (linear_km or 0.0) >= AXIS_LINEAR_KM else "stale_osm"
    if diff < 0:
        # снимок видит больше: тень и тёмные крыши дают единицы процентов в плотной застройке,
        # а большой разрыв — это вода, которой в OSM нет (не размечена или появилась позже)
        if (built_share or 0.0) >= URBAN_SHADOW_BUILT and abs(diff) <= URBAN_SHADOW_MAX_GAP:
            return "shadow"
        return "stale_osm"
    # OSM видит больше: приливная река в отлив, мост или док за шлюзом
    if tidal:
        return "tidal_flat"
    return "stale_osm"


def confidence_for(osm: float, wc: float, *, ring_diff: float, reason: str | None) -> str:
    """Уверенность в выбранном значении: high — кольцо и клин в допуске, low — клин вне нормы."""
    if abs(_diff(osm, wc)) > SECTOR_TOLERANCE or reason is not None:
        return "low"
    if abs(ring_diff) > RING_TOLERANCE:
        return "medium"
    return "high"


def choose(osm: float, wc: float, *, reason: str | None) -> float:
    """Выбранное значение доли: OSM, кроме случая, когда OSM воды почти не видит."""
    blind = osm < BLIND_OSM or osm < BLIND_RATIO * wc
    if reason in {"axis_line", "stale_osm"} and blind and wc >= BLIND_WC:
        return round(float(wc), 4)
    return round(float(osm), 4)


def explain(reason: str | None) -> str | None:
    """Расшифровка кода причины для текста отчёта: в пакете код, человеку — фраза."""
    return None if reason is None else REASONS.get(reason, reason)


__all__ = ["ALGO_VERSION", "RING_TOLERANCE", "SECTOR_TOLERANCE", "REASONS", "RULE", "reason_for",
           "confidence_for", "choose", "explain"]
