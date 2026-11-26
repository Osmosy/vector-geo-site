"""Имена любой письменности: нормализация, языки, родовые слова, сезонная вода (G3).

Зачем: за пределами Европы имя редко приходит латиницей в `name`. Гонконг пишет `城門河`,
Каир — `نهر النيل`, Джакарта — `Sungai Ciliwung`, Стамбул — `Kâğıthane Deresi`. Если смотреть
только `name`, половина мира приходит безымянной, а «безымянная вода» не склеивается с одноимённой
и не попадает в отчёт.

Правила:
- нормализация Unicode NFKC — «ﬁ» → «fi», полноширинные цифры и латиница → обычные, чтобы одно
  и то же имя не давало двух ключей;
- имя берётся из `name`, `name:en`, `name:zh`, `name:ru`, `int_name`, `official_name`, `loc_name`,
  `alt_name`; список альтернатив сохраняется, потому что навыку нужны оба имени;
- родовое слово (`data/generic_words.json`) говорит, что за вода, когда тега нет: `湖` — озеро,
  `運河` — канал, `Sungai` — река, `نهر` — река, `Rio` — река;
- сезонная вода: `True` — если тег или родовое слово говорят «временная»; `False` — только если
  тег прямо говорит «постоянная»; иначе `None`: имя без тега сезонности ничего не доказывает.
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

ALGO_VERSION = "1.0.0"

DATA = Path(__file__).resolve().parents[2] / "data" / "generic_words.json"

# порядок важен: `name` — основное, дальше то, что чаще всего заполнено у воды
NAME_KEYS = ("name", "name:en", "name:zh", "name:ru", "int_name", "official_name", "loc_name",
             "alt_name")

KIND_ALIASES = {
    "river": "river", "stream": "stream", "canal": "canal", "lake": "lake", "pond": "pond",
    "sea": "sea", "bay": "bay", "harbour": "harbour", "harbor": "harbour", "wetland": "wetland",
    "wadi": "stream", "creek": "stream",
}


@lru_cache(maxsize=1)
def rules() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def nfkc(text: str | None) -> str | None:
    """Нормализация Unicode: одна форма — один ключ склейки."""
    if not text:
        return None
    value = unicodedata.normalize("NFKC", str(text)).strip()
    return " ".join(value.split()) or None


def names_of(tags: dict) -> list[str]:
    """Все имена воды из тегов: основное первым, затем альтернативы без повторов."""
    out: list[str] = []
    for key in NAME_KEYS:
        raw = tags.get(key)
        if not raw:
            continue
        for part in str(raw).split(";"):
            value = nfkc(part)
            if value and value not in out:
                out.append(value)
    return out


def primary_name(tags: dict) -> str | None:
    names = names_of(tags)
    return names[0] if names else None


def kind_from_name(name: str | None) -> dict | None:
    """Что за вода по родовому слову: `{"kind", "language", "word"}` или None.

    Смотрим три места: последний иероглиф (китайские и японские имена ставят родовое слово в
    конец), первое слово (арабский, индонезийский, тюркский, романские — в начало), и вхождение
    внутри (русский: «Чёрная речка», «Москва-река»).
    """
    value = nfkc(name)
    if not value:
        return None
    cfg = rules()

    for word in sorted(cfg.get("suffix_cjk") or {}, key=len, reverse=True):
        if value.endswith(word) and len(value) > len(word):
            return {"kind": KIND_ALIASES.get(cfg["suffix_cjk"][word], "other"),
                    "language": "zh" if _is_cjk(value) else "unknown", "word": word}

    tokens = [t for t in value.replace("-", " ").split() if t]
    # артикли не родовые слова: «La Seine» — не озеро
    while len(tokens) > 1 and tokens[0].lower() in (cfg.get("articles") or []):
        tokens.pop(0)
    lowered = [t.lower() for t in tokens]
    if lowered:
        for language, table in (cfg.get("prefix") or {}).items():
            first = lowered[0]
            if first in table:
                return {"kind": KIND_ALIASES.get(table[first], "other"),
                        "language": language, "word": tokens[0]}
        for language, table in (cfg.get("prefix") or {}).items():
            last = lowered[-1]
            if last in table:
                return {"kind": KIND_ALIASES.get(table[last], "other"),
                        "language": language, "word": tokens[-1]}

    if lowered:
        for language, table in (cfg.get("suffix_words") or {}).items():
            last = lowered[-1]
            if last in table:
                return {"kind": KIND_ALIASES.get(table[last], "other"),
                        "language": language, "word": tokens[-1]}

    for language, table in (cfg.get("contains") or {}).items():
        for word, kind in table.items():
            if word in lowered or word in value.lower():
                return {"kind": KIND_ALIASES.get(kind, "other"), "language": language,
                        "word": word}
    return None


def _is_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff" for char in text)


def is_seasonal(tags: dict, names: list[str] | None = None) -> bool | None:
    """Временная вода: `intermittent`, `seasonal`, `waterway=wadi` или родовое слово.

    `None` — «не смотрели», `False` — теги сказали «постоянная» или молчат, но имя не временное.
    Разница нужна навыку: временную воду он называет временной, а не «слабой».
    """
    cfg = rules()
    negative = False
    for key, values in (cfg.get("seasonal_tags") or {}).items():
        tag = str(tags.get(key) or "").strip().lower()
        if not tag:
            continue
        if tag in values:
            return True
        if tag in {"no", "permanent", "perennial"}:
            negative = True
    text = " ".join(names or names_of(tags)).lower()
    for language, words in (cfg.get("seasonal_words") or {}).items():
        if any(word in text for word in words):
            return True
    # «постоянная» — только когда это сказано тегом: имя без тега ничего не доказывает
    return False if negative else None


def language_of(name: str | None) -> str | None:
    """Язык письменности: `zh` для иероглифов, `ar` для арабского, `cyrillic`, `latin`."""
    value = nfkc(name)
    if not value:
        return None
    if _is_cjk(value):
        return "zh"
    for char in value:
        code = ord(char)
        if 0x0600 <= code <= 0x06FF:
            return "ar"
        if 0x0400 <= code <= 0x04FF:
            return "cyrillic"
        if char.isalpha():
            return "latin"
    return None
