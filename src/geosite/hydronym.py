"""Нормализация имён водотоков и доля подземного участка.

OSM хранит одну реку под разными именами: `River Fleet` и `The Fleet`, `River Thames` и
`Thames`. Без нормализации это два водотока, и в отчёте река дублируется. Правила — в
`data/name_aliases.json`, чтобы список можно было пополнять без правки кода.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "name_aliases.json"


@lru_cache(maxsize=1)
def rules() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def normalize_name(name: str | None) -> str | None:
    """Привести имя к ключу склейки: без артикля, без родового слова, в нижнем регистре."""
    if not name:
        return None
    cfg = rules()
    text = name.strip().lower().replace("ё", "е")
    text = " ".join(text.split())
    alias = (cfg.get("aliases") or {}).get(text)
    if alias:
        return alias
    words = text.split()
    while words and words[0] in cfg.get("articles", []):
        words.pop(0)
    while words and words[0] in cfg.get("generic_leading", []):
        words.pop(0)
    text = " ".join(words)
    if not text:
        return None
    return (cfg.get("aliases") or {}).get(text, text)


def is_hidden(tags: dict) -> bool:
    """Кусок идёт под землёй: труба, накрыт, подземный уровень."""
    cfg = (rules().get("hidden_tags") or {})
    for key, values in cfg.items():
        value = (tags.get(key) or "").lower()
        if value and value in values:
            return True
    return False
