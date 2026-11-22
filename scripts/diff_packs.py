#!/usr/bin/env python3
"""Сравнить два пакета участка: какие поля изменились (ТЗ Б4).

Использование:
    python scripts/diff_packs.py old/site_pack.json new/site_pack.json

Печатает только изменившиеся значения, по одному полю на строку. Водные объекты
сопоставляются по id, а не по порядку, иначе перестановка выглядит как правка.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def flatten(obj, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            key = value.get("id") if isinstance(value, dict) and "id" in value else idx
            out.update(flatten(value, f"{prefix}[{key}]"))
    else:
        out[prefix] = obj
    return out


def main(old_path: str, new_path: str) -> int:
    old = flatten(json.loads(Path(old_path).read_text(encoding="utf-8")))
    new = flatten(json.loads(Path(new_path).read_text(encoding="utf-8")))
    changed = 0
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key, "<нет>"), new.get(key, "<нет>")
        if a != b:
            changed += 1
            print(f"{key}: {a!r} → {b!r}")
    print(f"--- изменённых полей: {changed}")
    return 0 if changed == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
