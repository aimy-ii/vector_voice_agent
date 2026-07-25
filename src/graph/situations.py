"""Словарь ситуативных заглушек вне скрипта.

Слаг → набор готовых фраз, плюс универсальный ``default``. Вызовов модели
в рантайме ноль: выбор случайный, мимо уже звучавших в звонке.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

#: Имя ключа с универсальными фразами.
DEFAULT_SLUG = "default"

_DATA_PATH = Path(__file__).resolve().parent / "data" / "situations.json"


@lru_cache(maxsize=1)
def load_situations() -> Mapping[str, tuple[str, ...]]:
    """Загружает словарь ситуаций из JSON один раз.

    Returns:
        Слаг → кортеж готовых фраз.
    """
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    result: dict[str, tuple[str, ...]] = {}
    for slug, phrases in raw.items():
        cleaned = tuple(p.strip() for p in phrases if isinstance(p, str) and p.strip())
        if cleaned:
            result[str(slug)] = cleaned
    if DEFAULT_SLUG not in result:
        raise RuntimeError(f"В {_DATA_PATH} нет обязательного ключа «{DEFAULT_SLUG}»")
    return result


def pick_filler(slug: str | None, *, spoken: Sequence[str] = ()) -> str:
    """Готовая фраза-заглушка по слагу ситуации.

    Слаг не из словаря / пустой → берётся default. Молчания не бывает.
    Выбор случайный, мимо уже звучавших в этом звонке. Вызовов модели ноль.

    Args:
        slug: слаг ситуации от контекстера.
        spoken: фразы, уже звучавшие в этом звонке.

    Returns:
        Непустая готовая фраза.
    """
    catalog = load_situations()
    key = (slug or "").strip()
    pool = list(catalog.get(key) or catalog[DEFAULT_SLUG])
    used = set(spoken or [])
    fresh = [p for p in pool if p not in used]
    choice = random.choice(fresh or pool)
    if not choice:
        # Защита: словарь обязан быть непустым, но молчания всё равно нет.
        return catalog[DEFAULT_SLUG][0]
    return choice
