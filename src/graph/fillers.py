"""Фразы в эфир, пока идёт поиск города и филиала.

Только из данных, вызовов модели ноль. Шаблон — реакция, а не объявление о
поиске. Генератор видит, какая фраза прозвучала, и не начинает со второго
вступления.
"""

from __future__ import annotations

import random
from typing import Sequence

#: Запасные шаблоны, если в скрипте пусто. Плейсхолдер ``{place}``.
_DEFAULT_CITY = (
    "так, {place}… секунду, открываю по {place}",
    "а, {place}… так, гляну по {place}",
    "{place}, да… минутку, открою карточку",
    "так, по {place}… секундочку",
    "угу, {place}… сейчас открою",
)

_DEFAULT_BRANCH = (
    "так, район… секунду, гляну филиалы рядом",
    "а, поняла… минутку, открою адреса",
    "так, сейчас сверю с филиалами",
    "угу… секунду, посмотрю что ближе",
    "район, да… сейчас подберу пару адресов",
)


def pick_filler(
    templates: Sequence[str],
    *,
    place: str | None = None,
    used: Sequence[str] | None = None,
    defaults: Sequence[str] = (),
) -> str:
    """Выбирает фразу-заглушку без вызова модели.

    Args:
        templates: шаблоны из данных скрипта.
        place: город или ориентир для подстановки.
        used: уже звучавшие фразы в этом звонке.
        defaults: запасные шаблоны.

    Returns:
        Готовая фраза для эфира.
    """
    pool = [t for t in templates if t and t.strip()] or list(defaults)
    if not pool:
        return "так… секунду"
    used_set = set(used or [])
    fresh = [t for t in pool if t not in used_set]
    choice = random.choice(fresh or list(pool))
    if place and "{place}" in choice:
        return choice.format(place=place)
    if place and "{city}" in choice:
        return choice.format(city=place)
    return choice


def city_filler(
    templates: Sequence[str],
    *,
    city_name: str,
    used: Sequence[str] | None = None,
) -> str:
    """Заглушка на поиск города."""
    return pick_filler(templates, place=city_name, used=used, defaults=_DEFAULT_CITY)


def branch_filler(
    templates: Sequence[str],
    *,
    used: Sequence[str] | None = None,
) -> str:
    """Заглушка на поиск филиала."""
    return pick_filler(templates, place=None, used=used, defaults=_DEFAULT_BRANCH)
