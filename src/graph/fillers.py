"""Фразы в эфир, пока идёт поиск города и филиала.

Только из данных, вызовов модели ноль. Предмет подставляет код из закрытого
набора: город, филиал, стоимость. Свободный текст клиента в шаблон не
попадает никогда. Нет предмета — нет заглушки.
"""

from __future__ import annotations

import random
from typing import Sequence

#: Предметы, которые код вправе подставить в шаблон.
FILLER_SUBJECTS: frozenset[str] = frozenset({"город", "филиал", "стоимость"})

SUBJECT_CITY = "город"
SUBJECT_BRANCH = "филиал"
SUBJECT_COST = "стоимость"

#: Знаки, после которых точку в конец фразы не добавляем.
_TERMINAL_PUNCT = ".!?,:;"

#: Запасные шаблоны, если в скрипте пусто. Плейсхолдер ``{place}`` — именительный.
_DEFAULT_CITY = (
    "Так, {place}. Секунду, открываю.",
    "Угу, {place}. Сейчас гляну.",
    "{place}, да. Минутку, открою карточку.",
    "Так, {place}. Сейчас сверю.",
)

_DEFAULT_BRANCH = (
    "Так, {place}. Секунду, гляну адреса.",
    "Угу, {place}. Минутку, открою.",
    "{place}, да. Сейчас подберу пару адресов.",
    "Так, {place}. Сейчас посмотрю, что рядом.",
)

_DEFAULT_COST = (
    "Так, {place}. Секунду, открою.",
    "Угу, {place}. Минутку, гляну.",
    "{place}, да. Сейчас сверю.",
)


def _ensure_terminal_punct(text: str) -> str:
    """Нормализует хвост фразы: многоточие → точка, иначе точка при нужде.

    Многоточие в конце не закрывает сегмент для синтеза, поэтому отклик
    не уходит в эфир вовремя и сливается с репликой генератора.
    """
    if not text:
        return text
    if text.endswith("..."):
        return text[:-3] + "."
    if text.endswith("…"):
        return text[:-1] + "."
    if text[-1] not in _TERMINAL_PUNCT:
        return text + "."
    return text


def pick_filler(
    templates: Sequence[str],
    *,
    subject: str | None,
    used: Sequence[str] | None = None,
    defaults: Sequence[str] = (),
) -> str | None:
    """Выбирает фразу-заглушку без вызова модели.

    Args:
        templates: шаблоны из данных скрипта.
        subject: предмет из ``FILLER_SUBJECTS``; иначе молчим.
        used: уже звучавшие фразы в этом звонке.
        defaults: запасные шаблоны.

    Returns:
        Готовая фраза для эфира или ``None``, если предмета нет.
    """
    if subject not in FILLER_SUBJECTS:
        return None
    pool = [t for t in templates if t and t.strip()] or list(defaults)
    if not pool:
        return None
    used_set = set(used or [])
    fresh = [t for t in pool if t not in used_set]
    choice = random.choice(fresh or list(pool))
    if "{place}" in choice:
        phrase = choice.format(place=subject)
    elif "{city}" in choice:
        phrase = choice.format(city=subject)
    else:
        # Шаблон без плейсхолдера — предмет всё равно зафиксирован кодом.
        phrase = choice
    return _ensure_terminal_punct(phrase)


def city_filler(
    templates: Sequence[str],
    *,
    used: Sequence[str] | None = None,
) -> str | None:
    """Заглушка на поиск города; предмет всегда «город»."""
    return pick_filler(
        templates,
        subject=SUBJECT_CITY,
        used=used,
        defaults=_DEFAULT_CITY,
    )


def branch_filler(
    templates: Sequence[str],
    *,
    used: Sequence[str] | None = None,
) -> str | None:
    """Заглушка на поиск филиала; предмет всегда «филиал»."""
    return pick_filler(
        templates,
        subject=SUBJECT_BRANCH,
        used=used,
        defaults=_DEFAULT_BRANCH,
    )


def cost_filler(
    templates: Sequence[str],
    *,
    used: Sequence[str] | None = None,
) -> str | None:
    """Заглушка на поиск стоимости; предмет всегда «стоимость»."""
    return pick_filler(
        templates,
        subject=SUBJECT_COST,
        used=used,
        defaults=_DEFAULT_COST,
    )
