"""Имя для обращения: только имя, без отчества.

Отчество режется по окончаниям ``-ович``, ``-евич``, ``-ична``, ``-овна``,
``-евна``. Фамилия не трогается.
"""

from __future__ import annotations

import re

_PATRONYMIC = re.compile(
    r"\s+\S+(?:ович|евич|ична|овна|евна)\b",
    re.IGNORECASE,
)


def given_name(value: str) -> str:
    """Оставляет имя для обращения, отрезая отчество.

    Args:
        value: как представился человек.

    Returns:
        Имя без отчества; фамилия сохраняется.

    Examples:
        «Андрей Андреевич» → «Андрей»;
        «Андрей Петров» без изменений;
        «Мария Ивановна» → «Мария».
    """
    text = (value or "").strip()
    if not text:
        return ""
    return _PATRONYMIC.sub("", text).strip()
