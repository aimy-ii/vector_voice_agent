"""Фоновое дозаполнение базовых полей профиля из сказанного.

Только детерминированные эвристики — без модели. Решения (тариф, время
встречи, исход разговора) отсюда не проставляются.
"""

from __future__ import annotations

import re
from typing import Mapping

from graph.names import given_name

#: Поля, которые фон имеет право заполнять.
_BASIC_KEYS: frozenset[str] = frozenset({"city", "caller_name", "transmission", "theory_format"})

#: Поля, которые фон никогда не трогает.
_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "outcome",
        "package",
        "tariff",
        "meeting_time",
        "appointment",
        "slot",
        "payment_pref",
    }
)

_NAME_RE = re.compile(r"(?i)(?:меня\s+зовут|зовите\s+меня|мо[её]\s+имя)\s+([А-ЯЁA-Z][а-яёa-z]+)")
_CITY_RE = re.compile(
    r"(?i)(?:(?:из|в|город(?:е)?)\s+)([А-ЯЁA-Z][а-яёa-z\-]+(?:[\s\-][А-ЯЁA-Z]?[а-яёa-z]+)?)"
)
_TRANSMISSION_RE = re.compile(
    r"(?i)\b(механик[аеиу]?|МКПП|автомат(?:е|а|ом)?|АКПП|automatic|manual)\b"
)
_THEORY_RE = re.compile(r"(?i)\b(очно|в\s+классе|дистанционн\w*|онлайн|удалённ\w*|удаленн\w*)\b")


def fill_basic_profile(
    text: str,
    profile: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Достаёт незаполненные базовые поля из уже сказанного.

    Args:
        text: реплика или накопленный partial клиента.
        profile: текущий профиль; заполненные ключи не перезаписываются.

    Returns:
        Словарь только новых базовых полей. Исход, тариф и время встречи
        сюда никогда не попадают.
    """
    current = dict(profile or {})
    found: dict[str, str] = {}
    haystack = text or ""
    if not haystack.strip():
        return found

    if "caller_name" not in current or not str(current.get("caller_name") or "").strip():
        match = _NAME_RE.search(haystack)
        if match:
            name = given_name(match.group(1)) or match.group(1)
            if name and name.lower() not in {"из", "в", "на", "по"}:
                found["caller_name"] = name

    if "city" not in current or not str(current.get("city") or "").strip():
        match = _CITY_RE.search(haystack)
        if match:
            city = match.group(1).strip(" .,!")
            # Отсекаем частые ложные срабатывания.
            if city.lower() not in {"меня", "вас", "наш", "этой", "этом"}:
                found["city"] = city

    if "transmission" not in current or not str(current.get("transmission") or "").strip():
        match = _TRANSMISSION_RE.search(haystack)
        if match:
            raw = match.group(1).lower()
            if "автомат" in raw or "акпп" in raw or raw == "automatic":
                found["transmission"] = "автомат"
            else:
                found["transmission"] = "механика"

    if "theory_format" not in current or not str(current.get("theory_format") or "").strip():
        match = _THEORY_RE.search(haystack)
        if match:
            raw = match.group(1).lower()
            if "очно" in raw or "класс" in raw:
                found["theory_format"] = "очно"
            else:
                found["theory_format"] = "дистанционно"

    # Защита: запрещённые ключи наружу не отдаём.
    return {
        key: value
        for key, value in found.items()
        if key in _BASIC_KEYS and key not in _FORBIDDEN_KEYS and value.strip()
    }
