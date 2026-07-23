"""Сверка намеченной реплики с произнесённой.

Правду о перебивании знает только бот: в историю он пишет фактически
произнесённую часть, посчитанную по темпу проигрывания. Произнесённое всегда
префикс намеченного, поэтому недосказанное считается вычитанием. Если перебили
до первого звука, записи в истории не появится вообще.

Отсюда правило: **шаг закрывается по произнесённому, а не по сгенерированному.**
Граф намечает реплику и запоминает её длину и число реплик бота в истории; на
следующем ходу сравнивает — и, если нас не дослушали, возвращает шаг в работу.

В состоянии при этом лежат только числа и идентификаторы, текстов нет.
Все функции чистые и тестируются офлайн.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage

#: Какую долю намеченного надо произнести, чтобы считать шаг отработанным.
#: Ниже порога — клиент перебил слишком рано и содержания не услышал.
SPOKEN_ENOUGH = 0.6


def count_agent_messages(messages: Sequence[BaseMessage]) -> int:
    """Считает реплики бота в истории.

    Args:
        messages: история звонка.

    Returns:
        Число реплик бота.
    """
    return sum(1 for m in messages if isinstance(m, AIMessage))


def spoken_ratio(planned_len: int, spoken_len: int) -> float:
    """Какая доля намеченной реплики прозвучала.

    Args:
        planned_len: длина намеченного текста.
        spoken_len: длина произнесённого текста.

    Returns:
        Доля от нуля до единицы; для пустого намеченного — единица.
    """
    if planned_len <= 0:
        return 1.0
    return min(1.0, max(0.0, spoken_len / planned_len))


def was_delivered(
    *,
    planned_len: int,
    spoken_len: int,
    ai_count_before: int,
    ai_count_now: int,
) -> bool:
    """Донесли ли до клиента то, что наметили на прошлом ходу.

    Args:
        planned_len: длина намеченной реплики.
        spoken_len: длина последней реплики бота в истории.
        ai_count_before: сколько реплик бота было на конец прошлого хода.
        ai_count_now: сколько их сейчас.

    Returns:
        False, если реплика не прозвучала вовсе или прозвучала слишком коротко.
    """
    if planned_len <= 0:
        return True
    if ai_count_now <= ai_count_before:
        # Записи не появилось: перебили до первого звука.
        return False
    return spoken_ratio(planned_len, spoken_len) >= SPOKEN_ENOUGH


def reopen_if_interrupted(
    *,
    state: Mapping[str, Any],
    messages: Sequence[BaseMessage],
    last_spoken: str,
) -> dict[str, Any]:
    """Возвращает шаг в работу, если прошлую реплику не дослушали.

    Args:
        state: состояние звонка на входе хода.
        messages: история звонка после чистки от системных сообщений.
        last_spoken: последняя реплика бота из истории.

    Returns:
        Правки состояния: пустой словарь, если сверять нечего или всё в порядке.
    """
    pending = state.get("pending_step")
    if not pending:
        return {}

    delivered = was_delivered(
        planned_len=int(state.get("pending_len") or 0),
        spoken_len=len(last_spoken),
        ai_count_before=int(state.get("pending_ai_count") or 0),
        ai_count_now=count_agent_messages(messages),
    )
    if delivered:
        return {"pending_step": None, "pending_len": 0}

    status = dict(state.get("step_status") or {})
    if status.get(pending) == "done":
        status[pending] = "open"
    return {"step_status": status, "pending_step": None, "pending_len": 0}
