"""Сверка намеченной реплики с произнесённой.

Правду о перебивании знает только бот: в историю он пишет фактически
произнесённую часть. Если перебили слишком рано, информирующий шаг не
закрывается чекером на следующем ходу.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage

#: Какую долю намеченного надо произнести, чтобы считать шаг отработанным.
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
        return False
    return spoken_ratio(planned_len, spoken_len) >= SPOKEN_ENOUGH


def delivery_patch(
    *,
    state: Mapping[str, Any],
    messages: Sequence[BaseMessage],
    last_spoken: str,
) -> dict[str, Any]:
    """Считает, дослушали ли прошлую реплику, и чистит pending-поля.

    Закрытие шага делает чекер; здесь только факт доставки.

    Args:
        state: состояние звонка на входе хода.
        messages: история после чистки.
        last_spoken: последняя реплика бота.

    Returns:
        Правки: ``pending_*`` и ``last_delivered``.
    """
    pending = state.get("pending_step")
    if not pending:
        return {"last_delivered": True}

    delivered = was_delivered(
        planned_len=int(state.get("pending_len") or 0),
        spoken_len=len(last_spoken),
        ai_count_before=int(state.get("pending_ai_count") or 0),
        ai_count_now=count_agent_messages(messages),
    )
    return {
        "pending_step": None,
        "pending_len": 0,
        "last_delivered": delivered,
        "delivered_step": pending if delivered else None,
        "undelivered_step": None if delivered else pending,
    }


#: Совместимое имя для старых тестов: при недоставке шаг остаётся pending.
def reopen_if_interrupted(
    *,
    state: Mapping[str, Any],
    messages: Sequence[BaseMessage],
    last_spoken: str,
) -> dict[str, Any]:
    """Возвращает правки после сверки произнесённого.

    Args:
        state: состояние звонка на входе хода.
        messages: история звонка.
        last_spoken: последняя реплика бота из истории.

    Returns:
        Правки состояния.
    """
    patch = delivery_patch(state=state, messages=messages, last_spoken=last_spoken)
    undelivered = patch.get("undelivered_step")
    if not undelivered:
        return {
            k: v for k, v in patch.items() if k in {"pending_step", "pending_len", "last_delivered"}
        }

    status = dict(state.get("step_status") or {})
    # Незакрытый информативный шаг остаётся pending; closed не откатываем здесь —
    # закрытие только у чекера. Для совместимости со старыми слепками done→pending.
    if status.get(undelivered) in {"done", "closed"}:
        status[undelivered] = "pending"
        patch["step_status"] = status
    return {
        "pending_step": None,
        "pending_len": 0,
        "last_delivered": False,
        "step_status": patch.get("step_status", status),
    }
