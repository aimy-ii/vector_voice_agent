"""Состояние звонка и контекст запуска.

**`messages` заменяется, а не накапливается.** Плагин на каждом ходу подаёт
всю историю звонка. Редьюсер на замену, иначе при перебивании дубли.

**Прогресс скрипта** в рантайме живёт в Redis; в треде — зеркало для промаха
и слепок на конец звонка.
"""

from __future__ import annotations

from typing import Annotated, Any, Sequence

from langchain_core.messages import BaseMessage, convert_to_messages
from typing_extensions import TypedDict


def replace_messages(
    current: list[BaseMessage],
    incoming: Sequence[BaseMessage | dict[str, Any]] | None,
) -> list[BaseMessage]:
    """Редьюсер `messages`: новое значение вытесняет старое.

    Args:
        current: что лежало в состоянии.
        incoming: что пришло от плагина/сервера или положил узел.

    Returns:
        Входящий список, приведённый к объектам сообщений; ``None`` —
        «не трогать»; пустой список — очистить.
    """
    if incoming is None:
        return current
    return convert_to_messages(incoming)


def merge_dicts(
    current: dict[str, Any] | None,
    incoming: dict[str, Any] | None,
) -> dict[str, Any]:
    """Редьюсер словарей: точечное слияние при параллельной записи узлов.

    Args:
        current: что уже лежит в состоянии.
        incoming: правка узла; ``None`` — не трогать.

    Returns:
        Копия ``current`` с наложенным ``incoming``.
    """
    merged = dict(current or {})
    if not incoming:
        return merged
    merged.update(incoming)
    return merged


class CallContext(TypedDict, total=False):
    """Параметры запуска, приходящие снаружи на каждый ход."""

    script_id: str
    script_version: str
    city_slug: str


class CallState(TypedDict, total=False):
    """Состояние звонка.

    Attributes:
        messages: история звонка на этот ход (редьюсер на замену).
        script_id / script_version: закреплённый скрипт звонка.
        step_status: pending / closed по шагам (зеркало Redis).
        step_attempts: счётчик попыток задать шаг (сколько раз был в шапке).
        step_taken_turn: ход первого взятия шага.
        script_progress: слепок прогресса (на конец звонка — на постоянку).
        profile: собранный профиль.
        city_slug / city_name / branch_slug: фиксации справочника.
        conversation_context: единый документ контекста.
        head_steps: шапка шагов этого хода.
        head_new_step: идентификатор шага, взятого в шапку впервые на этом
            ходу; ``None`` — новых шагов нет, добор отсечён или все шаги
            висящие. Считать по счётчикам в промпте нельзя: к моменту
            генерации они уже увеличены.
        expect_continuation: после реплики бот сам запустит следующий ход
            без реплики клиента (контекст ещё готовится).
        turn_kind: ``client`` — обычный ход по реплике клиента;
            ``continuation`` — продолжение собственной речи бота;
            ``silence`` — человек молчит, бот мягко возвращает в разговор.
        branch_candidates: отобранные резолвером слаги филиалов.
        partial_reply: накопленный распознанный текст текущей реплики
            клиента; вход служебного графа ``vector_checker``.
        partial_utterance_id: идентификатор текущей реплики от бота;
            смена значения — новая реплика, точка отсчёта прироста сбрасывается.
        partial_is_final: финальный кусок реплики; при True лайв-канал
            разбирает всегда, порог прироста не применяется.
        last_checked_partial: текст последнего служебного прохода чекера
            (порог прироста внутри текущей реплики).
        last_checked_utterance_id: ``partial_utterance_id``, к которому
            относится ``last_checked_partial``.
    """

    messages: Annotated[list[BaseMessage], replace_messages]

    script_id: str
    script_version: str

    step_status: dict[str, str]
    step_attempts: dict[str, int]
    step_taken_turn: dict[str, int]
    script_progress: dict[str, Any]

    profile: Annotated[dict[str, str], merge_dicts]
    client_asks_inform: bool
    city_slug: str | None
    city_name: str | None
    branch_slug: str | None
    conversation_context: dict[str, Any]

    resume_step: str | None
    asides_done: list[str]
    current_step: str | None
    next_step: str | None
    head_steps: list[str]
    head_new_step: str | None
    outcome: str | None

    tool_log: list[dict[str, Any]]
    turn: int
    last_error: str | None

    pending_step: str | None
    pending_len: int
    pending_ai_count: int
    last_delivered: bool
    delivered_step: str | None

    facts: dict[str, Any]
    spoken: list[str]
    branch_candidates: list[str]
    turn_result: dict[str, Any]
    call_finished: bool
    expect_continuation: bool
    turn_kind: str
    partial_reply: str
    partial_utterance_id: str
    partial_is_final: bool
    last_checked_partial: str
    last_checked_utterance_id: str


def new_state_defaults() -> dict[str, Any]:
    """Дефолты полей состояния, чтобы узлы не падали на None."""
    return {
        "step_status": {},
        "step_attempts": {},
        "step_taken_turn": {},
        "script_progress": {},
        "profile": {},
        "client_asks_inform": False,
        "city_slug": None,
        "city_name": None,
        "branch_slug": None,
        "conversation_context": {},
        "resume_step": None,
        "asides_done": [],
        "current_step": None,
        "next_step": None,
        "head_steps": [],
        "head_new_step": None,
        "outcome": None,
        "tool_log": [],
        "turn": 0,
        "last_error": None,
        "pending_step": None,
        "pending_len": 0,
        "pending_ai_count": 0,
        "last_delivered": True,
        "delivered_step": None,
        "facts": {},
        "spoken": [],
        "branch_candidates": [],
        "turn_result": {},
        "call_finished": False,
        "expect_continuation": False,
        "turn_kind": "client",
        "partial_reply": "",
        "partial_utterance_id": "",
        "partial_is_final": False,
        "last_checked_partial": "",
        "last_checked_utterance_id": "",
    }
