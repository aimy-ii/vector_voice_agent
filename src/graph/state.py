"""Состояние звонка и контекст запуска.

Два решения, которые здесь важнее остальных.

**`messages` заменяется, а не накапливается.** Плагин на каждом ходу подаёт
всю историю звонка, собранную из своего `chat_ctx`, со своими же
идентификаторами. При перебивании бот пишет в историю обрезанный текст с одним
идентификатором, а положенный узлом полный ответ — с другим; `add_messages`
не найдёт совпадения и оставит обе редакции рядом. Одна реплика дважды, и так
каждый ход. Поэтому редьюсер на замену, а `messages` — вход и рабочий буфер на
один ход, не летопись.

**Прогресс — только идентификаторы и значения.** Никаких текстов скрипта в
состоянии: тексты достаются из скомпилированного скрипта по идентификатору.
Состояние получается маленьким и дешёвым, скрипт — тяжёлым и общим. В нём же
лежит точная версия скрипта: выкатили новую посреди смены — идущие звонки
доигрывают на своей.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


def replace_messages(current: list[BaseMessage], incoming: list[BaseMessage]) -> list[BaseMessage]:
    """Редьюсер `messages`: новое значение вытесняет старое.

    Args:
        current: что лежало в состоянии.
        incoming: что пришло от плагина или положил узел.

    Returns:
        Входящий список; None трактуется как «не трогать».
    """
    if incoming is None:
        return current
    return list(incoming)


class CallContext(TypedDict, total=False):
    """Параметры запуска, приходящие снаружи на каждый ход.

    В чекпоинт не пишутся. Всё здесь необязательное: рабочий путь — бот
    подключается без контекста, а скрипт и город берутся по умолчанию и
    выясняются в разговоре.

    Attributes:
        script_id: какой скрипт вести; пусто — из настроек.
        script_version: точная версия скрипта; пусто — последняя.
        city_slug: город, если он каким-то образом известен заранее. Сейчас
            неприменимо: телефон у сети один федеральный, по номеру звонка
            город не вычислить. Оставлено на будущее — например, под обзвон
            по базе, где город известен из карточки.
    """

    script_id: str
    script_version: str
    city_slug: str


class CallState(TypedDict, total=False):
    """Состояние звонка.

    Attributes:
        messages: история звонка на этот ход (редьюсер на замену).
        script_id: идентификатор скрипта.
        script_version: точная версия скрипта.
        step_status: статус по каждому шагу: open / done / refused / skipped.
        step_attempts: сколько раз возвращались к шагу — предохранитель
            от зацикливания.
        profile: собранный профиль, плоско по ключам полей.
        city_slug: подтверждённый слаг города из справочника.
        branch_slug: подтверждённый слаг филиала.
        resume_step: куда вернуться после справки или возражения.
        asides_done: какие справки и возражения уже отработали.
        current_step: шаг, которым занимались на этом ходу.
        outcome: чем закончили: визит, дистанционная предзапись, мессенджер.
        tool_log: журнал походов в справочник. Отдельным полем, а не в
            `messages`: тот затрётся на следующем ходу.
        turn: номер хода.
        last_error: что сломалось на прошлом ходу, если сломалось.
        pending_step: шаг, закрытый на прошлом ходу, но ещё не сверенный
            с произнесённым.
        pending_len: длина намеченной реплики прошлого хода. Правду о
            перебивании знает только бот: в историю он пишет фактически
            произнесённое. Сверив длины, узнаём, дослушали ли нас, — и
            держим шаг открытым, если не дослушали. Хранится число, не текст.
        pending_ai_count: сколько реплик бота было в истории на конец прошлого
            хода. Если перебили до первого звука, записи не появится вовсе —
            это видно по несдвинувшемуся счётчику.
        facts: факты справочника этого хода (рабочий буфер, затирается).
        route: каким путём пошёл ход: verbatim / lookup / respond.
        skip_model: модель на этом ходу не нужна — разбирать нечего.
        spoken: куски, отданные в эфир на этом ходу.
        turn_result: разбор реплики клиента моделью на этом ходу.
    """

    messages: Annotated[list[BaseMessage], replace_messages]

    script_id: str
    script_version: str

    step_status: dict[str, str]
    step_attempts: dict[str, int]
    profile: dict[str, str]

    city_slug: str | None
    branch_slug: str | None

    resume_step: str | None
    asides_done: list[str]
    current_step: str | None
    outcome: str | None

    tool_log: list[dict[str, Any]]
    turn: int
    last_error: str | None

    pending_step: str | None
    pending_len: int
    pending_ai_count: int

    facts: dict[str, Any]
    route: str | None
    skip_model: bool
    spoken: list[str]
    turn_result: dict[str, Any]


def new_state_defaults() -> dict[str, Any]:
    """Дефолты полей состояния, чтобы узлы не падали на None.

    Returns:
        Словарь с пустыми коллекциями и нулевыми счётчиками.
    """
    return {
        "step_status": {},
        "step_attempts": {},
        "profile": {},
        "city_slug": None,
        "branch_slug": None,
        "resume_step": None,
        "asides_done": [],
        "current_step": None,
        "outcome": None,
        "tool_log": [],
        "turn": 0,
        "last_error": None,
        "pending_step": None,
        "pending_len": 0,
        "pending_ai_count": 0,
        "facts": {},
        "route": None,
        "skip_model": False,
        "spoken": [],
        "turn_result": {},
    }
