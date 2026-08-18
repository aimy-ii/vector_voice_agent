"""Тесты агента конца разговора и его встраивания в лайв-канал."""

from __future__ import annotations

import logging
from typing import Any, Sequence
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from graph.checker import CheckerVerdict
from graph.checker_graph import live_check_node
from graph.context import ConversationContext
from graph.context_agent import ContextDecision
from graph.context_store import MemoryContextStore
from graph.farewell_agent import (
    _HISTORY_TAIL,
    FAREWELL_SYSTEM,
    FarewellDecision,
    LlmFarewellAgent,
    _format_history,
    decide_farewell,
)
from graph.prompts import _INITIATIVE_BLOCK
from graph.schemas import TurnResult
from script.store import ScriptProgress, progress_to_state


class _FakeFarewell:
    """Возвращает заданное решение или падает."""

    def __init__(
        self,
        result: FarewellDecision | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result or FarewellDecision(conversation_ended=False)
        self.error = error
        self.calls = 0
        self.reply_seen: str = ""
        self.history_seen: list[BaseMessage] = []

    async def decide(
        self,
        reply: str,
        history: Sequence[BaseMessage] = (),
    ) -> FarewellDecision:
        self.calls += 1
        self.reply_seen = reply
        self.history_seen = list(history)
        if self.error is not None:
            raise self.error
        return self.result


async def test_агент_прощание_истина_деловая_ложь():
    """Прощание → True, деловая реплика → False."""
    bye = _FakeFarewell(FarewellDecision(conversation_ended=True))
    business = _FakeFarewell(FarewellDecision(conversation_ended=False))

    ended = await decide_farewell(
        "всё, до свидания",
        history=[HumanMessage(content="всё, до свидания")],
        agent=bye,
    )
    assert ended is not None
    assert ended.conversation_ended is True

    open_ = await decide_farewell(
        "давайте на среду в три",
        history=[HumanMessage(content="давайте на среду в три")],
        agent=business,
    )
    assert open_ is not None
    assert open_.conversation_ended is False


async def test_сбой_агента_возвращает_none():
    """Ошибка агента → None, без исключения наружу."""
    agent = _FakeFarewell(error=RuntimeError("модель недоступна"))
    result = await decide_farewell("до свидания", agent=agent)
    assert result is None


def test_схема_генератора_без_признака_завершения():
    """TurnResult содержит только reply."""
    assert list(TurnResult.model_fields) == ["reply"]
    assert "conversation_ended" not in TurnResult.model_fields
    parsed = TurnResult.model_validate({"reply": "Здравствуйте."})
    assert parsed.reply == "Здравствуйте."


@pytest.fixture
def _offline_farewell(monkeypatch):
    """Офлайн: кеш контекста в памяти, агенты без модели."""
    from graph import contexter as contexter_module
    from graph import nodes as nodes_module
    from graph.profile_agent import ProfileGuess

    mem = MemoryContextStore()
    monkeypatch.setattr(nodes_module, "context_store", mem)
    monkeypatch.setattr(contexter_module, "context_store", mem)

    async def _no_need(*_a, **_k):
        return ContextDecision(need=False)

    async def _no_profile(*_a, **_k):
        return ProfileGuess()

    monkeypatch.setattr("graph.contexter.decide_context", _no_need)
    monkeypatch.setattr("graph.contexter_worker.guess_profile", _no_profile)
    return mem


def _long_history(n: int = 5) -> list:
    """Диалог из n сообщений (порог farewell_min_messages по умолчанию)."""
    messages: list = []
    for i in range(n):
        if i % 2 == 0:
            messages.append(AIMessage(content=f"реплика бота {i}"))
        else:
            messages.append(HumanMessage(content=f"реплика клиента {i}"))
    return messages


def _name_progress() -> ScriptProgress:
    return ScriptProgress(
        status={"name": "pending"},
        attempts={"name": 1},
        taken_turn={"name": 1},
    )


def _live_state(
    script,
    *,
    partial: str,
    messages: list,
    progress: ScriptProgress | None = None,
    turn_kind: str = "client",
) -> dict[str, Any]:
    prog = progress or _name_progress()
    state: dict[str, Any] = {
        "script_id": script.id,
        "script_version": script.version,
        "messages": messages,
        "profile": {},
        "turn": 2,
        "turn_kind": turn_kind,
        "partial_reply": partial,
        "partial_utterance_id": "",
        "partial_is_final": True,
        "last_checked_partial": "",
        "last_checked_utterance_id": "",
    }
    state.update(progress_to_state(prog))
    return state


class _FakeChecker:
    def __init__(self) -> None:
        self.calls: list = []

    async def judge(self, **kwargs):
        self.calls.append(kwargs)
        return CheckerVerdict(reply_usable=True, step_closed=False)


async def test_короткий_диалог_агент_не_зовётся(script, monkeypatch, _offline_farewell):
    """Диалог короче порога — агент не вызывается, флаг не меняется."""
    await _offline_farewell.save(
        "local",
        ConversationContext(conversation_ended=True),
    )
    calls: list[str] = []

    async def fake_decide(reply, *, history=(), agent=None):
        calls.append(reply)
        return FarewellDecision(conversation_ended=False)

    monkeypatch.setattr("graph.checker_graph.decide_farewell", fake_decide)

    text = "Меня зовут Андрей и ещё символов"
    state = _live_state(
        script,
        partial=text,
        messages=[AIMessage(content="Как вас зовут?")],
    )

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", _FakeChecker()),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.farewell_min_messages = 5
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert calls == []
    ctx = out.get("conversation_context") or {}
    assert ctx.get("conversation_ended") is True
    loaded = await _offline_farewell.load("local")
    assert loaded is not None
    assert loaded.conversation_ended is True


async def test_длинный_диалог_агент_пишет_флаг(script, monkeypatch, _offline_farewell):
    """Диалог длиннее порога — агент вызывается, флаг в контексте."""
    await _offline_farewell.save("local", ConversationContext())
    calls: list[str] = []

    async def fake_decide(reply, *, history=(), agent=None):
        calls.append(reply)
        return FarewellDecision(conversation_ended=True)

    monkeypatch.setattr("graph.checker_graph.decide_farewell", fake_decide)

    text = "всё, до свидания, хорошего дня вам"
    state = _live_state(script, partial=text, messages=_long_history(5))

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", _FakeChecker()),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.farewell_min_messages = 5
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert calls == [text]
    ctx = out.get("conversation_context") or {}
    assert ctx.get("conversation_ended") is True


async def test_флаг_остаётся_поднятым_если_агент_вернул_ложь(
    script, monkeypatch, _offline_farewell
):
    """Агент вернул ложь — признак конца разговора не опускается."""
    await _offline_farewell.save(
        "local",
        ConversationContext(conversation_ended=True),
    )

    async def fake_decide(reply, *, history=(), agent=None):
        return FarewellDecision(conversation_ended=False)

    monkeypatch.setattr("graph.checker_graph.decide_farewell", fake_decide)

    text = "а ещё вопрос про рассрочку пожалуйста"
    state = _live_state(script, partial=text, messages=_long_history(6))

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", _FakeChecker()),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.farewell_min_messages = 5
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert out["conversation_context"]["conversation_ended"] is True
    loaded = await _offline_farewell.load("local")
    assert loaded is not None
    assert loaded.conversation_ended is True


async def test_флаг_не_поднимается_если_агент_вернул_ложь(script, monkeypatch, _offline_farewell):
    """Признак не был поднят — ложный вердикт агента его не поднимает."""
    await _offline_farewell.save("local", ConversationContext(conversation_ended=False))

    async def fake_decide(reply, *, history=(), agent=None):
        return FarewellDecision(conversation_ended=False)

    monkeypatch.setattr("graph.checker_graph.decide_farewell", fake_decide)

    text = "а ещё вопрос про рассрочку пожалуйста"
    state = _live_state(script, partial=text, messages=_long_history(6))

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", _FakeChecker()),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.farewell_min_messages = 5
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert out["conversation_context"]["conversation_ended"] is False
    loaded = await _offline_farewell.load("local")
    assert loaded is not None
    assert loaded.conversation_ended is False


@pytest.mark.parametrize("turn_kind", ["continuation", "silence", "pull"])
async def test_продолжение_и_молчание_агент_не_зовётся(
    script, monkeypatch, _offline_farewell, turn_kind: str
):
    """На ходах без реплики человека агент не вызывается, флаг не трогаем."""
    await _offline_farewell.save(
        "local",
        ConversationContext(conversation_ended=True),
    )
    calls: list[str] = []

    async def fake_decide(reply, *, history=(), agent=None):
        calls.append(reply)
        return FarewellDecision(conversation_ended=False)

    monkeypatch.setattr("graph.checker_graph.decide_farewell", fake_decide)

    text = "достаточно длинная реплика для прироста"
    state = _live_state(
        script,
        partial=text,
        messages=_long_history(6),
        turn_kind=turn_kind,
    )

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", _FakeChecker()),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.farewell_min_messages = 5
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert calls == []
    assert out["conversation_context"]["conversation_ended"] is True


async def test_ошибка_агента_не_роняет_узел(script, monkeypatch, _offline_farewell, caplog):
    """Ошибка агента — флаг прежний, в логе строка, узел жив."""
    await _offline_farewell.save(
        "local",
        ConversationContext(conversation_ended=True),
    )

    async def fake_decide(reply, *, history=(), agent=None):
        return None

    monkeypatch.setattr("graph.checker_graph.decide_farewell", fake_decide)

    text = "до свидания и ещё символов для порога"
    state = _live_state(script, partial=text, messages=_long_history(5))

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", _FakeChecker()),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
        caplog.at_level(logging.INFO),
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.farewell_min_messages = 5
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert out["conversation_context"]["conversation_ended"] is True
    done_msgs = [
        r.message
        for r in caplog.records
        if r.name == "graph.progress" and "[live-check|done]" in r.message
    ]
    assert done_msgs
    assert "прощание: ошибка агента" in done_msgs[0]


def test_системное_сообщение_два_случая_конца():
    """Конец — только прощание или отказ продолжать, других случаев нет."""
    assert "Ты решаешь, закончил ли собеседник разговор" in FAREWELL_SYSTEM
    assert "Два случая, других нет" in FAREWELL_SYSTEM
    assert "Он попрощался" in FAREWELL_SYSTEM
    assert "«до свидания»" in FAREWELL_SYSTEM
    assert "Он сказал, что дальше говорить не будет" in FAREWELL_SYSTEM
    assert "«мне пора»" in FAREWELL_SYSTEM
    assert "«я уже отвечал»" in FAREWELL_SYSTEM


def test_системное_сообщение_прощание_парное():
    """Прощание бота даёт конец только если собеседник ответил прощанием."""
    assert "Прощание в телефонном разговоре парное" in FAREWELL_SYSTEM
    assert "ответил прощанием — конец" in FAREWELL_SYSTEM
    assert "молчит или говорит о другом — не конец" in FAREWELL_SYSTEM
    assert "не намерен" not in FAREWELL_SYSTEM


def test_системное_сообщение_без_дал_понять():
    """Старая формулировка «дал понять» в промпте недопустима."""
    assert "дал понять" not in FAREWELL_SYSTEM


def test_системное_сообщение_отказ_не_конец():
    """Отказ от предложения явно перечислен среди того, что не конец."""
    assert "отказ от предложения" in FAREWELL_SYSTEM


def test_системное_сообщение_вопросов_нет_не_конец():
    """«вопросов нет» прямо названо ответом на вопрос бота, а не концом."""
    assert "Не конец разговора, даже если звучит похоже" in FAREWELL_SYSTEM
    assert "«вопросов нет»" in FAREWELL_SYSTEM
    assert "«нет вопросов»" in FAREWELL_SYSTEM
    assert "ответы на вопрос бота, а не завершение разговора" in FAREWELL_SYSTEM


def test_системное_сообщение_сомневаешься_не_конец():
    """Правило умолчания при сомнении сохранено."""
    assert "Сомневаешься — не конец" in FAREWELL_SYSTEM


def test_системное_сообщение_без_благодарности_в_исключениях():
    """«благодарность» убрана из исключений — блокировала «спасибо, до свидания»."""
    assert "благодарность" not in FAREWELL_SYSTEM


def test_описание_поля_согласовано_с_промптом():
    """Описание conversation_ended согласовано с критерием прощания."""
    description = FarewellDecision.model_fields["conversation_ended"].description
    assert description is not None
    assert "прощание прозвучало" in description
    assert "разговор исчерпан" in description
    assert "не намерен" not in description


async def test_decide_передаёт_константу_системного_сообщения():
    """LlmFarewellAgent.decide отдаёт в модель ровно FAREWELL_SYSTEM."""
    captured: list = []

    class _FakeLlm:
        """Заглушка контекстного менеджера get_llm."""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    async def fake_astream_structured(llm, messages, **kwargs):
        captured.extend(messages)
        return {"conversation_ended": False}

    with (
        patch("graph.farewell_agent.get_llm", return_value=_FakeLlm()),
        patch("graph.farewell_agent.astream_structured", side_effect=fake_astream_structured),
    ):
        result = await LlmFarewellAgent().decide("до свидания")

    assert result.conversation_ended is False
    assert len(captured) >= 1
    assert captured[0].content == FAREWELL_SYSTEM


async def test_пустая_реплика_и_пустая_история_без_вызова_модели():
    """Пустая реплика и пустая история → False, модель не зовётся."""
    called = False

    class _FakeLlm:
        async def __aenter__(self):
            nonlocal called
            called = True
            return self

        async def __aexit__(self, *args):
            return None

    async def fake_astream_structured(*args, **kwargs):
        nonlocal called
        called = True
        return {"conversation_ended": True}

    with (
        patch("graph.farewell_agent.get_llm", return_value=_FakeLlm()),
        patch("graph.farewell_agent.astream_structured", side_effect=fake_astream_structured),
    ):
        result = await LlmFarewellAgent().decide("   ")

    assert result.conversation_ended is False
    assert called is False


#: Хвост с прощанием бота: конец только если собеседник ответил прощанием.
_ХВОСТ_ПРОЩАНИЕ_БОТА = [
    AIMessage(content="Тогда всего доброго, хорошего дня."),
]

#: Середина презентации: бот спросил про практику, цены и записи ещё не было.
_СЕРЕДИНА_ПРЕЗЕНТАЦИИ = [
    AIMessage(content="Учиться будете сами или узнаёте для кого-то?"),
    HumanMessage(content="Сам."),
    AIMessage(content="Практика проходит на учебной площадке. Вопросы есть?"),
]


class _CapturingLlm:
    """Заглушка контекстного менеджера get_llm."""

    async def __aenter__(self) -> _CapturingLlm:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


async def _decide_offline(
    reply: str,
    history: Sequence[BaseMessage],
    *,
    verdict: bool,
) -> tuple[FarewellDecision | None, str]:
    """Гоняет агента через ``decide_farewell`` офлайн с подменённой моделью.

    Args:
        reply: текущая реплика собеседника.
        history: хвост диалога.
        verdict: что возвращает подменённая модель.

    Returns:
        Решение агента и текст запроса, ушедшего в модель.
    """
    captured: list[BaseMessage] = []

    async def fake_astream_structured(
        llm: object,
        messages: Sequence[BaseMessage],
        **kwargs: object,
    ) -> dict[str, bool]:
        captured.extend(messages)
        return {"conversation_ended": verdict}

    with (
        patch("graph.farewell_agent.get_llm", return_value=_CapturingLlm()),
        patch("graph.farewell_agent.astream_structured", side_effect=fake_astream_structured),
    ):
        decision = await decide_farewell(reply, history=history)
    human = str(captured[-1].content) if captured else ""
    return decision, human


@pytest.mark.parametrize(
    "reply",
    [
        "до свидания",
        "Мне пора, извините.",
        "Я уже отвечал.",
    ],
)
async def test_реплика_и_хвост_доходят_до_модели_признак_возвращается(reply: str):
    """Реплика и хвост уходят в модель, решение модели становится признаком."""
    decision, human = await _decide_offline(
        reply,
        _ХВОСТ_ПРОЩАНИЕ_БОТА,
        verdict=True,
    )

    assert decision is not None
    assert decision.conversation_ended is True
    assert reply in human
    assert "всего доброго" in human


@pytest.mark.parametrize(
    "reply",
    [
        "Давайте на среду в три",
        "Записывайте меня",
        "А рассрочка без процентов?",
        "Вторая категория мне не нужна",
    ],
)
async def test_деловые_реплики_концом_не_становятся(reply: str):
    """Договорённость, запись, вопрос и отказ признак не поднимают."""
    decision, human = await _decide_offline(
        reply,
        _СЕРЕДИНА_ПРЕЗЕНТАЦИИ,
        verdict=False,
    )

    assert decision is not None
    assert decision.conversation_ended is False
    assert reply in human


async def test_вопросов_нету_в_середине_презентации_не_конец():
    """«Вопросов нету» — ответ на вопрос бота, не завершение разговора."""
    decision, human = await _decide_offline(
        "Вопросов нету.",
        _СЕРЕДИНА_ПРЕЗЕНТАЦИИ,
        verdict=False,
    )

    assert decision is not None
    assert decision.conversation_ended is False
    assert "Вопросов нету." in human
    assert "Вопросы есть?" in human


async def test_ошибка_модели_даёт_none():
    """Исключение модели → None: признак не выдумываем."""

    async def fake_astream_structured(*args: object, **kwargs: object) -> dict[str, bool]:
        raise RuntimeError("таймаут модели")

    with (
        patch("graph.farewell_agent.get_llm", return_value=_CapturingLlm()),
        patch("graph.farewell_agent.astream_structured", side_effect=fake_astream_structured),
    ):
        result = await decide_farewell(
            "до свидания",
            history=[HumanMessage(content="до свидания")],
        )

    assert result is None


def test_системное_сообщение_описывает_конец_без_прощания():
    """Отказ продолжать — второй случай конца; домыслов о состоянии нет."""
    assert "дальше говорить не будет" in FAREWELL_SYSTEM
    assert "«я уже отвечал»" in FAREWELL_SYSTEM
    assert "всё уже узнал" not in FAREWELL_SYSTEM
    assert "отвечает односложно, сам ничего не спрашивает" not in FAREWELL_SYSTEM
    assert "не подхватывает" not in FAREWELL_SYSTEM
    assert "тяготит" not in FAREWELL_SYSTEM


def test_системное_сообщение_держит_границу_живого_разговора():
    """Живой разговор дороже пропущенного прощания; исчерпанность — не критерий."""
    assert "Оборвать живой разговор дороже, чем пропустить прощание" in FAREWELL_SYSTEM
    assert "Сколько тем у бота осталось впереди — не твоё дело" in FAREWELL_SYSTEM
    assert "исчерпанн" not in FAREWELL_SYSTEM


def test_системное_сообщение_без_реплик_образцов():
    """Образцы в кавычках — признаки, не список для дословного сравнения."""
    assert "Это признаки, а не список для дословного сравнения" in FAREWELL_SYSTEM
    assert "те же мысли другими словами — тоже конец" in FAREWELL_SYSTEM


def test_хвост_диалога_длиннее_середины_звонка():
    """Хвост — двенадцать последних сообщений, обрезается с начала."""
    assert _HISTORY_TAIL == 12
    history = [HumanMessage(content=f"реплика {i}") for i in range(20)]
    lines = _format_history(history).splitlines()
    assert len(lines) == 12
    assert lines[0] == "клиент: реплика 8"
    assert lines[-1] == "клиент: реплика 19"


async def test_пустая_реплика_с_историей_идёт_в_модель():
    """Пустая реплика при непустой истории — вызов модели состоялся."""
    called = False

    class _FakeLlm:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    async def fake_astream_structured(*args, **kwargs):
        nonlocal called
        called = True
        return {"conversation_ended": True}

    with (
        patch("graph.farewell_agent.get_llm", return_value=_FakeLlm()),
        patch("graph.farewell_agent.astream_structured", side_effect=fake_astream_structured),
    ):
        result = await LlmFarewellAgent().decide(
            "   ",
            history=[AIMessage(content="И Вам спасибо! Хорошего дня")],
        )

    assert called is True
    assert result.conversation_ended is True


def test_системное_сообщение_граница_темы_и_разговора():
    """Отказ от темы в ответ на вопрос бота — не конец; смотреть на реплику бота."""
    assert "отказ обсуждать тему сейчас в ответ на вопрос бота" in FAREWELL_SYSTEM
    assert "Реплика бота перед ответом видна в диалоге ниже" in FAREWELL_SYSTEM
    assert "по ней и смотреть, на что человек отвечает" in FAREWELL_SYSTEM


def test_системное_сообщение_без_неудобно_говорить_как_признака_конца():
    """«неудобно говорить» больше не признак конца; отказ продолжать разговор на месте."""
    assert "неудобно говорить" not in FAREWELL_SYSTEM
    assert "«мне пора»" in FAREWELL_SYSTEM
    assert "«перезвоните позже»" in FAREWELL_SYSTEM


def test_инициатива_запрещает_разрешение_на_переход_к_теме():
    """В блоке инициативы нельзя спрашивать разрешение перейти к теме."""
    assert "Разрешения на переход к теме тоже спрашивать нельзя" in _INITIATIVE_BLOCK
    assert "«Если хотите, расскажу подробнее?»" in _INITIATIVE_BLOCK
    assert "«Готова рассказать, если интересно?»" in _INITIATIVE_BLOCK
