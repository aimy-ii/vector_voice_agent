"""Тесты агента конца разговора и его встраивания в лайв-канал."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph.checker import CheckerVerdict
from graph.checker_graph import live_check_node
from graph.context import ConversationContext
from graph.context_agent import ContextDecision
from graph.context_store import MemoryContextStore
from graph.farewell_agent import (
    FAREWELL_SYSTEM,
    FarewellDecision,
    LlmFarewellAgent,
    decide_farewell,
)
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
        self.history_seen: list = []

    async def decide(self, reply: str, history=()) -> FarewellDecision:
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
    monkeypatch.setattr("graph.checker_graph.guess_profile", _no_profile)
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


async def test_флаг_переставляется_обратно_в_ложь(script, monkeypatch, _offline_farewell):
    """Агент вернул ложь — флаг снова False, необратимости нет."""
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

    assert out["conversation_context"]["conversation_ended"] is False
    loaded = await _offline_farewell.load("local")
    assert loaded is not None
    assert loaded.conversation_ended is False


@pytest.mark.parametrize("turn_kind", ["continuation", "silence"])
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


def test_системное_сообщение_прощание_бота_и_ответ_или_молчание():
    """Прощание бота засчитывается только с ответом или молчанием собеседника."""
    assert "попрощался бот" in FAREWELL_SYSTEM
    assert "ответил тем же" in FAREWELL_SYSTEM
    assert "или молчит" in FAREWELL_SYSTEM
    assert "не намерен" not in FAREWELL_SYSTEM


def test_системное_сообщение_без_дал_понять():
    """Старая формулировка «дал понять» в промпте недопустима."""
    assert "дал понять" not in FAREWELL_SYSTEM


def test_системное_сообщение_отказ_не_конец():
    """Отказ от предложения явно перечислен среди того, что не конец."""
    assert "отказ от предложения" in FAREWELL_SYSTEM


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
