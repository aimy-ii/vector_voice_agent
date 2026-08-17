"""Офлайн-тесты раздела про обрыв реплики бота."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph import nodes as nodes_module
from graph.context import ConversationContext
from graph.history import text_of
from graph.prompts import (
    build_filler_messages,
    build_pull_messages,
    build_turn_messages,
    build_waiting_messages,
    interrupted_reply_block,
)
from graph.state import new_state_defaults
from graph.transcript import ROLE_AGENT, ROLE_CLIENT, TranscriptEntry

#: Услышанный кусок — уникальная фраза, которой нет в скрипте и персоне.
_HEARD = "Набор в группу Камышина ещё идёт"
#: Неуслышанный хвост той же реплики.
_UNHEARD = ", занятия начинаются в сентябре."
_FULL_REPLY = _HEARD + _UNHEARD

#: Предписания, которых в разделе обрыва быть не должно.
_ALWAYS_FINISH = (
    "всегда договаривай",
    "договаривай, если",
    "всегда довести",
    "обязана договорить",
    "обязаны договорить",
)


def _prompt_text(messages: list[Any]) -> str:
    """Склеивает все сообщения сборки в одну строку."""
    return "\n".join(str(message.content) for message in messages)


def _turn_kwargs(script: Any, **extra: Any) -> dict[str, Any]:
    """Общие аргументы полной сборки хода."""
    kwargs: dict[str, Any] = {
        "script": script,
        "steps": [script.step("city")],
        "profile": {},
        "facts": {},
        "history": [HumanMessage(content="алло")],
        "asides_done": [],
    }
    kwargs.update(extra)
    return kwargs


def _pull_kwargs(script: Any, **extra: Any) -> dict[str, Any]:
    """Общие аргументы короткой сборки вытаскивания."""
    kwargs: dict[str, Any] = {
        "messages": [HumanMessage(content="алло")],
        "profile": {},
        "step": script.step("city"),
        "facts": {},
        "context_text": "",
    }
    kwargs.update(extra)
    return kwargs


def _respond(script: Any, **extra: Any) -> list[Any]:
    """Дёргает точку выбора сборки с общими аргументами."""
    kwargs: dict[str, Any] = {
        "prompt_kind": "full",
        "script": script,
        "state": new_state_defaults(),
        "history": [HumanMessage(content="алло")],
        "profile": {},
        "facts": {},
        "lead": script.step("city"),
        "head": [script.step("city")],
        "context_text": "",
        "dynamic_status": "",
        "pending_fields": [],
        "turn_kind": "client",
    }
    kwargs.update(extra)
    return nodes_module._build_respond_messages(**kwargs)


@pytest.fixture()
def ctx_store(monkeypatch):
    from graph.context_store import MemoryContextStore

    mem = MemoryContextStore()
    monkeypatch.setattr(nodes_module, "context_store", mem)
    return mem


@pytest.fixture()
def quiet(monkeypatch) -> None:
    monkeypatch.setattr(nodes_module, "stage", lambda *a, **k: None)


def _runtime(*, interrupted_reply: str = "", turn_kind: str = "client") -> SimpleNamespace:
    context: dict[str, str] = {}
    if interrupted_reply:
        context["interrupted_reply"] = interrupted_reply
    return SimpleNamespace(context=context, configurable={"turn_kind": turn_kind})


def _base_state(*, messages: list[Any], turn: int = 2) -> dict[str, Any]:
    return {
        **new_state_defaults(),
        "messages": messages,
        "script_id": "vector_ru",
        "script_version": "2",
        "turn": turn,
    }


@pytest.fixture()
def turn_kind(monkeypatch):
    """Подменяет ``_turn_kind`` значением из аргумента теста."""

    def _set(kind: str) -> None:
        monkeypatch.setattr(nodes_module, "_turn_kind", lambda: kind)

    return _set


def test_раздел_обрыва_с_непустым_значением_содержит_услышанное():
    """Непустой кусок попадает в текст раздела целиком."""
    block = interrupted_reply_block(_HEARD)
    assert block
    assert _HEARD in block
    assert "прервана" in block
    assert "не дослушав" in block
    assert "Остального он не слышал" in block
    assert "решать по разговору" in block


def test_раздел_обрыва_пустое_и_none_дают_пустую_строку():
    """Пустая строка, пробелы и None — раздела нет."""
    assert interrupted_reply_block("") == ""
    assert interrupted_reply_block("   ") == ""
    assert interrupted_reply_block(None) == ""


def test_раздел_обрыва_без_предписания_всегда_договаривать():
    """В разделе нет правил, которые решают за модель."""
    block = interrupted_reply_block(_HEARD).lower()
    for phrase in _ALWAYS_FINISH:
        assert phrase not in block, phrase


def test_полная_сборка_с_обрывом_содержит_раздел(script):
    """build_turn_messages с непустым значением кладёт раздел обрыва."""
    messages = build_turn_messages(**_turn_kwargs(script, interrupted_reply=_HEARD))
    content = messages[0].content
    assert "# РЕПЛИКА ПРЕРВАНА" in content
    assert interrupted_reply_block(_HEARD) in content
    assert _HEARD in content


def test_полная_сборка_без_обрыва_символ_в_символ(script):
    """Без значения системное сообщение совпадает с прежним символ в символ."""
    base = build_turn_messages(**_turn_kwargs(script))
    empty = build_turn_messages(**_turn_kwargs(script, interrupted_reply=""))
    blanks = build_turn_messages(**_turn_kwargs(script, interrupted_reply="   "))
    assert base[0].content == empty[0].content == blanks[0].content
    assert [message.content for message in base] == [message.content for message in empty]


def test_вытаскивание_с_обрывом_содержит_раздел(script):
    """build_pull_messages с непустым значением кладёт текст обрыва."""
    messages = build_pull_messages(script, **_pull_kwargs(script, interrupted_reply=_HEARD))
    content = messages[0].content
    assert interrupted_reply_block(_HEARD) in content
    assert _HEARD in content


def test_вытаскивание_без_обрыва_символ_в_символ(script):
    """Без значения системное сообщение вытаскивания совпадает с прежним."""
    base = build_pull_messages(script, **_pull_kwargs(script))
    empty = build_pull_messages(script, **_pull_kwargs(script, interrupted_reply=""))
    blanks = build_pull_messages(script, **_pull_kwargs(script, interrupted_reply="   "))
    assert base[0].content == empty[0].content == blanks[0].content
    assert [message.content for message in base] == [message.content for message in empty]


def test_заглушка_и_ожидание_раздела_обрыва_не_содержат(script):
    """Служебные сборки не получают раздел ни при каком значении."""
    heard_state = {**new_state_defaults(), "interrupted_reply": _HEARD}
    filler = _respond(script, prompt_kind="filler", state=heard_state)
    waiting = _respond(script, prompt_kind="waiting", state=heard_state)
    filler_direct = build_filler_messages(
        script,
        messages=[HumanMessage(content="алло")],
        history_limit=2,
    )
    waiting_direct = build_waiting_messages(
        script,
        messages=[HumanMessage(content="алло")],
        profile={},
        pending_fields=[],
        step=script.step("city"),
        history_limit=2,
    )
    for messages in (filler, waiting, filler_direct, waiting_direct):
        text = _prompt_text(messages)
        assert _HEARD not in text
        assert "РЕПЛИКА ПРЕРВАНА" not in text
        assert "прервана" not in text


def test_услышанный_кусок_в_промпте_один_раз(script):
    """Кусок есть в разделе обрыва и не повторяется хвостом истории."""
    history = [
        AIMessage(content=_FULL_REPLY),
        HumanMessage(content="подождите, а в кредит можно?"),
    ]
    turn = build_turn_messages(**_turn_kwargs(script, history=history, interrupted_reply=_HEARD))
    pull = build_pull_messages(
        script,
        **_pull_kwargs(script, messages=history, interrupted_reply=_HEARD),
    )
    for messages in (turn, pull):
        text = _prompt_text(messages)
        assert text.count(_HEARD) == 1
        assert _UNHEARD.strip() not in text
        assert interrupted_reply_block(_HEARD) in messages[0].content


def test_раздел_обрыва_стоит_перед_текущим_шагом(script):
    """В полной сборке обрыв — после контекста и перед текущим шагом."""
    messages = build_turn_messages(
        **_turn_kwargs(
            script,
            context_text="Город: Пермь",
            interrupted_reply=_HEARD,
        )
    )
    content = messages[0].content
    assert content.index("# КОНТЕКСТ") < content.index("# РЕПЛИКА ПРЕРВАНА")
    assert content.index("# РЕПЛИКА ПРЕРВАНА") < content.index("# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ")


async def test_поле_читается_из_параметров_запуска_и_доходит_до_сборки(
    script, ctx_store, quiet, turn_kind
):
    """Параметры запуска доходят до сборки: ingest кладёт кусок в состояние."""
    turn_kind("client")
    await ctx_store.save(
        "local",
        ConversationContext(
            transcript=[
                TranscriptEntry(entry_id="agent:1:0", role=ROLE_AGENT, text=_FULL_REPLY),
            ],
        ),
    )
    state = _base_state(messages=[HumanMessage(content="подождите")])
    out = await nodes_module.ingest_node(
        state,
        _runtime(interrupted_reply=_HEARD),  # type: ignore[arg-type]
    )
    assert out["interrupted_reply"] == _HEARD
    assert _HEARD in [text_of(message) for message in out["messages"]]
    assert _UNHEARD.strip() not in " ".join(text_of(message) for message in out["messages"])

    assembled = _respond(
        script,
        state={**new_state_defaults(), "interrupted_reply": out["interrupted_reply"]},
        history=out["messages"],
    )
    text = _prompt_text(assembled)
    assert interrupted_reply_block(_HEARD) in assembled[0].content
    assert text.count(_HEARD) == 1


async def test_пустой_снимок_с_обрывом_урезает_историю_до_услышанного(ctx_store, quiet, turn_kind):
    """Пустой снимок сверку не включает, но ключ обрыва приводит историю к услышанному."""
    turn_kind("client")
    await ctx_store.save(
        "local",
        ConversationContext(
            transcript=[
                TranscriptEntry(entry_id="agent:1:0", role=ROLE_AGENT, text=_FULL_REPLY),
            ],
        ),
    )
    state = _base_state(messages=[HumanMessage(content="подождите")])
    out = await nodes_module.ingest_node(
        state,
        _runtime(interrupted_reply=_HEARD),  # type: ignore[arg-type]
    )
    texts = [text_of(message) for message in out["messages"]]
    assert texts[0] == _HEARD
    assert texts[-1] == "подождите"
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert loaded.transcript[0].text == _HEARD
    assert loaded.transcript[0].role == ROLE_AGENT


async def test_снимок_с_префиксом_и_обрывом_не_задваивает_историю(ctx_store, quiet, turn_kind):
    """Сверка уже урезала запись — ключ обрыва историю второй раз не меняет."""
    turn_kind("client")
    await ctx_store.save(
        "local",
        ConversationContext(
            transcript=[
                TranscriptEntry(entry_id="agent:1:0", role=ROLE_AGENT, text=_FULL_REPLY),
            ],
        ),
    )
    state = _base_state(messages=[AIMessage(content=_HEARD), HumanMessage(content="стоп")])
    out = await nodes_module.ingest_node(
        state,
        _runtime(interrupted_reply=_HEARD),  # type: ignore[arg-type]
    )
    agent_texts = [
        text_of(message) for message in out["messages"] if isinstance(message, AIMessage)
    ]
    assert agent_texts == [_HEARD]
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert [item.role for item in loaded.transcript] == [ROLE_AGENT, ROLE_CLIENT]
    assert loaded.transcript[0].text == _HEARD


async def test_без_ключа_обрыва_пустой_снимок_историю_не_трогает(ctx_store, quiet, turn_kind):
    """Нет ключа — полная реплика в истории остаётся, как и раньше."""
    turn_kind("client")
    await ctx_store.save(
        "local",
        ConversationContext(
            transcript=[
                TranscriptEntry(entry_id="agent:1:0", role=ROLE_AGENT, text=_FULL_REPLY),
            ],
        ),
    )
    state = _base_state(messages=[HumanMessage(content="алло")])
    out = await nodes_module.ingest_node(state, _runtime())  # type: ignore[arg-type]
    assert out["interrupted_reply"] == ""
    assert [text_of(message) for message in out["messages"]] == [_FULL_REPLY, "алло"]
