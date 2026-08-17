"""Прощание в реплике самого бота: локальная проверка и флаг конца разговора."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from graph import nodes as nodes_module
from graph.context import ConversationContext
from graph.context_store import MemoryContextStore
from graph.state import new_state_defaults
from script.store import MemoryScriptStore

#: Реплики бота из живых звонков, после которых разговор действительно кончен.
FAREWELL_REPLIES: tuple[str, ...] = (
    "До свидания, Андрей! Хорошего дня.",
    "Всегда пожалуйста! Хорошего дня, Андрей. До связи!",
    "Спасибо за звонок, всего доброго.",
)

#: Реплики бота из живых звонков, после которых разговор продолжается.
OPEN_REPLIES: tuple[str, ...] = (
    "Если появятся вопросы — пишите, всегда на связи.",
    "Я напомню в Max ближе к дате.",
    "Буду ждать Вашего сообщения.",
    "Жду Вас в офисе на Коломяжском.",
)


@pytest.fixture()
def store(monkeypatch) -> MemoryScriptStore:
    """Кеш прогресса в памяти вместо Redis."""
    mem = MemoryScriptStore()
    monkeypatch.setattr(nodes_module, "script_store", mem)
    return mem


@pytest.fixture()
def ctx_store(monkeypatch) -> MemoryContextStore:
    """Кеш контекста в памяти вместо Redis."""
    mem = MemoryContextStore()
    monkeypatch.setattr(nodes_module, "context_store", mem)
    return mem


@pytest.fixture()
def quiet(monkeypatch) -> None:
    """Гасит вывод прогресса: тестам нужны значения, а не лог."""
    monkeypatch.setattr(nodes_module, "stage", lambda *a, **k: None)


@pytest.fixture()
def use_v2(monkeypatch) -> None:
    """Ход-тесты идут на скрипте v2."""
    monkeypatch.setattr(nodes_module.settings, "script_version", "2")


def _commit_state(reply: str, *, turn_kind: str = "client", **extra: Any) -> dict[str, Any]:
    """Состояние хода, в котором бот произнёс ``reply``."""
    return {
        **new_state_defaults(),
        "messages": [HumanMessage(content="спасибо")],
        "script_id": "vector_ru",
        "script_version": "2",
        "spoken": [reply],
        "turn_result": {"reply": reply},
        "turn_kind": turn_kind,
        "conversation_context": {},
        **extra,
    }


@pytest.mark.parametrize("reply", FAREWELL_REPLIES)
def test_прощание_бота_распознаётся(reply: str):
    """Прощальные обороты из живых звонков считаются концом разговора."""
    assert nodes_module.is_farewell_reply(reply) is True


@pytest.mark.parametrize("reply", OPEN_REPLIES)
def test_обещание_продолжения_не_прощание(reply: str):
    """Обещание написать, напомнить и ждать разговор не заканчивает."""
    assert nodes_module.is_farewell_reply(reply) is False


def test_вопрос_в_конце_не_прощание():
    """Реплика с вопросом ждёт ответа — прощанием не считается."""
    assert nodes_module.is_farewell_reply("Хорошего дня! В каком городе будете учиться?") is False


def test_прощальный_оборот_в_середине_не_считается():
    """Оборот в начале длинной реплики — часть мысли, а не прощание."""
    text = (
        "Спасибо за звонок, я всё записала. Занятия начинаются в понедельник. "
        "Оплату вносят на месте."
    )
    assert nodes_module.is_farewell_reply(text) is False


def test_договорённость_о_встрече_не_прощание():
    """«До встречи» — оборот договорённости; разговор после него продолжается."""
    assert nodes_module.is_farewell_reply("До встречи в офисе на Коломяжском.") is False


def test_пустая_реплика_не_прощание():
    """Пустой эфир флага не поднимает."""
    assert nodes_module.is_farewell_reply("") is False
    assert nodes_module.is_farewell_reply("   ") is False


@pytest.mark.parametrize("reply", FAREWELL_REPLIES)
async def test_прощание_бота_поднимает_флаг(store, ctx_store, quiet, use_v2, reply: str):
    """Прощание в реплике хода поднимает флаг и в состоянии, и в кеше."""
    await ctx_store.save("local", ConversationContext(conversation_ended=False))
    out = await nodes_module.commit_node(_commit_state(reply), None)  # type: ignore[arg-type]
    assert out["conversation_ended"] is True
    assert out["conversation_context"]["conversation_ended"] is True
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert loaded.conversation_ended is True


@pytest.mark.parametrize("reply", OPEN_REPLIES)
async def test_обещание_продолжения_флаг_не_поднимает(store, ctx_store, quiet, use_v2, reply: str):
    """Обещание продолжения оставляет флаг таким, каким его видит кеш."""
    await ctx_store.save("local", ConversationContext(conversation_ended=False))
    out = await nodes_module.commit_node(_commit_state(reply), None)  # type: ignore[arg-type]
    assert out["conversation_ended"] is False
    assert out["conversation_context"]["conversation_ended"] is False


@pytest.mark.parametrize("turn_kind", ["continuation", "silence", "pull"])
async def test_прощание_поднимает_флаг_и_без_реплики_человека(
    store, ctx_store, quiet, use_v2, turn_kind: str
):
    """На ходах без реплики человека флаг заморожен — прощание бота его размораживает."""
    await ctx_store.save("local", ConversationContext(conversation_ended=False))
    state = _commit_state("Хорошего дня, Андрей. До связи!", turn_kind=turn_kind)
    out = await nodes_module.commit_node(state, None)  # type: ignore[arg-type]
    assert out["conversation_ended"] is True
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert loaded.conversation_ended is True


@pytest.mark.parametrize("turn_kind", ["continuation", "silence", "pull"])
async def test_деловая_реплика_без_человека_флаг_не_трогает(
    store, ctx_store, quiet, use_v2, turn_kind: str
):
    """Без прощания на ходе без реплики человека флаг по-прежнему не пишется."""
    await ctx_store.save("local", ConversationContext(conversation_ended=False))
    state = _commit_state("Я напомню в Max ближе к дате.", turn_kind=turn_kind)
    out = await nodes_module.commit_node(state, None)  # type: ignore[arg-type]
    assert "conversation_ended" not in out


async def test_прощание_не_затирает_динамику_контекста(store, ctx_store, quiet, use_v2):
    """Запись флага не трогает соседние поля динамики, которые ведёт лайв-канал."""
    await ctx_store.save(
        "local",
        ConversationContext(dynamic_text="цена уточнена", dynamic_status="готово"),
    )
    await nodes_module.commit_node(
        _commit_state("Спасибо за звонок, всего доброго."),
        None,  # type: ignore[arg-type]
    )
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert loaded.conversation_ended is True
    assert loaded.dynamic_text == "цена уточнена"
    assert loaded.dynamic_status == "готово"
