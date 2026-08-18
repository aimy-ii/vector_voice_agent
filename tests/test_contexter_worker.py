"""Офлайн-тесты фонового воркера контекстера и постановки задачи диспетчером."""

from __future__ import annotations

from typing import Any, Sequence
from unittest.mock import patch

import pytest

from graph.checker_graph import live_check_node
from graph.context import (
    DYN_MISSING,
    DYN_READY,
    DYN_SEARCHING,
    ConversationContext,
)
from graph.context_agent import ContextDecision
from graph.context_store import MemoryContextStore
from graph.contexter import reply_hash
from graph.contexter_worker import contexter_task_node
from script.store import progress_to_state
from tests.test_live_checker import FakeChecker, _name_progress, _state


class _SpyStore(MemoryContextStore):
    """Память с счётчиком обращений — пустая задача не должна ходить в кеш."""

    def __init__(self) -> None:
        super().__init__()
        self.loads: int = 0
        self.saves: int = 0

    async def load(self, call_id: str) -> ConversationContext | None:
        self.loads += 1
        return await super().load(call_id)

    async def save(self, call_id: str, context: ConversationContext) -> bool:
        self.saves += 1
        return await super().save(call_id, context)


class _CityStub:
    """Заглушка инструмента города: сеть не трогает, слаг ставит по флагу."""

    name = "city"
    description = "город"

    def __init__(self, *, answer: str = "", set_slug: bool = False) -> None:
        self.answer = answer
        self.set_slug = set_slug
        self.calls: list[str] = []

    async def run(
        self,
        query: str,
        context: ConversationContext,
        *,
        slugs: Sequence[str] = (),
        reply: str = "",
    ) -> str:
        _ = (slugs, reply)
        self.calls.append(query)
        if self.set_slug:
            context.city_slug = "perm"
            context.city_name = "Пермь"
        return self.answer


class _StubScript:
    """Скрипт-заглушка: возражений нет, инструменты подменяются снаружи."""

    objections: dict[str, Any] = {}


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> MemoryContextStore:
    """Кеш воркера в памяти процесса."""
    mem = MemoryContextStore()
    monkeypatch.setattr("graph.contexter_worker.context_store", mem)
    monkeypatch.setattr("graph.contexter.context_store", mem)
    return mem


def _stub_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет загрузку скрипта заглушкой с пустыми возражениями."""
    monkeypatch.setattr(
        "graph.contexter_worker.registry.get",
        lambda *_args, **_kwargs: _StubScript(),
    )


def _task(
    *,
    call_id: str = "call-1",
    reply: str = "Я из Перми",
    needs: list[str] | None = None,
    profile: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Собирает минимальную задачу воркеру."""
    return {
        "call_id": call_id,
        "reply": reply,
        "needs": needs if needs is not None else ["city_choices"],
        "step_needs": [],
        "profile": profile if profile is not None else {"city": "Пермь"},
        "script_id": "vector_ru",
        "script_version": "2",
    }


async def test_сквозной_город_в_кеше_итоговый_статус(
    store: MemoryContextStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Инструмент города ставит слаг: в кеше слаг, динамика и итоговый статус."""
    stub = _CityStub(answer="Город: Пермь.", set_slug=True)
    _stub_script(monkeypatch)
    monkeypatch.setattr(
        "graph.contexter_worker.build_context_tools",
        lambda _script: [stub],
    )

    async def _no_need(*_args: object, **_kwargs: object) -> ContextDecision:
        return ContextDecision(need=False)

    monkeypatch.setattr("graph.contexter.decide_context", _no_need)
    await store.save("call-1", ConversationContext())

    result = await contexter_task_node(_task())

    assert result == {}
    assert stub.calls == ["Пермь"]
    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.city_slug == "perm"
    assert "Город: Пермь." in loaded.dynamic_text
    assert loaded.dynamic_status == DYN_READY
    assert loaded.dynamic_status != DYN_SEARCHING


async def test_повтор_реплики_не_зовёт_контекстер(
    store: MemoryContextStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Совпадение last_reply_hash — run_contexter не вызван, кеш не изменён."""
    reply = "Я из Перми"
    original = ConversationContext(
        last_reply_hash=reply_hash(reply),
        dynamic_text="уже разобрано",
        dynamic_status=DYN_READY,
        city_slug="perm",
    )
    await store.save("call-1", original)

    calls: list[str] = []

    async def _counted(*_args: object, **_kwargs: object) -> ConversationContext:
        calls.append("run")
        return ConversationContext(dynamic_text="не должен записаться")

    monkeypatch.setattr("graph.contexter_worker.run_contexter", _counted)

    await contexter_task_node(_task(reply=reply))

    assert calls == []
    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.dynamic_text == "уже разобрано"
    assert loaded.city_slug == "perm"
    assert loaded.dynamic_status == DYN_READY
    assert loaded.last_reply_hash == reply_hash(reply)


async def test_завершённый_разговор_без_контекстера(
    store: MemoryContextStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """conversation_ended в кеше — выход без вызова run_contexter."""
    await store.save("call-1", ConversationContext(conversation_ended=True, city_slug="perm"))

    calls: list[str] = []

    async def _counted(*_args: object, **_kwargs: object) -> ConversationContext:
        calls.append("run")
        return ConversationContext()

    monkeypatch.setattr("graph.contexter_worker.run_contexter", _counted)

    result = await contexter_task_node(_task())

    assert result == {}
    assert calls == []
    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.conversation_ended is True
    assert loaded.city_slug == "perm"


async def test_пустая_задача_не_ходит_в_кеш(monkeypatch: pytest.MonkeyPatch) -> None:
    """Нет call_id или пустая реплика — без обращений к кешу."""
    spy = _SpyStore()
    monkeypatch.setattr("graph.contexter_worker.context_store", spy)

    calls: list[str] = []

    async def _counted(*_args: object, **_kwargs: object) -> ConversationContext:
        calls.append("run")
        return ConversationContext()

    monkeypatch.setattr("graph.contexter_worker.run_contexter", _counted)

    assert await contexter_task_node(_task(call_id="", reply="Пермь")) == {}
    assert await contexter_task_node(_task(call_id="call-1", reply="")) == {}
    assert await contexter_task_node(_task(call_id="  ", reply="   ")) == {}

    assert spy.loads == 0
    assert spy.saves == 0
    assert calls == []


@pytest.mark.parametrize(
    ("dynamic_text", "expected"),
    [
        ("нашли данные", DYN_READY),
        ("", DYN_MISSING),
    ],
)
async def test_зависший_поиск_доводится_до_итога(
    store: MemoryContextStore,
    monkeypatch: pytest.MonkeyPatch,
    dynamic_text: str,
    expected: str,
) -> None:
    """После разбора статус «в поиске» сменяется итогом по наличию динамики."""
    _stub_script(monkeypatch)

    async def _stuck(
        context: ConversationContext,
        **_kwargs: object,
    ) -> ConversationContext:
        return context.model_copy(
            update={
                "dynamic_status": DYN_SEARCHING,
                "dynamic_text": dynamic_text,
                "situation_slug": "город и условия в нём",
            }
        )

    monkeypatch.setattr("graph.contexter_worker.run_contexter", _stuck)
    await store.save(
        "call-1",
        ConversationContext(dynamic_status=DYN_SEARCHING, situation_slug="город"),
    )

    await contexter_task_node(_task())

    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.dynamic_status == expected
    assert loaded.situation_slug is None
    assert loaded.dynamic_text == dynamic_text


async def test_параллельная_запись_не_затирает_чужой_текст(
    store: MemoryContextStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Текст динамики, появившийся между чтением и записью, остаётся в кеше."""
    _stub_script(monkeypatch)
    await store.save(
        "call-1",
        ConversationContext(dynamic_text="Уже было от хода."),
    )

    async def _with_concurrent(
        context: ConversationContext,
        **_kwargs: object,
    ) -> ConversationContext:
        cached = await store.load("call-1")
        assert cached is not None
        other = cached.model_copy(deep=True)
        other.dynamic_text = (other.dynamic_text + "\nЧужой блок.").strip()
        await store.save("call-1", other)
        return context.model_copy(
            update={
                "dynamic_text": "Локальный разбор",
                "dynamic_status": DYN_READY,
            }
        )

    monkeypatch.setattr("graph.contexter_worker.run_contexter", _with_concurrent)

    await contexter_task_node(_task())

    loaded = await store.load("call-1")
    assert loaded is not None
    assert "Уже было от хода." in loaded.dynamic_text
    assert "Чужой блок." in loaded.dynamic_text
    assert "Локальный разбор" in loaded.dynamic_text
    assert loaded.dynamic_status == DYN_READY


async def test_завершение_во_время_разбора_ничего_не_пишет(
    store: MemoryContextStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Перед записью в кеше conversation_ended — воркер результат не сохраняет."""
    _stub_script(monkeypatch)
    await store.save(
        "call-1",
        ConversationContext(dynamic_text="исходный", city_slug=""),
    )

    async def _end_midway(
        context: ConversationContext,
        **_kwargs: object,
    ) -> ConversationContext:
        cached = await store.load("call-1")
        assert cached is not None
        ended = cached.model_copy(update={"conversation_ended": True, "dynamic_text": "исходный"})
        await store.save("call-1", ended)
        return context.model_copy(
            update={
                "dynamic_text": "не должен записаться",
                "dynamic_status": DYN_READY,
                "city_slug": "perm",
            }
        )

    monkeypatch.setattr("graph.contexter_worker.run_contexter", _end_midway)

    result = await contexter_task_node(_task())

    assert result == {}
    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.conversation_ended is True
    assert loaded.dynamic_text == "исходный"
    assert not (loaded.city_slug or "").strip()


async def _run_live_check(script: Any, monkeypatch: pytest.MonkeyPatch) -> MemoryContextStore:
    """Гоняет live_check_node с офлайн-заглушками судьи, анкеты и прощания."""
    from graph.farewell_agent import FarewellDecision
    from graph.profile_agent import ProfileGuess

    mem = MemoryContextStore()
    monkeypatch.setattr("graph.nodes.context_store", mem)
    await mem.save("local", ConversationContext())

    progress = _name_progress()
    text = "пока ещё думаю над ответом длинный"
    state = _state(script, partial=text, progress=progress, last_checked="")

    async def fake_load(_state: object) -> object:
        return progress

    async def fake_save(prog: object, *, persist_state: bool = True, fields: object = None) -> dict:
        _ = (persist_state, fields)
        return progress_to_state(prog)  # type: ignore[arg-type]

    async def fake_warmup(*_args: object, **kwargs: Any) -> Any:
        return kwargs["ctx"]

    async def _no_profile(*_args: object, **_kwargs: object) -> ProfileGuess:
        return ProfileGuess()

    async def _no_farewell(*_args: object, **_kwargs: object) -> FarewellDecision:
        return FarewellDecision(conversation_ended=False)

    monkeypatch.setattr("graph.contexter_worker.guess_profile", _no_profile)
    monkeypatch.setattr("graph.checker_graph.decide_farewell", _no_farewell)

    with (
        patch("graph.checker_graph._checker_client", FakeChecker([None])),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.farewell_min_messages = 5
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        mock_settings.live_thread_suffix = "-live"
        await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    return mem


async def test_диспетчер_очередь_не_зовёт_контекстер_синхронно(
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Постановка удалась: run_contexter не вызван, в кеше статус «в поиске»."""
    calls: list[str] = []

    async def _queued(*_args: object, **_kwargs: object) -> bool:
        return True

    async def _counted(ctx: ConversationContext, **_kwargs: object) -> ConversationContext:
        calls.append("run")
        return ctx

    monkeypatch.setattr("graph.checker_graph._enqueue_contexter", _queued)
    monkeypatch.setattr("graph.checker_graph.run_contexter", _counted)

    mem = await _run_live_check(script, monkeypatch)

    assert calls == []
    loaded = await mem.load("local")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_SEARCHING


async def test_диспетчер_запасной_путь_зовёт_контекстер(
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Постановка не удалась: run_contexter вызван синхронно."""
    calls: list[str] = []

    async def _missed(*_args: object, **_kwargs: object) -> bool:
        return False

    async def _counted(ctx: ConversationContext, **_kwargs: object) -> ConversationContext:
        calls.append("run")
        return ctx.model_copy(update={"dynamic_status": DYN_READY})

    monkeypatch.setattr("graph.checker_graph._enqueue_contexter", _missed)
    monkeypatch.setattr("graph.checker_graph.run_contexter", _counted)

    mem = await _run_live_check(script, monkeypatch)

    assert calls == ["run"]
    loaded = await mem.load("local")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_READY
