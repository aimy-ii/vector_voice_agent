"""Офлайн-тесты фоновой добычи города и постановки задачи в очередь."""

from __future__ import annotations

import sys
import types
from typing import Any, Sequence

import pytest

from graph.city_worker import city_task_node
from graph.context import ConversationContext
from graph.context_store import (
    CONTEXT_FIELDS_DYNAMIC,
    CONTEXT_FIELDS_STATIC,
    MemoryContextStore,
    merge_context_fields,
)
from graph.contexter import _enqueue_city_task, _fulfill_needs


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
    """Заглушка инструмента города: сеть не трогает, вызовы запоминает."""

    name = "city"
    description = "город"

    def __init__(self, *, answer: str = "", set_slug: bool = False) -> None:
        self.answer = answer
        self.set_slug = set_slug
        self.calls: list[str] = []
        self.on_run: Any = None

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
        if self.on_run is not None:
            await self.on_run(context)
        if self.set_slug:
            context.city_slug = "perm"
            context.city_name = "Пермь"
        return self.answer


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> MemoryContextStore:
    mem = MemoryContextStore()
    monkeypatch.setattr("graph.city_worker.context_store", mem)
    return mem


@pytest.fixture
def city_stub(monkeypatch: pytest.MonkeyPatch) -> _CityStub:
    stub = _CityStub()

    async def _script() -> object:
        return object()

    monkeypatch.setattr("graph.city_worker._load_script", _script)
    monkeypatch.setattr("graph.city_worker.build_context_tools", lambda _script: [stub])
    return stub


def _patch_get_client(monkeypatch: pytest.MonkeyPatch, impl: Any) -> None:
    """Подменяет ``langgraph_sdk.get_client`` без сети."""
    mod = sys.modules.get("langgraph_sdk")
    if mod is None:
        mod = types.ModuleType("langgraph_sdk")
        monkeypatch.setitem(sys.modules, "langgraph_sdk", mod)
    monkeypatch.setattr(mod, "get_client", impl, raising=False)


async def test_city_task_node_добывает_город_в_кеш(
    store: MemoryContextStore,
    city_stub: _CityStub,
) -> None:
    """Успех: слаг в кеше, динамика дописана, счётчик попыток обнулён."""
    city_stub.answer = "Город: Пермь."
    city_stub.set_slug = True
    await store.save("call-1", ConversationContext(city_attempts=2))

    result = await city_task_node({"call_id": "call-1", "probe": "Пермь"})

    assert result == {}
    assert city_stub.calls == ["Пермь"]
    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.city_slug == "perm"
    assert "Город: Пермь." in loaded.dynamic_text
    assert loaded.city_attempts == 0


async def test_city_task_node_пустой_ответ_считает_попытку(
    store: MemoryContextStore,
    city_stub: _CityStub,
) -> None:
    """Пустой ответ: попытка +1, слага нет, empty_needs ещё пуст."""
    city_stub.answer = ""
    await store.save("call-1", ConversationContext())

    await city_task_node({"call_id": "call-1", "probe": "Пермь"})

    assert city_stub.calls == ["Пермь"]
    loaded = await store.load("call-1")
    assert loaded is not None
    assert not (loaded.city_slug or "").strip()
    assert loaded.city_attempts == 1
    assert loaded.empty_needs == []


async def test_city_task_node_пропускает_уже_добытый_город(
    store: MemoryContextStore,
    city_stub: _CityStub,
) -> None:
    """Свежесть по кешу: слаг уже стоит — инструмент не зовём."""
    await store.save("call-1", ConversationContext(city_slug="perm", city_name="Пермь"))

    await city_task_node({"call_id": "call-1", "probe": "Пермь"})

    assert city_stub.calls == []


async def test_city_task_node_пропускает_исчерпанные_попытки(
    store: MemoryContextStore,
    city_stub: _CityStub,
) -> None:
    """city_choices в empty_needs — инструмент не зовём."""
    await store.save("call-1", ConversationContext(empty_needs=["city_choices"]))

    await city_task_node({"call_id": "call-1", "probe": "Пермь"})

    assert city_stub.calls == []


async def test_city_task_node_пустая_задача_не_ходит_в_кеш(
    monkeypatch: pytest.MonkeyPatch,
    city_stub: _CityStub,
) -> None:
    """Пустой call_id или probe — без обращений к кешу и инструменту."""
    spy = _SpyStore()
    monkeypatch.setattr("graph.city_worker.context_store", spy)

    assert await city_task_node({"call_id": "", "probe": "Пермь"}) == {}
    assert await city_task_node({"call_id": "call-1", "probe": ""}) == {}
    assert await city_task_node({"call_id": "  ", "probe": "   "}) == {}

    assert spy.loads == 0
    assert spy.saves == 0
    assert city_stub.calls == []


async def test_city_task_node_не_затирает_чужую_динамику(
    store: MemoryContextStore,
    city_stub: _CityStub,
) -> None:
    """Текст динамики другого писателя между чтением и записью сохраняется."""
    await store.save(
        "call-1",
        ConversationContext(dynamic_text="Уже было от хода."),
    )
    city_stub.answer = "Город: Пермь."
    city_stub.set_slug = True

    async def _concurrent(_context: ConversationContext) -> None:
        cached = await store.load("call-1")
        assert cached is not None
        other = cached.model_copy(deep=True)
        other.dynamic_text = (other.dynamic_text + "\nЧужой блок.").strip()
        await store.save("call-1", other)

    city_stub.on_run = _concurrent

    await city_task_node({"call_id": "call-1", "probe": "Пермь"})

    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.city_slug == "perm"
    assert "Уже было от хода." in loaded.dynamic_text
    assert "Чужой блок." in loaded.dynamic_text
    assert "Город: Пермь." in loaded.dynamic_text

    # Тот же принцип, что у узла: свежий base + overlay не режет чужой текст.
    base = ConversationContext(dynamic_text="Уже было от хода.\nЧужой блок.")
    overlay = ConversationContext(
        city_slug="perm",
        dynamic_text="Уже было от хода.\nГород: Пермь.",
    )
    concurrent = (base.dynamic_text or "").strip()
    local = (overlay.dynamic_text or "").strip()
    if concurrent and concurrent not in (overlay.dynamic_text or ""):
        overlay.dynamic_text = f"{concurrent}\n{local}".strip()
    merged = merge_context_fields(
        base,
        overlay,
        CONTEXT_FIELDS_STATIC | CONTEXT_FIELDS_DYNAMIC,
    )
    assert "Чужой блок." in merged.dynamic_text
    assert merged.city_slug == "perm"


async def test_enqueue_недоступен_идем_синхронно(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Клиент очереди падает — False, _fulfill_needs зовёт инструмент сам."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("нет клиента")

    _patch_get_client(monkeypatch, _boom)
    monkeypatch.setattr("graph.contexter._call_id", lambda: "call-1")

    assert await _enqueue_city_task("call-1", "Пермь") is False

    stub = _CityStub(answer="Город: Пермь.", set_slug=True)
    context = ConversationContext()
    got, invoked = await _fulfill_needs(
        context,
        reply="Пермь",
        needs=["city_choices"],
        tools=[stub],
        profile={"city": "Пермь"},
    )
    assert stub.calls == ["Пермь"]
    assert invoked is True
    assert got is True


async def test_fulfill_needs_при_очереди_инструмент_не_зовёт(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Удачная постановка: синхронный инструмент не вызывается, invoked истинный."""

    async def _queued(_call_id: str, _probe: str) -> bool:
        return True

    monkeypatch.setattr("graph.contexter._enqueue_city_task", _queued)
    monkeypatch.setattr("graph.contexter._call_id", lambda: "call-1")

    stub = _CityStub(answer="Город: Пермь.", set_slug=True)
    context = ConversationContext()
    got, invoked = await _fulfill_needs(
        context,
        reply="Пермь",
        needs=["city_choices"],
        tools=[stub],
        profile={"city": "Пермь"},
    )
    assert stub.calls == []
    assert invoked is True
    assert got is False
    assert context.city_attempts == 0
    assert context.empty_needs == []
