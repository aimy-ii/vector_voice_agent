"""Офлайн-тесты итогового статуса фоновой добычи города."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from graph.city_worker import city_task_node
from graph.context import DYN_MISSING, DYN_READY, DYN_SEARCHING, ConversationContext
from graph.context_store import MemoryContextStore
from graph.contexter import _fulfill_needs


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


async def test_city_task_node_успех_ставит_готово(
    store: MemoryContextStore,
    city_stub: _CityStub,
) -> None:
    """Успех: статус «готово», предмет пуст, заглушка сброшена, слаг стоит."""
    city_stub.answer = "Город: Пермь."
    city_stub.set_slug = True
    await store.save(
        "call-1",
        ConversationContext(
            dynamic_status=DYN_SEARCHING,
            situation_slug="город и условия в нём",
            filler_spoken=True,
        ),
    )

    await city_task_node({"call_id": "call-1", "probe": "Пермь"})

    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_READY
    assert loaded.situation_slug is None
    assert loaded.filler_spoken is False
    assert loaded.city_slug == "perm"


async def test_city_task_node_пусто_ставит_не_нашлось(
    store: MemoryContextStore,
    city_stub: _CityStub,
) -> None:
    """Пустой ответ: статус «не нашлось», предмет пуст, попытка +1."""
    city_stub.answer = ""
    await store.save(
        "call-1",
        ConversationContext(
            dynamic_status=DYN_SEARCHING,
            situation_slug="город и условия в нём",
        ),
    )

    await city_task_node({"call_id": "call-1", "probe": "Пермь"})

    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_MISSING
    assert loaded.situation_slug is None
    assert loaded.city_attempts == 1
    assert not (loaded.city_slug or "").strip()


async def test_city_task_node_пропуск_добытого_снимает_поиск(
    store: MemoryContextStore,
    city_stub: _CityStub,
) -> None:
    """Слаг уже стоит, статус завис «в поиске» — «готово», инструмент не зовём."""
    await store.save(
        "call-1",
        ConversationContext(
            city_slug="perm",
            city_name="Пермь",
            dynamic_status=DYN_SEARCHING,
            situation_slug="город и условия в нём",
        ),
    )

    await city_task_node({"call_id": "call-1", "probe": "Пермь"})

    assert city_stub.calls == []
    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_READY
    assert loaded.situation_slug is None
    assert loaded.city_slug == "perm"


async def test_city_task_node_пропуск_исчерпанных_снимает_поиск(
    store: MemoryContextStore,
    city_stub: _CityStub,
) -> None:
    """Попытки исчерпаны, статус завис «в поиске» — «не нашлось», без инструмента."""
    await store.save(
        "call-1",
        ConversationContext(
            empty_needs=["city_choices"],
            dynamic_status=DYN_SEARCHING,
            situation_slug="город и условия в нём",
        ),
    )

    await city_task_node({"call_id": "call-1", "probe": "Пермь"})

    assert city_stub.calls == []
    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_MISSING
    assert loaded.situation_slug is None


async def test_city_task_node_пустая_задача_не_меняет_статус(
    monkeypatch: pytest.MonkeyPatch,
    city_stub: _CityStub,
) -> None:
    """Пустая задача: кеш не тронут, статус в кеше не менялся."""
    spy = _SpyStore()
    monkeypatch.setattr("graph.city_worker.context_store", spy)
    await spy.save(
        "call-1",
        ConversationContext(
            dynamic_status=DYN_SEARCHING,
            situation_slug="город и условия в нём",
        ),
    )
    spy.loads = 0
    spy.saves = 0

    assert await city_task_node({"call_id": "", "probe": "Пермь"}) == {}
    assert await city_task_node({"call_id": "call-1", "probe": ""}) == {}

    assert spy.loads == 0
    assert spy.saves == 0
    assert city_stub.calls == []
    loaded = await spy.load("call-1")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_SEARCHING
    assert loaded.situation_slug == "город и условия в нём"


async def test_city_task_node_параллельная_запись_сохраняет_текст_и_итог(
    store: MemoryContextStore,
    city_stub: _CityStub,
) -> None:
    """Чужой текст динамики между чтением и записью цел, статус итоговый."""
    await store.save(
        "call-1",
        ConversationContext(
            dynamic_text="Уже было от хода.",
            dynamic_status=DYN_SEARCHING,
            situation_slug="город и условия в нём",
        ),
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
    assert "Уже было от хода." in loaded.dynamic_text
    assert "Чужой блок." in loaded.dynamic_text
    assert loaded.dynamic_status == DYN_READY
    assert loaded.situation_slug is None
    assert loaded.city_slug == "perm"


async def test_fulfill_needs_очередь_ставит_предмет(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Удачная постановка: предмет поиска — город и условия в нём."""

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
    assert context.situation_slug == "город и условия в нём"


async def test_fulfill_needs_очередь_неудачна_предмет_не_ставит(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Неудачная постановка: предмет не ставится, работает синхронный путь."""

    async def _missed(_call_id: str, _probe: str) -> bool:
        return False

    monkeypatch.setattr("graph.contexter._enqueue_city_task", _missed)
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
    assert stub.calls == ["Пермь"]
    assert invoked is True
    assert got is True
    assert context.situation_slug is None
    assert context.city_slug == "perm"
