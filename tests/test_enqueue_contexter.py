"""Офлайн-тесты постановки разбора реплики в очередь контекстера."""

from __future__ import annotations

from typing import Any

import pytest

from graph.checker_graph import _CONTEXTER_THREADS, _enqueue_contexter

_CALL_ID = "call-1"
_CREATED_THREAD = "11111111-1111-1111-1111-111111111111"
_FOUND_THREAD = "22222222-2222-2222-2222-222222222222"


class _FakeThreads:
    """Поддельные треды SDK: поиск и создание без сети."""

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.search_result: list[dict[str, str]] = []
        self.created_thread_id: str = _CREATED_THREAD

    async def search(self, **kwargs: Any) -> list[dict[str, str]]:
        self.search_calls.append(kwargs)
        return list(self.search_result)

    async def create(self, **kwargs: Any) -> dict[str, str]:
        self.create_calls.append(kwargs)
        return {"thread_id": self.created_thread_id}


class _FakeRuns:
    """Поддельные запуски SDK: постановка рана без сети."""

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.create_error: Exception | None = None

    async def create(self, thread_id: str, assistant_id: str, **kwargs: Any) -> dict[str, str]:
        self.create_calls.append({"thread_id": thread_id, "assistant_id": assistant_id, **kwargs})
        if self.create_error is not None:
            raise self.create_error
        return {}


class _FakeClient:
    """Поддельный клиент LangGraph SDK с тредыми и запусками."""

    def __init__(self) -> None:
        self.threads = _FakeThreads()
        self.runs = _FakeRuns()


@pytest.fixture(autouse=True)
def _clear_thread_cache() -> None:
    """Чистит кеш тредов между тестами, чтобы они не зависели друг от друга."""
    _CONTEXTER_THREADS.clear()
    yield
    _CONTEXTER_THREADS.clear()


def _install_client(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    """Подменяет get_client модуля langgraph_sdk поддельным клиентом."""
    monkeypatch.setattr("langgraph_sdk.get_client", lambda: client)


def _task_kwargs() -> dict[str, Any]:
    """Минимальные аргументы постановки, кроме call_id."""
    return {
        "reply": "Я из Перми",
        "needs": ["city_choices"],
        "step_needs": [],
        "profile": {"city": "Пермь"},
        "state": {"script_id": "vector_ru", "script_version": "4"},
    }


async def test_первый_вызов_ищет_создаёт_тред_и_ставит_в_очередь(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустой кеш: поиск по метаданным, создание без thread_id, постановка в UUID."""
    client = _FakeClient()
    _install_client(monkeypatch, client)

    result = await _enqueue_contexter(_CALL_ID, **_task_kwargs())

    assert result is True
    assert client.threads.search_calls == [{"metadata": {"vector_call_id": _CALL_ID}, "limit": 1}]
    assert client.threads.create_calls == [{"metadata": {"vector_call_id": _CALL_ID}}]
    assert "thread_id" not in client.threads.create_calls[0]
    assert client.runs.create_calls[0]["thread_id"] == _CREATED_THREAD
    assert client.runs.create_calls[0]["assistant_id"] == "vector_contexter"
    assert client.runs.create_calls[0]["multitask_strategy"] == "enqueue"


async def test_повторный_вызов_берёт_тред_из_кеша(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Тот же call_id: поиск и создание не вызываются, тред тот же."""
    client = _FakeClient()
    _install_client(monkeypatch, client)

    await _enqueue_contexter(_CALL_ID, **_task_kwargs())
    client.threads.search_calls.clear()
    client.threads.create_calls.clear()
    client.runs.create_calls.clear()

    result = await _enqueue_contexter(_CALL_ID, **_task_kwargs())

    assert result is True
    assert client.threads.search_calls == []
    assert client.threads.create_calls == []
    assert len(client.runs.create_calls) == 1
    assert client.runs.create_calls[0]["thread_id"] == _CREATED_THREAD


async def test_поиск_нашёл_тред_создание_не_нужно(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search вернул тред: create не вызван, задача уходит в найденный."""
    client = _FakeClient()
    client.threads.search_result = [{"thread_id": _FOUND_THREAD}]
    _install_client(monkeypatch, client)

    result = await _enqueue_contexter(_CALL_ID, **_task_kwargs())

    assert result is True
    assert client.threads.create_calls == []
    assert client.runs.create_calls[0]["thread_id"] == _FOUND_THREAD
    assert client.runs.create_calls[0]["assistant_id"] == "vector_contexter"


async def test_ошибка_постановки_чистит_кеш_и_ищет_снова(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runs.create падает: False, кеш очищен, следующий вызов снова ищет тред."""
    client = _FakeClient()
    client.runs.create_error = RuntimeError("очередь недоступна")
    _install_client(monkeypatch, client)

    result = await _enqueue_contexter(_CALL_ID, **_task_kwargs())

    assert result is False
    assert _CALL_ID not in _CONTEXTER_THREADS
    assert len(client.threads.search_calls) == 1

    await _enqueue_contexter(_CALL_ID, **_task_kwargs())

    assert len(client.threads.search_calls) == 2
