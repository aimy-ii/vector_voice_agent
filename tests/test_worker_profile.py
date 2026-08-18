"""Офлайн-тесты разбора анкеты в воркере контекстера."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from graph.checker_graph import live_check_node
from graph.context import ConversationContext
from graph.context_store import MemoryContextStore
from graph.contexter_worker import contexter_task_node
from graph.farewell_agent import FarewellDecision
from graph.profile_agent import ProfileGuess, ProfileValue
from script.store import MemoryScriptStore, ScriptProgress, progress_to_state
from tests.test_live_checker import FakeChecker, _name_progress, _state
from tests.test_nearby import FakeNearbyKB
from tests.test_worker_nearby import _ITEMS


class _StubScript:
    """Скрипт-заглушка: возражений нет, перечень полей берётся из формы."""

    objections: dict[str, Any] = {}
    profile_fields: dict[str, Any] = {}


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> MemoryContextStore:
    """Кеш воркера в памяти процесса."""
    mem = MemoryContextStore()
    monkeypatch.setattr("graph.contexter_worker.context_store", mem)
    monkeypatch.setattr("graph.contexter.context_store", mem)
    return mem


@pytest.fixture
def progress_store(monkeypatch: pytest.MonkeyPatch) -> MemoryScriptStore:
    """Прогресс звонка в памяти процесса."""
    mem = MemoryScriptStore()
    monkeypatch.setattr("graph.contexter_worker.script_store", mem)
    return mem


def _stub_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет загрузку скрипта заглушкой с пустыми возражениями."""
    monkeypatch.setattr(
        "graph.contexter_worker.registry.get",
        lambda *_args, **_kwargs: _StubScript(),
    )


async def _unchanged(context: ConversationContext, **_kwargs: object) -> ConversationContext:
    """Заглушка контекстера: контекст не меняет."""
    return context


def _task(
    *,
    call_id: str = "call-1",
    reply: str = "Меня зовут Андрей",
    profile: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Собирает задачу воркеру без заранее заполненного профиля."""
    return {
        "call_id": call_id,
        "reply": reply,
        "needs": [],
        "step_needs": [],
        "profile": profile if profile is not None else {},
        "script_id": "vector_ru",
        "script_version": "2",
    }


def _install_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kb: FakeNearbyKB | None = None,
) -> None:
    """Офлайн-подмены воркера: скрипт, контекстер, справочник, пустая анкета."""
    _stub_script(monkeypatch)
    monkeypatch.setattr("graph.contexter_worker.run_contexter", _unchanged)
    monkeypatch.setattr(
        "graph.contexter_worker.vector_kb",
        kb if kb is not None else FakeNearbyKB(items=[]),
    )

    async def _no_profile(*_args: object, **_kwargs: object) -> ProfileGuess:
        return ProfileGuess()

    monkeypatch.setattr("graph.contexter_worker.guess_profile", _no_profile)


async def test_анкета_пишет_имя_в_прогресс(
    store: MemoryContextStore,
    progress_store: MemoryScriptStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Заглушка guess_profile с именем — после узла в прогрессе есть caller_name."""
    _install_worker(monkeypatch)
    await store.save("call-1", ConversationContext())
    await progress_store.save("call-1", ScriptProgress())

    async def fake_guess(*_args: object, **_kwargs: object) -> ProfileGuess:
        return ProfileGuess(values=[ProfileValue(key="caller_name", value="Андрей")])

    monkeypatch.setattr("graph.contexter_worker.guess_profile", fake_guess)

    await contexter_task_node(_task())

    stored = await progress_store.load("call-1")
    assert stored is not None
    assert stored.profile.get("caller_name") == "Андрей"


async def test_заполненное_не_перезаписывается(
    store: MemoryContextStore,
    progress_store: MemoryScriptStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Город уже в прогрессе, заглушка без права перезаписи — город прежний."""
    _install_worker(monkeypatch)
    await store.save("call-1", ConversationContext())
    await progress_store.save("call-1", ScriptProgress(profile={"city": "Пермь"}))

    async def fake_guess(*_args: object, **_kwargs: object) -> ProfileGuess:
        return ProfileGuess(values=[ProfileValue(key="city", value="Казань")])

    monkeypatch.setattr("graph.contexter_worker.guess_profile", fake_guess)

    await contexter_task_node(_task(reply="я из Казани на самом деле"))

    stored = await progress_store.load("call-1")
    assert stored is not None
    assert stored.profile.get("city") == "Пермь"


async def test_свежая_локация_даёт_подбор_в_той_же_задаче(
    store: MemoryContextStore,
    progress_store: MemoryScriptStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """location_hint из анкеты виден подбору в том же прогоне узла."""
    fake_kb = FakeNearbyKB(point=(58.0, 56.0), items=_ITEMS)
    _install_worker(monkeypatch, kb=fake_kb)
    await store.save(
        "call-1",
        ConversationContext(city_slug="perm", city_name="Пермь", static_text="Город: Пермь"),
    )
    await progress_store.save("call-1", ScriptProgress())

    async def fake_guess(*_args: object, **_kwargs: object) -> ProfileGuess:
        return ProfileGuess(values=[ProfileValue(key="location_hint", value="Солнечный")])

    monkeypatch.setattr("graph.contexter_worker.guess_profile", fake_guess)

    await contexter_task_node(_task(reply="Мне удобнее в Солнечном районе"))

    loaded = await store.load("call-1")
    assert loaded is not None
    assert "ул. Чернышевского, 4" in loaded.nearby_text


async def test_сбой_агента_анкеты_не_роняет_задачу(
    store: MemoryContextStore,
    progress_store: MemoryScriptStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Исключение guess_profile: узел завершился, контекст записан, прогресс цел."""
    _install_worker(monkeypatch)
    seeded_progress = ScriptProgress(profile={"caller_name": "Мария"})
    await store.save("call-1", ConversationContext(static_text="статика"))
    await progress_store.save("call-1", seeded_progress)

    async def boom_guess(*_args: object, **_kwargs: object) -> ProfileGuess:
        raise RuntimeError("profile agent down")

    monkeypatch.setattr("graph.contexter_worker.guess_profile", boom_guess)

    out = await contexter_task_node(_task())

    assert out == {}
    loaded = await store.load("call-1")
    assert loaded is not None
    stored = await progress_store.load("call-1")
    assert stored is not None
    assert stored.profile == {"caller_name": "Мария"}


async def test_проход_пишет_только_статус(
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Служебный проход вызывает _save_progress только с полем status."""
    mem = MemoryContextStore()
    monkeypatch.setattr("graph.nodes.context_store", mem)
    await mem.save("local", ConversationContext())

    progress = _name_progress()
    text = "пока ещё думаю над ответом длинный"
    state = _state(script, partial=text, progress=progress, last_checked="")
    saved_fields: list[object] = []

    async def fake_load(_state: object) -> object:
        return progress

    async def fake_save(prog: object, *, persist_state: bool = True, fields: object = None) -> dict:
        _ = persist_state
        saved_fields.append(fields)
        return progress_to_state(prog)  # type: ignore[arg-type]

    async def fake_warmup(*_args: object, **kwargs: Any) -> Any:
        return kwargs["ctx"]

    async def _queued(*_args: object, **_kwargs: object) -> bool:
        return True

    async def _no_farewell(*_args: object, **_kwargs: object) -> FarewellDecision:
        return FarewellDecision(conversation_ended=False)

    monkeypatch.setattr("graph.checker_graph._enqueue_contexter", _queued)
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

    assert saved_fields
    for fields in saved_fields:
        assert fields == frozenset({"status"})
