"""Офлайн-тесты подбора ближайших филиалов в воркере и слияния прохода."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from graph.checker_graph import live_check_node
from graph.context import DYN_SEARCHING, ConversationContext
from graph.context_store import MemoryContextStore
from graph.contexter_worker import contexter_task_node
from graph.farewell_agent import FarewellDecision
from graph.profile_agent import ProfileGuess
from script.store import progress_to_state
from tests.test_live_checker import FakeChecker, _name_progress, _state
from tests.test_nearby import FakeNearbyKB

_KEPT_ADDRESSES = "Ближайшие филиалы к месту «метро Проспект Просвещения» — три адреса"
_ITEMS = [
    {
        "slug": "perm_a",
        "address": "ул. Чернышевского, 4",
        "landmark": "",
        "distance_km": 0.4,
    },
    {
        "slug": "perm_b",
        "address": "пр. Ленина, 10",
        "landmark": "",
        "distance_km": 1.2,
    },
]


class _StubScript:
    """Скрипт-заглушка: возражений нет, инструменты подменяются снаружи."""

    objections: dict[str, Any] = {}


class _SnapshotStore(MemoryContextStore):
    """Память, которая копит слепки каждой записи — для промежуточного подбора."""

    def __init__(self) -> None:
        super().__init__()
        self.snapshots: list[ConversationContext] = []

    async def save(self, call_id: str, context: ConversationContext) -> bool:
        self.snapshots.append(ConversationContext.model_validate(context.model_dump()))
        return await super().save(call_id, context)


class _InjectNearbyStore(MemoryContextStore):
    """После статуса «в поиске» подкладывает удачный подбор, как воркер посреди прохода."""

    def __init__(self) -> None:
        super().__init__()
        self._injected = False

    async def save(self, call_id: str, context: ConversationContext) -> bool:
        ok = await super().save(call_id, context)
        stored = self._data.get(call_id)
        if stored is not None and stored.dynamic_status == DYN_SEARCHING and not self._injected:
            stored.nearby_text = "ул. Чернышевского, 4"
            stored.nearby_found = True
            stored.nearby_key = "perm:солнечный"
            stored.branch_candidates = ["perm_a"]
            self._injected = True
        return ok


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


async def _unchanged(context: ConversationContext, **_kwargs: object) -> ConversationContext:
    """Заглушка контекстера: контекст не меняет."""
    return context


def _task(
    *,
    call_id: str = "call-1",
    reply: str = "Мне удобнее в Солнечном районе",
    profile: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Собирает задачу воркеру с местом в профиле."""
    return {
        "call_id": call_id,
        "reply": reply,
        "needs": [],
        "step_needs": [],
        "profile": profile if profile is not None else {"location_hint": "Солнечный"},
        "script_id": "vector_ru",
        "script_version": "2",
    }


def _install_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kb: FakeNearbyKB,
) -> None:
    """Офлайн-подмены воркера: скрипт, контекстер, справочник."""
    _stub_script(monkeypatch)
    monkeypatch.setattr("graph.contexter_worker.run_contexter", _unchanged)
    monkeypatch.setattr("graph.contexter_worker.vector_kb", kb)

    async def _no_profile(*_args: object, **_kwargs: object) -> ProfileGuess:
        return ProfileGuess()

    monkeypatch.setattr("graph.contexter_worker.guess_profile", _no_profile)


async def test_профиль_с_местом_запускает_подбор(
    store: MemoryContextStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Профиль с location_hint и город в кеше — nearby_* и кандидаты после узла."""
    fake_kb = FakeNearbyKB(point=(58.0, 56.0), items=_ITEMS)
    _install_worker(monkeypatch, kb=fake_kb)
    await store.save(
        "call-1",
        ConversationContext(city_slug="perm", city_name="Пермь", static_text="Город: Пермь"),
    )

    await contexter_task_node(_task())

    loaded = await store.load("call-1")
    assert loaded is not None
    assert "ул. Чернышевского, 4" in loaded.nearby_text
    assert loaded.nearby_key == "perm:солнечный"
    assert loaded.nearby_found is True
    assert loaded.branch_candidates == ["perm_a", "perm_b"]
    assert len(fake_kb.geocode_calls) == 1


async def test_повтор_ключа_не_ходит_в_справочник(
    store: MemoryContextStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nearby_key уже совпадает с местом — геокодер не вызван."""
    fake_kb = FakeNearbyKB(
        items=[{"slug": "x", "address": "ул. X", "landmark": "", "distance_km": 1}]
    )
    _install_worker(monkeypatch, kb=fake_kb)
    await store.save(
        "call-1",
        ConversationContext(
            city_slug="perm",
            city_name="Пермь",
            static_text="Город: Пермь",
            nearby_text="уже подобрано",
            nearby_key="perm:солнечный",
            nearby_found=True,
        ),
    )

    await contexter_task_node(_task())

    assert fake_kb.geocode_calls == []
    assert fake_kb.nearest_calls == []
    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.nearby_text == "уже подобрано"
    assert loaded.nearby_key == "perm:солнечный"
    assert loaded.nearby_found is True


async def test_неудача_не_затирает_удачный_подбор_в_кеше(
    store: MemoryContextStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """В кеше удача, поход вернул пусто — адреса и nearby_found целы."""
    fake_kb = FakeNearbyKB(point=None, items=[])
    _install_worker(monkeypatch, kb=fake_kb)
    await store.save(
        "call-1",
        ConversationContext(
            city_slug="perm",
            city_name="Пермь",
            static_text="Город: Пермь",
            nearby_text=_KEPT_ADDRESSES,
            nearby_key="perm:метро проспект просвещения",
            nearby_found=True,
            branch_candidates=["perm_a", "perm_b"],
        ),
    )

    await contexter_task_node(_task(profile={"location_hint": "другой ориентир"}))

    loaded = await store.load("call-1")
    assert loaded is not None
    assert loaded.nearby_text == _KEPT_ADDRESSES
    assert loaded.nearby_found is True
    assert loaded.branch_candidates == ["perm_a", "perm_b"]


async def test_промежуточная_запись_пишет_что_подбор_идёт(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """До конца узла в кеше успевает появиться блок «подбор идёт»."""
    mem = _SnapshotStore()
    monkeypatch.setattr("graph.contexter_worker.context_store", mem)
    monkeypatch.setattr("graph.contexter.context_store", mem)
    fake_kb = FakeNearbyKB(point=(58.0, 56.0), items=_ITEMS)
    _install_worker(monkeypatch, kb=fake_kb)
    await mem.save(
        "call-1",
        ConversationContext(city_slug="perm", city_name="Пермь"),
    )

    await contexter_task_node(_task())

    assert any("подбираются" in (snap.nearby_text or "") for snap in mem.snapshots)


async def test_финальное_слияние_прохода_не_затирает_подбор_воркера(
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пока проход шёл, в кеш легли адреса — после прохода они на месте."""
    mem = _InjectNearbyStore()
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

    async def _queued(*_args: object, **_kwargs: object) -> bool:
        return True

    async def _no_profile(*_args: object, **_kwargs: object) -> ProfileGuess:
        return ProfileGuess()

    async def _no_farewell(*_args: object, **_kwargs: object) -> FarewellDecision:
        return FarewellDecision(conversation_ended=False)

    monkeypatch.setattr("graph.checker_graph._enqueue_contexter", _queued)
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

    loaded = await mem.load("local")
    assert loaded is not None
    assert "ул. Чернышевского, 4" in loaded.nearby_text
    assert loaded.nearby_found is True
    assert loaded.branch_candidates == ["perm_a"]


def test_динамика_прошлой_реплики_не_подклеивается():
    """Указания не копятся за звонок: динамика описывает сказанное сейчас.

    На разборе провалившегося прогона в динамике к последнему ходу лежала
    подсказка «уточни город обучения» со второго хода — вместе с блоком
    возражения из последних. Пока город не определён, подсказка верна;
    после — заставит переспрашивать город без причины.
    """
    from graph.context import ConversationContext
    from graph.contexter_worker import _keep_concurrent_dynamic

    base = ConversationContext(dynamic_text="старое указание", last_reply_hash="реплика-1")
    overlay = ConversationContext(dynamic_text="свежий блок")

    _keep_concurrent_dynamic(base, overlay, current_hash="реплика-2")

    assert overlay.dynamic_text == "свежий блок"


def test_динамика_той_же_реплики_сохраняется():
    """Параллельный проход по той же реплике терять нельзя."""
    from graph.context import ConversationContext
    from graph.contexter_worker import _keep_concurrent_dynamic

    base = ConversationContext(dynamic_text="блок соседа", last_reply_hash="реплика-1")
    overlay = ConversationContext(dynamic_text="свой блок")

    _keep_concurrent_dynamic(base, overlay, current_hash="реплика-1")

    assert "блок соседа" in overlay.dynamic_text
    assert "свой блок" in overlay.dynamic_text


def test_динамика_без_пометки_реплики_сохраняется():
    """Проход ещё идёт и пометку не поставил — текст сохраняем как прежде."""
    from graph.context import ConversationContext
    from graph.contexter_worker import _keep_concurrent_dynamic

    base = ConversationContext(dynamic_text="блок на лету")
    overlay = ConversationContext(dynamic_text="свой блок")

    _keep_concurrent_dynamic(base, overlay, current_hash="реплика-2")

    assert "блок на лету" in overlay.dynamic_text
