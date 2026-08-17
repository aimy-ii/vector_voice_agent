"""Сдвиг ведущего в ``plan_node``, если шаг прошлого хода снова впереди."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from graph import nodes as nodes_module
from graph.state import new_state_defaults
from script.store import MemoryScriptStore, ScriptProgress


@pytest.fixture()
def store(monkeypatch) -> MemoryScriptStore:
    mem = MemoryScriptStore()
    monkeypatch.setattr(nodes_module, "script_store", mem)
    return mem


@pytest.fixture()
def use_v2(monkeypatch) -> None:
    monkeypatch.setattr(nodes_module.settings, "script_version", "2")


@pytest.fixture()
def plan_logs(monkeypatch) -> list[str]:
    """Собирает тексты ``[plan|done]`` без побочных эффектов stage."""
    texts: list[str] = []

    def _stage(name: str, text: str, phase: str = "done", **_kwargs: Any) -> None:
        if name == "plan" and phase == "done":
            texts.append(text)

    monkeypatch.setattr(nodes_module, "stage", _stage)
    return texts


def _base_state(**extra: Any) -> dict[str, Any]:
    return {
        **new_state_defaults(),
        "messages": [HumanMessage(content="ну не знаю")],
        "script_id": "vector_ru",
        "script_version": "2",
        "turn": 3,
        "turn_kind": "client",
        **extra,
    }


async def _seed_pending_head(store: MemoryScriptStore) -> None:
    """Шапка из двух висящих: name → city (и свежий who_studies при soft_cap≥3)."""
    await store.save(
        "local",
        ScriptProgress(
            status={"name": "pending", "city": "pending"},
            attempts={"name": 1, "city": 1},
            taken_turn={"name": 1, "city": 1},
            in_work=["name", "city"],
        ),
    )


async def test_сдвиг_ведущего_если_совпал_с_прошлым_ходом(store, use_v2, plan_logs, monkeypatch):
    """Совпал с прошлым ходом + реплика клиента — ведущий = следующий в шапке."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    state = _base_state(current_step="name")
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["head_steps"][:2] == ["name", "city"]
    assert out["current_step"] == "city"


async def test_без_сдвига_если_ведущий_другой(store, use_v2, plan_logs):
    """Пересчитанный ведущий не совпал с прошлым — ничего не меняем."""
    await store.save(
        "local",
        ScriptProgress(
            status={"name": "closed", "city": "pending"},
            attempts={"name": 1, "city": 1},
            taken_turn={"name": 1, "city": 1},
            in_work=["name", "city"],
            profile={"caller_name": "Мария"},
        ),
    )
    state = _base_state(
        current_step="name",
        profile={"caller_name": "Мария"},
    )
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["current_step"] == "city"
    assert "сдвиг" not in (plan_logs[-1] if plan_logs else "")


async def test_один_открытый_в_шапке_ведущий_не_меняется(store, use_v2, plan_logs, monkeypatch):
    """В шапке только один открытый — сдвигать некуда."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 1)
    await store.save(
        "local",
        ScriptProgress(
            status={"name": "pending"},
            attempts={"name": 1},
            taken_turn={"name": 1},
            in_work=["name"],
        ),
    )
    state = _base_state(current_step="name")
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["head_steps"] == ["name"]
    assert out["current_step"] == "name"
    assert "сдвиг" not in plan_logs[-1]


async def test_сдвиг_на_ходе_без_реплики_клиента(store, use_v2, plan_logs, monkeypatch):
    """Ход continuation — ведущий тоже сдвигается по шапке."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    state = _base_state(current_step="name", turn_kind="continuation")
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["current_step"] == "city"
    assert out["head_steps"][:2] == ["name", "city"]
    assert "сдвиг" in plan_logs[-1]


async def test_молчание_не_сдвигает_ведущего(store, use_v2, plan_logs, monkeypatch):
    """Ход silence — ведущий остаётся прежним, хотя в шапке есть куда сдвинуть."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    state = _base_state(current_step="name", turn_kind="silence")
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["current_step"] == "name"
    assert out["head_steps"][:2] == ["name", "city"]
    assert "сдвиг" not in plan_logs[-1]


async def test_молчание_копит_счётчик_повторов(store, use_v2, monkeypatch):
    """Без сдвига на молчании ``lead_repeat`` растёт — режим промпта сменится сам."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    out = await nodes_module.plan_node(
        _base_state(current_step="name", lead_repeat=1, turn_kind="silence"),
        None,  # type: ignore[arg-type]
    )
    assert out["current_step"] == "name"
    assert out["lead_repeat"] == 2


async def test_молчание_с_другим_ведущим_сбрасывает_счётчик(store, use_v2, monkeypatch):
    """Пересчитанный ведущий не совпал с прошлым — счётчик единица и на молчании."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    out = await nodes_module.plan_node(
        _base_state(current_step="city", lead_repeat=3, turn_kind="silence"),
        None,  # type: ignore[arg-type]
    )
    assert out["current_step"] == "name"
    assert out["lead_repeat"] == 1


@pytest.mark.parametrize("turn_kind", ["continuation", "silence", "pull"])
async def test_ход_без_реплики_не_трогает_счётчики(
    store, use_v2, plan_logs, monkeypatch, turn_kind: str
):
    """Попытки, взятие в работу и статусы на ходах без реплики остаются прежними."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    before = await store.load("local")
    assert before is not None
    state = _base_state(current_step="name", turn_kind=turn_kind)
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    after = await store.load("local")
    assert after is not None
    assert after.attempts == before.attempts
    assert after.in_work == before.in_work
    assert after.taken_turn == before.taken_turn
    assert after.status == before.status
    assert out["head_new_step"] is None


async def test_лог_сдвига_содержит_прежний_шаг(store, use_v2, plan_logs, monkeypatch):
    """В ``[plan|done]`` есть пометка о сдвиге и шаг, который вёл раньше."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    state = _base_state(current_step="name")
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["current_step"] == "city"
    assert plan_logs
    assert "сдвиг с name" in plan_logs[-1]
    assert "шаг city" in plan_logs[-1]


async def test_повтор_ведущего_шага_даёт_lead_repeat_два(store, use_v2, monkeypatch):
    """Один и тот же ведущий на двух ходах подряд — ``lead_repeat`` равен двум."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 1)
    await store.save(
        "local",
        ScriptProgress(
            status={"name": "pending"},
            attempts={"name": 1},
            taken_turn={"name": 1},
            in_work=["name"],
        ),
    )
    first = await nodes_module.plan_node(_base_state(), None)  # type: ignore[arg-type]
    assert first["current_step"] == "name"
    assert first["lead_repeat"] == 1

    second = await nodes_module.plan_node(
        _base_state(current_step="name", lead_repeat=1),
        None,  # type: ignore[arg-type]
    )
    assert second["current_step"] == "name"
    assert second["lead_repeat"] == 2


async def test_смена_ведущего_шага_сбрасывает_lead_repeat(store, use_v2):
    """Смена ведущего шага сбрасывает ``lead_repeat`` в единицу."""
    await store.save(
        "local",
        ScriptProgress(
            status={"name": "closed", "city": "pending"},
            attempts={"name": 1, "city": 1},
            taken_turn={"name": 1, "city": 1},
            in_work=["name", "city"],
            profile={"caller_name": "Мария"},
        ),
    )
    out = await nodes_module.plan_node(
        _base_state(
            current_step="name",
            lead_repeat=5,
            profile={"caller_name": "Мария"},
        ),
        None,  # type: ignore[arg-type]
    )
    assert out["current_step"] == "city"
    assert out["lead_repeat"] == 1


def test_нулевой_порог_не_включает_характер_повтора(script, monkeypatch):
    """При ``lead_repeat_threshold = 0`` сборка штатная при любом счётчике."""
    from graph.prompts import LEAD_REPEAT_INTRO, RULE_MOVE_ON

    monkeypatch.setattr(nodes_module.settings, "lead_repeat_threshold", 0)
    step = script.step("city")
    messages = nodes_module._build_respond_messages(
        prompt_kind="full",
        script=script,
        state={**new_state_defaults(), "lead_repeat": 99},
        history=[],
        profile={},
        facts={},
        lead=step,
        head=[step],
        context_text="",
        dynamic_status="",
        pending_fields=[],
        turn_kind="client",
    )
    content = messages[0].content
    assert LEAD_REPEAT_INTRO not in content
    assert RULE_MOVE_ON in content


def test_no_client_reply_pull_истинно():
    """Вид хода ``pull`` считается ходом без реплики клиента."""
    assert nodes_module._no_client_reply("pull") is True


def test_повтор_имеет_приоритет_над_pull(script, monkeypatch):
    """При счётчике выше порога на ``pull`` собирается режим повтора."""
    from graph.prompts import LEAD_REPEAT_INTRO, PULL_TASK

    monkeypatch.setattr(nodes_module.settings, "lead_repeat_threshold", 2)
    step = script.step("city")
    captured: dict[str, Any] = {}
    original = nodes_module.build_turn_messages

    def _capture(**kwargs: Any) -> Any:
        captured["mode"] = kwargs.get("mode")
        return original(**kwargs)

    monkeypatch.setattr(nodes_module, "build_turn_messages", _capture)
    messages = nodes_module._build_respond_messages(
        prompt_kind="full",
        script=script,
        state={**new_state_defaults(), "lead_repeat": 99},
        history=[],
        profile={},
        facts={},
        lead=step,
        head=[step],
        context_text="",
        dynamic_status="",
        pending_fields=[],
        turn_kind="pull",
    )
    assert captured["mode"] == "repeat"
    assert LEAD_REPEAT_INTRO in messages[0].content
    assert PULL_TASK not in messages[0].content


async def test_тема_двигается_на_ходе_догона(store, use_v2, plan_logs, monkeypatch):
    """На ``pull`` ведущий сдвигается, порядок шапки не меняется."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    state = _base_state(current_step="name", turn_kind="pull")
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["current_step"] == "city"
    assert out["head_steps"][:2] == ["name", "city"]
    assert plan_logs
    assert "сдвиг" in plan_logs[-1]


async def test_двигать_некуда_включается_повтор(store, use_v2, script, monkeypatch):
    """Один шаг в шапке и два ``pull`` подряд — повтор на втором ходе."""
    from graph.prompts import LEAD_REPEAT_INTRO

    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 1)
    monkeypatch.setattr(nodes_module.settings, "lead_repeat_threshold", 2)
    await store.save(
        "local",
        ScriptProgress(
            status={"name": "pending"},
            attempts={"name": 1},
            taken_turn={"name": 1},
            in_work=["name"],
        ),
    )
    first = await nodes_module.plan_node(
        _base_state(turn_kind="pull"),
        None,  # type: ignore[arg-type]
    )
    assert first["current_step"] == "name"
    assert first["lead_repeat"] == 1

    second = await nodes_module.plan_node(
        _base_state(current_step="name", lead_repeat=1, turn_kind="pull"),
        None,  # type: ignore[arg-type]
    )
    assert second["current_step"] == "name"
    assert second["lead_repeat"] == 2

    step = script.step("name")
    messages = nodes_module._build_respond_messages(
        prompt_kind="full",
        script=script,
        state={**new_state_defaults(), "lead_repeat": second["lead_repeat"]},
        history=[],
        profile={},
        facts={},
        lead=step,
        head=[step],
        context_text="",
        dynamic_status="",
        pending_fields=[],
        turn_kind="pull",
    )
    assert LEAD_REPEAT_INTRO in messages[0].content


def test_догон_с_единичным_счётчиком_идёт_в_короткую_сборку(script, monkeypatch):
    """На ``pull`` при ``lead_repeat=1`` собирается короткое вытаскивание.

    Полная сборка с ``PULL_TASK`` осталась в файле, но со штатной точки
    выбора больше не приходит: добивку собирает ``build_pull_messages``.
    """
    from graph.prompts import _PULL_THINKING, LEAD_REPEAT_INTRO, PULL_TASK

    monkeypatch.setattr(nodes_module.settings, "lead_repeat_threshold", 2)
    step = script.step("city")
    messages = nodes_module._build_respond_messages(
        prompt_kind="full",
        script=script,
        state={**new_state_defaults(), "lead_repeat": 1},
        history=[],
        profile={},
        facts={},
        lead=step,
        head=[step],
        context_text="",
        dynamic_status="",
        pending_fields=[],
        turn_kind="pull",
    )
    content = messages[0].content
    assert _PULL_THINKING in content
    assert PULL_TASK not in content
    assert LEAD_REPEAT_INTRO not in content


def test_silence_выше_порога_даёт_repeat(script, monkeypatch):
    """При ``turn_kind="silence"`` и счётчике выше порога уходит ``mode="repeat"``."""
    from graph.prompts import LEAD_REPEAT_INTRO

    monkeypatch.setattr(nodes_module.settings, "lead_repeat_threshold", 2)
    step = script.step("city")
    captured: dict[str, Any] = {}
    original = nodes_module.build_turn_messages

    def _capture(**kwargs: Any) -> Any:
        captured["mode"] = kwargs.get("mode")
        return original(**kwargs)

    monkeypatch.setattr(nodes_module, "build_turn_messages", _capture)
    messages = nodes_module._build_respond_messages(
        prompt_kind="full",
        script=script,
        state={**new_state_defaults(), "lead_repeat": 3},
        history=[],
        profile={},
        facts={},
        lead=step,
        head=[step],
        context_text="",
        dynamic_status="",
        pending_fields=[],
        turn_kind="silence",
    )
    assert captured["mode"] == "repeat"
    assert LEAD_REPEAT_INTRO in messages[0].content
    assert "СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" in messages[0].content


def _respond_messages(script, *, lead, head):
    """Собирает сообщения генератора для проверки порядка разделов."""
    return nodes_module._build_respond_messages(
        prompt_kind="full",
        script=script,
        state=new_state_defaults(),
        history=[],
        profile={},
        facts={},
        lead=lead,
        head=head,
        context_text="",
        dynamic_status="",
        pending_fields=[],
        turn_kind="client",
    )


def test_ведущий_из_плана_идёт_первым_разделом(script):
    """Ведущий из плана — первый раздел промпта, прежний шаг — во втором."""
    step_a = script.step("name")
    step_b = script.step("city")
    messages = _respond_messages(script, lead=step_b, head=[step_a, step_b])
    content = messages[0].content
    now_at = content.index("# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ")
    hang_at = content.index("# ЕЩЁ НЕ ЗАКРЫТО")
    lead_title_at = content.index(f"## {step_b.goal}", now_at)
    hang_title_at = content.index(f"## {step_a.goal}", hang_at)
    assert now_at < lead_title_at < hang_at < hang_title_at


def test_прежний_ведущий_не_пропадает(script):
    """При сдвиге ведущего прежний шаг остаётся в тексте промпта."""
    step_a = script.step("name")
    step_b = script.step("city")
    messages = _respond_messages(script, lead=step_b, head=[step_a, step_b])
    assert step_a.goal in messages[0].content


def test_lead_of_берёт_шаг_плана(script):
    """``_lead_of`` возвращает шаг плана, если он лежит в шапке."""
    step_a = script.step("name")
    step_b = script.step("city")
    assert nodes_module._lead_of([step_a, step_b], step_b) is step_b


def test_lead_of_откатывается_на_первый_если_шага_нет(script):
    """Нет выбранного или он вне шапки — берётся первый шаг шапки."""
    step_a = script.step("name")
    step_b = script.step("city")
    step_c = script.step("who_studies")
    assert nodes_module._lead_of([step_a, step_b], None) is step_a
    assert nodes_module._lead_of([step_a, step_b], step_c) is step_a


def test_lead_of_на_пустой_шапке(script):
    """Пустая шапка — ``_lead_of`` всегда None."""
    step_a = script.step("name")
    assert nodes_module._lead_of([], None) is None
    assert nodes_module._lead_of([], step_a) is None


def test_lead_first_один_шаг_в_шапке(script):
    """Один шаг в шапке — перечень из него, раздела незакрытых нет."""
    step_a = script.step("name")
    assert nodes_module._lead_first([step_a], step_a) == [step_a]
    content = _respond_messages(script, lead=step_a, head=[step_a])[0].content
    assert "# ЕЩЁ НЕ ЗАКРЫТО" not in content


def test_lead_first_пустая_шапка(script):
    """Пустая шапка — пустой перечень и текст про закрытые шаги."""
    assert nodes_module._lead_first([], None) == []
    content = _respond_messages(script, lead=None, head=[])[0].content
    assert "Все шаги скрипта закрыты" in content


def test_lead_first_шаг_вне_шапки_не_добавляется(script):
    """Ведущий вне шапки не попадает в перечень — порядок шапки как есть."""
    step_a = script.step("name")
    step_b = script.step("city")
    step_c = script.step("who_studies")
    assert nodes_module._lead_first([step_a, step_b], step_c) == [step_a, step_b]


async def test_шапка_в_состоянии_не_переставляется(store, use_v2, plan_logs, monkeypatch):
    """Сдвиг меняет ``current_step``, порядок ``head_steps`` не трогает."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    state = _base_state(current_step="name")
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["head_steps"][:2] == ["name", "city"]
    assert out["current_step"] == "city"
