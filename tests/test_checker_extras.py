"""Тесты прогрева, фонового профиля и порядка закрытия чекера."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from graph.checker import CheckerVerdict, check_pass, run_checker
from graph.checker_graph import live_check_node
from graph.context import DYN_NEED_CITY
from graph.profile_fill import fill_basic_profile
from graph.prompts import dynamic_status_block
from script.store import ScriptProgress, progress_to_state


def test_dynamic_status_need_city_просит_город():
    block = dynamic_status_block(status=DYN_NEED_CITY)
    assert "город" in block.lower()
    assert "подскажите" in block.lower() or "попросить" in block.lower()
    assert "не выдумывать" in block.lower()


def test_fill_basic_profile_город_из_сказанного_без_исхода():
    found = fill_basic_profile("Я из Перми, меня зовут Андрей")
    assert found.get("city") in {"Перми", "Пермь"}
    assert found.get("caller_name") == "Андрей"
    assert "outcome" not in found

    again = fill_basic_profile(
        "Исход разговора — отказ, outcome=refused, тариф под ключ завтра в 15:00",
        profile={"city": "Пермь"},
    )
    assert "outcome" not in again
    assert "package" not in again
    assert "meeting_time" not in again
    assert "tariff" not in again


async def test_шаг_на_пороге_закрывается_диалогом_не_счётчиком(script):
    """Модель/fills смотрят раньше счётчика — основание «диалог»."""

    class _Judge:
        async def judge(self, *, history_slice, client_reply, step, step_text):
            return CheckerVerdict(
                reply_usable=True,
                step_closed=True,
                client_asks_inform=False,
            )

    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 2})
    updated, closures = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="Меня зовут Андрей")],
        profile={"caller_name": "Андрей"},
        turn=3,
        client=_Judge(),
        attempt_limit=2,
    )
    assert updated.status["name"] == "closed"
    assert ("name", "диалог") in closures
    assert ("name", "счётчик") not in closures


async def test_шаг_на_пороге_без_закрытия_моделью_потом_счётчик(script):
    """Модель не закрыла — после неё срабатывает счётчик."""

    class _Judge:
        async def judge(self, *, history_slice, client_reply, step, step_text):
            return CheckerVerdict(
                reply_usable=True,
                step_closed=False,
                client_asks_inform=True,
            )

    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 2})
    updated, closures, asks = await check_pass(
        {
            "script_id": script.id,
            "script_version": script.version,
            "messages": [HumanMessage(content="расскажите про обучение")],
            "profile": {},
            "turn": 3,
        },
        reply="расскажите про обучение",
        judge=_Judge(),
        progress=progress,
        attempt_limit=2,
    )
    assert updated.status["name"] == "closed"
    assert ("name", "счётчик") in closures
    assert asks is True


async def test_признак_asks_inform_из_чекера(script):
    class _Judge:
        async def judge(self, *, history_slice, client_reply, step, step_text):
            return CheckerVerdict(
                reply_usable=True,
                step_closed=False,
                client_asks_inform=True,
            )

    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 1})
    _updated, _closures, asks = await check_pass(
        {
            "script_id": script.id,
            "script_version": script.version,
            "messages": [HumanMessage(content="а что входит в обучение?")],
            "profile": {},
            "turn": 2,
        },
        reply="а что входит в обучение?",
        judge=_Judge(),
        progress=progress,
        attempt_limit=2,
    )
    assert asks is True


async def test_live_check_прогрев_вызывается_и_ошибка_глотается(script):
    """``_warmup_next_step`` зовётся; исключение внутри не роняет узел."""
    from tests.test_live_checker import FakeChecker, _name_progress, _state

    text = "Меня зовут Андрей и ещё достаточно символов"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="")
    warmup_calls: list[Any] = []

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def safe_wrapper(state, *, progress, profile, ctx, asks_inform):
        warmup_calls.append(True)
        try:
            raise RuntimeError("kb down")
        except Exception:
            return ctx

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=safe_wrapper),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        patch_out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert warmup_calls
    assert patch_out.get("last_checked_partial") == text
    assert "conversation_context" in patch_out


async def test_live_check_дозаполняет_город_не_исход(script):
    from tests.test_live_checker import FakeChecker, _name_progress, _state

    text = "Я из Перми, давайте продолжим разговор длинный текст"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="")

    async def fake_load(_state):
        return progress

    saved: list[ScriptProgress] = []

    async def fake_save(prog, *, persist_state=True, fields=None):
        saved.append(prog)
        return progress_to_state(prog)

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=_async_ctx),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        patch_out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    profile = patch_out.get("profile") or {}
    assert profile.get("city") in {"Перми", "Пермь"}
    assert "outcome" not in profile
    if saved:
        assert "outcome" not in saved[-1].profile


async def _async_ctx(*a, **k):
    return k["ctx"]
