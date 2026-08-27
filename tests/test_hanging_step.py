"""Зависший шаг закрывается страховкой, когда судья не смог.

Разбор живого звонка: четыре шага висели в шапке одновременно, до восьми
ходов каждый. Все четыре — презентационные: человек их выслушивает, а не
отвечает на них, и судье закрывать нечем. Снаружи это читается как
«повторяет вопросы, не удерживает данные».
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from core.config import settings
from graph.checker import CheckerVerdict, check_pass
from script.source import registry
from script.store import ScriptProgress

LIMIT = settings.step_head_limit


class NeverCloses:
    """Судья, который не закрывает ничего: так ведёт себя живой на показах."""

    async def judge(self, **_kwargs: Any) -> CheckerVerdict:
        return CheckerVerdict(reply_usable=True, step_closed=False, asking_pointless=False)


def state_with(progress: ScriptProgress) -> dict[str, Any]:
    """Состояние звонка с готовым прогрессом.

    Args:
        progress: прогресс шагов.

    Returns:
        Состояние для ``check_pass``.
    """
    return {
        "script_id": "vector_ru",
        "script_version": "4",
        "profile": {},
        "turn": 10,
        "messages": [HumanMessage(content="ага, понятно")],
        "script_progress": progress.to_dict(),
    }


def progress_for(step_id: str, attempts: int) -> ScriptProgress:
    """Прогресс, где один шаг взят в работу и провисел заданное число ходов.

    Args:
        step_id: слаг шага.
        attempts: сколько ходов шаг был в шапке.

    Returns:
        Прогресс звонка.
    """
    return ScriptProgress.from_mapping(
        {
            "status": {step_id: "pending"},
            "attempts": {step_id: attempts},
            "in_work": [step_id],
            "taken_turn": {step_id: 1},
            "profile": {},
        }
    )


async def test_шаг_в_пределах_порога_остаётся_открытым():
    progress = progress_for("practice", LIMIT - 1)
    updated, closures, _ = await check_pass(
        state_with(progress),
        reply="ага, понятно",
        judge=NeverCloses(),
        progress=progress,
    )
    assert updated.status["practice"] != "closed"
    assert closures == []


async def test_зависший_шаг_закрывается_страховкой():
    progress = progress_for("practice", LIMIT)
    updated, closures, _ = await check_pass(
        state_with(progress),
        reply="ага, понятно",
        judge=NeverCloses(),
        progress=progress,
    )
    assert updated.status["practice"] == "closed"
    assert any(step == "practice" and "висит" in reason for step, reason in closures)


class ClosesByDialogue:
    """Судья, который закрывает шаг по диалогу."""

    async def judge(self, **_kwargs: Any) -> CheckerVerdict:
        return CheckerVerdict(reply_usable=True, step_closed=True, asking_pointless=False)


async def test_судья_остаётся_главным():
    """Шаг за порогом, но судья закрыл его сам — основание «диалог».

    Страховка не должна перехватывать закрытие у судьи: иначе в журнале
    пропадёт настоящая причина, и разбор звонка станет гаданием.
    """
    progress = progress_for("practice", LIMIT + 3)
    _updated, closures, _ = await check_pass(
        state_with(progress),
        reply="да, всё понятно про практику",
        judge=ClosesByDialogue(),
        progress=progress,
    )
    assert closures == [("practice", "диалог")]


@pytest.mark.parametrize("step_id", ["practice", "included", "terms", "theory_format"])
async def test_страховка_работает_для_любого_шага(step_id: str):
    progress = progress_for(step_id, LIMIT + 2)
    updated, _closures, _ = await check_pass(
        state_with(progress),
        reply="ага, понятно",
        judge=NeverCloses(),
        progress=progress,
    )
    assert updated.status[step_id] == "closed"


def test_скрипт_v4_поднимается():
    """Страховка считается по реальному скрипту, а не по выдуманному."""
    assert registry.get("vector_ru", "4").steps["practice"] is not None
