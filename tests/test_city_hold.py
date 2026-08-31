"""Шаг города не закрывается, пока назван район, а не город сети.

Разбор провалившегося звонка. Клиент на вопрос о городе ответил
«Приморский район рядом с метро Пионерская». Судья счёл вопрос отвеченным
и закрыл шаг, а резолвер в том же проходе сказал, что это район. Города у
разговора не стало, и вместе с ним — ни цены, ни сроков, ни филиалов: без
слага код в справочник не ходит вовсе. Остаток звонка бот пять раз
ответил «сейчас уточню», уточнять было неоткуда, человек бросил трубку.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from graph.checker import CheckerVerdict, check_pass
from graph.context import DISTRICT_HINT, ConversationContext, city_unresolved
from script.store import ScriptProgress


class ClosesEverything:
    """Судья, закрывающий любой шаг: так он и повёл себя на звонке."""

    async def judge(self, **_kwargs: Any) -> CheckerVerdict:
        return CheckerVerdict(reply_usable=True, step_closed=True, asking_pointless=False)


def _state(context: dict[str, Any], progress: ScriptProgress) -> dict[str, Any]:
    """Состояние звонка с заданным контекстом."""
    return {
        "script_id": "vector_ru",
        "script_version": "4",
        "profile": {"caller_name": "Андрей"},
        "turn": 2,
        "messages": [HumanMessage(content="Приморский район рядом с метро Пионерская")],
        "script_progress": progress.to_dict(),
        "conversation_context": context,
    }


def _progress() -> ScriptProgress:
    """Прогресс, где город взят в работу и ещё открыт."""
    return ScriptProgress.from_mapping(
        {
            "status": {"city": "pending"},
            "attempts": {"city": 1},
            "in_work": ["city"],
            "taken_turn": {"city": 1},
            "profile": {},
        }
    )


def test_признак_нераспознанного_города() -> None:
    """Города нет, пока нет слага — по какой причине, неважно.

    Сначала признак требовал ещё и подсказки про район. Этого оказалось
    мало: на прогоне распознавание переврало «Питер» в «Итер», клиент
    назвал район, резолвер не понял вовсе и подсказку не поставил. Шаг
    закрылся, справочник не тронули, и бот весь звонок отвечал
    «уточняется».
    """
    assert city_unresolved(ConversationContext(dynamic_text=DISTRICT_HINT))
    assert city_unresolved(ConversationContext()), "пустой контекст — города нет"
    assert not city_unresolved(
        ConversationContext(city_slug="sankt-peterburg", dynamic_text=DISTRICT_HINT)
    ), "город определён — подсказка из прошлого хода роли не играет"


async def test_район_вместо_города_шаг_не_закрывает() -> None:
    """Судья закрыл, но резолвер сказал «район» — шаг остаётся открытым."""
    progress = _progress()
    updated, closures, _ = await check_pass(
        _state({"dynamic_text": DISTRICT_HINT}, progress),
        reply="Приморский район рядом с метро Пионерская",
        judge=ClosesEverything(),
        progress=progress,
    )

    assert updated.status["city"] != "closed"
    assert not any(step == "city" for step, _ in closures)


async def test_названный_город_шаг_закрывает() -> None:
    """Обычный ответ закрывается как прежде, заполнитель может отставать.

    Это защита от прошлой ошибки: сперва я держал шаг открытым по пустому
    полю анкеты, и бот переспрашивал бы город в каждом звонке — поле
    заполняет фоновый агент, он отстаёт на ход.
    """
    progress = _progress()
    updated, closures, _ = await check_pass(
        _state({"city_slug": "sankt-peterburg", "city_name": "Санкт-Петербург"}, progress),
        reply="В Санкт-Петербурге",
        judge=ClosesEverything(),
        progress=progress,
    )

    assert updated.status["city"] == "closed"
    assert ("city", "диалог") in closures


async def test_другие_шаги_держать_не_надо() -> None:
    """Правило узкое: только город. Остальные закрываются по диалогу."""
    progress = ScriptProgress.from_mapping(
        {
            "status": {"transmission": "pending"},
            "attempts": {"transmission": 1},
            "in_work": ["transmission"],
            "taken_turn": {"transmission": 1},
            "profile": {},
        }
    )
    updated, _closures, _ = await check_pass(
        _state({"dynamic_text": DISTRICT_HINT}, progress),
        reply="Механика",
        judge=ClosesEverything(),
        progress=progress,
    )

    assert updated.status["transmission"] == "closed"
