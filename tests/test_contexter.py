"""Тесты каркаса контекстера и выдачи справок."""

from __future__ import annotations

from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    ConversationContext,
)
from graph.contexter import NullContexterTools, run_contexter
from script.models import Help, Objection


class _CountingTools:
    """Инструменты, считающие вызовы search."""

    def __init__(self, result: str | None = None) -> None:
        self.calls: list[str] = []
        self.result = result

    async def search(self, query: str) -> str | None:
        self.calls.append(query)
        return self.result


async def test_контекстер_нечего_добавлять_статус_не_требуется():
    ctx = ConversationContext(static_text="Город: Пермь")
    tools = _CountingTools()
    out = await run_contexter(ctx, reply="да, хорошо", tools=tools)
    assert out.dynamic_status == DYN_NONE
    assert out.situation_slug is None
    assert tools.calls == []
    assert out.static_text == "Город: Пермь"


async def test_контекстер_ответ_в_статике_готово_без_инструментов():
    ctx = ConversationContext(
        static_text=("Город: Пермь.\nЦена (готовая фраза): Стоимость обучения — от 43900 рублей.")
    )
    tools = _CountingTools()
    out = await run_contexter(ctx, reply="а сколько стоимость обучения?", tools=tools)
    assert out.dynamic_status == DYN_READY
    assert tools.calls == []
    assert out.static_text == ctx.static_text


async def test_контекстер_нет_в_статике_и_helps_не_нашлось():
    ctx = ConversationContext(static_text="Город: Пермь. Автопарк: Solaris.")
    tools = NullContexterTools()
    out = await run_contexter(
        ctx,
        reply="а как записаться на дайвинг?",
        tools=tools,
        helps={},
    )
    assert out.dynamic_status == DYN_MISSING
    assert out.situation_slug is None
    assert out.static_text == ctx.static_text


async def test_контекстер_статика_не_трогается_при_находке():
    ctx = ConversationContext(static_text="статика города")
    tools = _CountingTools(result="нашёлся факт про маршрут")
    out = await run_contexter(ctx, reply="как добраться до филиала?", tools=tools)
    assert out.static_text == "статика города"
    assert out.dynamic_status == DYN_READY
    assert "маршрут" in out.dynamic_text
    assert tools.calls == ["как добраться до филиала?"]


async def test_контекстер_справка_из_helps_в_динамику():
    helps = {
        "medcheck": Help(
            id="medcheck",
            triggers=["медкомисс", "медсправк"],
            text="Медкомиссия понадобится к началу практики.",
        )
    }
    ctx = ConversationContext(static_text="Город: Пермь")
    tools = _CountingTools()
    out = await run_contexter(
        ctx,
        reply="а когда медкомиссию проходить?",
        tools=tools,
        helps=helps,
    )
    assert out.dynamic_status == DYN_READY
    assert "Медкомиссия понадобится" in out.dynamic_text
    assert tools.calls == []
    assert out.static_text == "Город: Пермь"


async def test_контекстер_справка_без_текста_не_нашлось():
    helps = {
        "empty": Help(id="empty", triggers=["квантовая скидка"], text="   "),
    }
    ctx = ConversationContext()
    out = await run_contexter(
        ctx,
        reply="есть квантовая скидка?",
        tools=NullContexterTools(),
        helps=helps,
    )
    assert out.dynamic_status == DYN_MISSING
    assert out.dynamic_text == ""


async def test_контекстер_возражение_не_трогает():
    objections = {
        "think": Objection(
            id="think",
            triggers=["подумаю", "надо подумать"],
            text="Конечно, подумайте.",
            sets={"urgency": "high"},
        )
    }
    helps = {
        "medcheck": Help(
            id="medcheck",
            triggers=["медкомисс"],
            text="Медкомиссия понадобится.",
        )
    }
    ctx = ConversationContext(static_text="Город: Пермь", dynamic_text="было")
    tools = _CountingTools()
    out = await run_contexter(
        ctx,
        reply="я ещё подумаю",
        tools=tools,
        helps=helps,
        objections=objections,
    )
    assert out.dynamic_status == DYN_NONE
    assert out.dynamic_text == "было"
    assert tools.calls == []
