"""Тесты каркаса контекстера."""

from __future__ import annotations

from graph.context import (
    DYN_NONE,
    DYN_READY,
    DYN_SEARCHING,
    ConversationContext,
)
from graph.contexter import DEFAULT_SITUATION, NullContexterTools, run_contexter


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


async def test_контекстер_нет_в_статике_в_поиске_и_слаг():
    ctx = ConversationContext(static_text="Город: Пермь. Автопарк: Solaris.")
    tools = NullContexterTools()
    out = await run_contexter(ctx, reply="а как проходит медкомиссия?", tools=tools)
    assert out.dynamic_status == DYN_SEARCHING
    assert out.situation_slug == DEFAULT_SITUATION
    assert out.filler_spoken is False
    assert out.static_text == ctx.static_text


async def test_контекстер_статика_не_трогается_при_находке():
    ctx = ConversationContext(static_text="статика города")
    tools = _CountingTools(result="нашёлся факт про маршрут")
    out = await run_contexter(ctx, reply="как добраться до филиала?", tools=tools)
    assert out.static_text == "статика города"
    assert out.dynamic_status == DYN_READY
    assert "маршрут" in out.dynamic_text
    assert tools.calls == ["как добраться до филиала?"]
