"""Тесты каркаса контекстера, справок и реестра инструментов."""

from __future__ import annotations

from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    ConversationContext,
)
from graph.contexter import run_contexter
from graph.tools_registry import HelpsTool, build_context_tools
from script.models import Help, Objection


class _FakeTool:
    """Фейковый инструмент реестра: отвечает по подстроке."""

    def __init__(self, name: str, needle: str, answer: str | None) -> None:
        self.name = name
        self.needle = needle
        self.answer = answer
        self.calls: list[str] = []

    async def try_answer(self, reply: str, context: ConversationContext) -> str | None:
        self.calls.append(reply)
        if self.needle.lower() in (reply or "").lower():
            return self.answer
        return None


async def test_контекстер_нечего_добавлять_статус_не_требуется():
    ctx = ConversationContext(static_text="Город: Пермь")
    out = await run_contexter(ctx, reply="да, хорошо", tools=[])
    assert out.dynamic_status == DYN_NONE
    assert out.situation_slug is None
    assert out.static_text == "Город: Пермь"


async def test_контекстер_ответ_в_статике_готово_без_инструментов():
    ctx = ConversationContext(
        static_text=("Город: Пермь.\nЦена (готовая фраза): Стоимость обучения — от 43900 рублей.")
    )
    out = await run_contexter(ctx, reply="а сколько стоимость обучения?", tools=[])
    assert out.dynamic_status == DYN_READY
    assert out.static_text == ctx.static_text


async def test_контекстер_нет_в_статике_и_реестре_не_нашлось():
    ctx = ConversationContext(static_text="Город: Пермь. Автопарк: Solaris.")
    out = await run_contexter(
        ctx,
        reply="а как записаться на дайвинг?",
        tools=[],
    )
    assert out.dynamic_status == DYN_MISSING
    assert out.situation_slug is None
    assert out.static_text == ctx.static_text


async def test_контекстер_статика_не_трогается_при_находке():
    ctx = ConversationContext(static_text="статика города")
    fake = _FakeTool("maps", "добраться", "нашёлся факт про маршрут")
    out = await run_contexter(ctx, reply="как добраться до филиала?", tools=[fake])
    assert out.static_text == "статика города"
    assert out.dynamic_status == DYN_READY
    assert "маршрут" in out.dynamic_text
    assert fake.calls == ["как добраться до филиала?"]


async def test_helps_tool_справка_из_helps_в_динамику():
    helps = {
        "medcheck": Help(
            id="medcheck",
            triggers=["медкомисс", "медсправк"],
            text="Медкомиссия понадобится к началу практики.",
        )
    }
    ctx = ConversationContext(static_text="Город: Пермь")
    out = await run_contexter(
        ctx,
        reply="а когда медкомиссию проходить?",
        tools=[HelpsTool(helps)],
    )
    assert out.dynamic_status == DYN_READY
    assert "Медкомиссия понадобится" in out.dynamic_text
    assert out.static_text == "Город: Пермь"


async def test_helps_tool_справка_без_текста_не_нашлось():
    helps = {
        "empty": Help(id="empty", triggers=["квантовая скидка"], text="   "),
    }
    ctx = ConversationContext()
    out = await run_contexter(
        ctx,
        reply="есть квантовая скидка?",
        tools=[HelpsTool(helps)],
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
    fake = _FakeTool("x", "подумаю", "не должен ответить")
    out = await run_contexter(
        ctx,
        reply="я ещё подумаю",
        tools=[fake, HelpsTool(helps)],
        objections=objections,
    )
    assert out.dynamic_status == DYN_NONE
    assert out.dynamic_text == "было"
    assert fake.calls == []


async def test_реестр_фейковый_инструмент_подхватывается(script):
    """Дописали инструмент в список — контекстер видит без правок себя."""
    fake = _FakeTool("vector", "дайвинг", "запись на дайвинг через приложение")
    tools = [fake, *build_context_tools(script)]
    ctx = ConversationContext(static_text="Город: Пермь")
    out = await run_contexter(
        ctx,
        reply="а как записаться на дайвинг?",
        tools=tools,
    )
    assert out.dynamic_status == DYN_READY
    assert "дайвинг" in out.dynamic_text
    assert fake.calls == ["а как записаться на дайвинг?"]


async def test_реестр_приоритет_первый_подходящий(script):
    """Два инструмента подходят — отвечает первый по списку."""
    first = _FakeTool("a", "медкомисс", "ответ первого")
    second = _FakeTool("b", "медкомисс", "ответ второго")
    out = await run_contexter(
        ConversationContext(),
        reply="а когда медкомиссию проходить?",
        tools=[first, second],
    )
    assert out.dynamic_status == DYN_READY
    assert "ответ первого" in out.dynamic_text
    assert "ответ второго" not in out.dynamic_text
    assert first.calls == ["а когда медкомиссию проходить?"]
    assert second.calls == []


async def test_build_context_tools_включает_helps(script):
    """Реестр по умолчанию содержит HelpsTool со справками скрипта."""
    tools = build_context_tools(script)
    assert len(tools) >= 1
    assert tools[0].name == "helps"
    med = script.helps["medcheck"].text
    out = await run_contexter(
        ConversationContext(),
        reply="а когда медкомиссию проходить?",
        tools=tools,
    )
    assert out.dynamic_status == DYN_READY
    assert med in out.dynamic_text
