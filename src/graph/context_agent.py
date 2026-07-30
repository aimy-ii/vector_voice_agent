"""Агент контекста: нужен ли поход за данными и каким инструментом.

Быстрая модель, короткая схема. Контекстер — оптимизация, не обязанность:
ошибка или таймаут модели → пустое решение, ход не роняем.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from graph.context import ConversationContext
from graph.tools_registry import ContextTool
from utils.llm_gen import LLMTurnFailed, astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)


class ContextDecision(BaseModel):
    """Решение агента контекста по реплике клиента."""

    need: bool = Field(default=False, description="Нужен ли поход за контекстом.")
    tool: str | None = Field(default=None, description="Имя инструмента строго из перечня.")
    query: str = Field(
        default="",
        description="Запрос инструменту: район, ориентир или дословный вопрос из перечня FAQ.",
    )
    subject: str = Field(
        default="",
        description="Предмет вопроса одним-двумя словами для фразы-заглушки.",
    )


class ContextAgent(Protocol):
    """Контракт агента контекста."""

    async def decide(
        self,
        reply: str,
        context: ConversationContext,
        tools: Sequence[ContextTool],
        faq_questions: Sequence[str],
    ) -> ContextDecision:
        """Решает, нужен ли контекст и каким инструментом его брать."""


class LlmContextAgent:
    """Агент контекста на быстрой модели."""

    async def decide(
        self,
        reply: str,
        context: ConversationContext,
        tools: Sequence[ContextTool],
        faq_questions: Sequence[str],
    ) -> ContextDecision:
        """Один вызов быстрой модели; при ошибке — пустое решение."""
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        faq_block = ""
        if faq_questions:
            faq_block = "Перечень вопросов FAQ города (без ответов):\n" + "\n".join(
                f"- {q}" for q in faq_questions
            )
        city_known = "да" if context.city_slug else "нет"
        branch_known = "да" if context.branch_slug else "нет"
        system = (
            "Ты решаешь, нужен ли дополнительный контекст под реплику клиента.\n"
            "Общая информация по городу уже лежит в статике — повторно её искать не надо.\n"
            "Имя инструмента — строго из перечня.\n"
            "Для city_faq в query положи вопрос дословно из перечня FAQ.\n"
            "subject — коротко и по-русски: это будет произнесено вслух в составе фразы."
        )
        human = (
            f"Реплика клиента: {reply}\n"
            f"Город известен: {city_known}. Филиал известен: {branch_known}.\n"
            f"Инструменты:\n{tool_lines or '— нет'}\n"
            f"{faq_block}"
        )
        schema = response_format_from(ContextDecision, name="vector_context")
        try:
            async with get_llm(fast=True, temperature=0.0) as llm:
                raw = await astream_structured(
                    llm,
                    [SystemMessage(content=system), HumanMessage(content=human)],
                    schema=schema,
                    text_field=None,
                    purpose="контекст",
                )
            return ContextDecision.model_validate(raw)
        except (LLMTurnFailed, Exception) as exc:  # noqa: BLE001
            log.warning("Агент контекста не ответил: %s", exc)
            return ContextDecision()


def _faq_questions(context: ConversationContext) -> list[str]:
    """Достаёт только тексты вопросов из FAQ города."""
    questions: list[str] = []
    for item in context.city_faq or []:
        question = str(item.get("question") or "").strip()
        if question:
            questions.append(question)
    return questions


async def decide_context(
    reply: str,
    context: ConversationContext,
    tools: Sequence[ContextTool],
    *,
    agent: ContextAgent | None = None,
) -> ContextDecision:
    """Точка входа агента контекста с валидацией решения.

    Args:
        reply: реплика клиента.
        context: текущий контекст разговора.
        tools: реестр доступных инструментов.
        agent: подмена для офлайн-тестов.

    Returns:
        Решение; ``tool`` вне реестра → ``need=False``. Ошибка агента
        наружу не летит — пустое решение.
    """
    worker = agent or LlmContextAgent()
    known = {t.name for t in tools}
    try:
        decision = await worker.decide(reply, context, tools, _faq_questions(context))
    except Exception as exc:  # noqa: BLE001
        log.warning("Агент контекста упал: %s", exc)
        return ContextDecision()
    if decision.need and (not decision.tool or decision.tool not in known):
        return ContextDecision(
            need=False,
            tool=None,
            query="",
            subject=decision.subject,
        )
    return decision
