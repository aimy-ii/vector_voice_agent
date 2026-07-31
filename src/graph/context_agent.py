"""Агент контекста: нужен ли поход за данными и каким инструментом.

Быстрая модель, короткая схема. Контекстер — оптимизация, не обязанность:
ошибка или таймаут модели → пустое решение, ход не роняем.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from graph.context import ConversationContext
from graph.tools_registry import ContextTool
from utils.llm_gen import astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)

#: Сколько символов хвоста динамики отдаём агенту, чтобы не искать повторно.
_DYNAMIC_TAIL = 400


class ContextDecision(BaseModel):
    """Решение агента контекста по реплике клиента."""

    need: bool = Field(default=False, description="Нужен ли поход за контекстом.")
    tool: str | None = Field(default=None, description="Имя инструмента строго из перечня.")
    query: str = Field(
        default="",
        description=(
            "Запрос инструменту: район, ориентир или дословный вопрос из перечня FAQ. "
            "Для инструмента города сюда кладётся название города из реплики клиента; "
            "поле пустым не оставлять, если инструмент выбран."
        ),
    )
    subject: str = Field(
        default="",
        description="Предмет вопроса одним-двумя словами для фразы-заглушки.",
    )
    branch_slugs: list[str] = Field(
        default_factory=list,
        description="Слаги филиалов, отобранные по реплике: не больше трёх, строго из перечня.",
    )


class ContextAgent(Protocol):
    """Контракт агента контекста."""

    async def decide(
        self,
        reply: str,
        context: ConversationContext,
        tools: Sequence[ContextTool],
        faq_questions: Sequence[str],
        branches: Sequence[Mapping[str, Any]],
    ) -> ContextDecision:
        """Решает, нужен ли контекст и каким инструментом его брать."""


def _format_branches(branches: Sequence[Mapping[str, Any]]) -> str:
    """Строки «слаг — адрес (ориентир)» для промпта решения."""
    lines: list[str] = []
    for branch in branches:
        slug = str(branch.get("slug") or "").strip()
        if not slug:
            continue
        address = str(branch.get("address") or "").strip()
        landmark = str(branch.get("landmark") or "").strip()
        if address and landmark:
            place = f"{address} ({landmark})"
        else:
            place = address or landmark or "—"
        lines.append(f"- {slug} — {place}")
    return "\n".join(lines)


class LlmContextAgent:
    """Агент контекста на быстрой модели."""

    async def decide(
        self,
        reply: str,
        context: ConversationContext,
        tools: Sequence[ContextTool],
        faq_questions: Sequence[str],
        branches: Sequence[Mapping[str, Any]],
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
        bot_reply = (context.last_agent_reply or "").strip() or "—"
        dynamic_tail = (context.dynamic_text or "").strip()
        if len(dynamic_tail) > _DYNAMIC_TAIL:
            dynamic_tail = dynamic_tail[-_DYNAMIC_TAIL:]
        dynamic_block = dynamic_tail or "—"
        if branches:
            branches_block = "Филиалы города:\n" + _format_branches(branches)
        else:
            count_line = ""
            for line in (context.static_text or "").splitlines():
                if line.startswith("Филиалов в городе:"):
                    count_line = line.strip()
                    break
            branches_block = count_line or "Список филиалов не передан."
        system = (
            "Ты решаешь, нужен ли дополнительный контекст под реплику клиента.\n"
            "Общая информация по городу уже лежит в статике — повторно её искать не надо.\n"
            "Имя инструмента — строго из перечня.\n"
            "Для city_faq в query положи вопрос дословно из перечня FAQ.\n"
            "Реплика без вопроса и без просьбы — need=false; короткие подтверждения "
            "(«да», «угу», «механика», «впервые») контекста не требуют, даже если тема "
            "разговора похожа на тему инструмента.\n"
            "Смотри на реплику бота: если бот только что задал вопрос, ответ клиента "
            "на него — не запрос контекста.\n"
            "Для branches отбирай слаги сам по названному району, улице или ориентиру, "
            "зная город; человек просит перечень — верни три ближайших к центру или "
            "произвольных три.\n"
            "subject — одно-два слова в именительном падеже, без города и предлогов: "
            "«филиалы», «медкомиссия», «пересдача». Это будет произнесено вслух "
            "в составе фразы."
        )
        human = (
            f"Реплика клиента: {reply}\n"
            f"Последняя реплика бота: {bot_reply}\n"
            f"Уже найдено (хвост динамики): {dynamic_block}\n"
            f"Город известен: {city_known}. Филиал известен: {branch_known}.\n"
            f"{branches_block}\n"
            f"Инструменты:\n{tool_lines or '— нет'}\n"
            f"{faq_block}"
        )
        log.debug("Агент контекста: системное сообщение: %s", system)
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
        except Exception as exc:  # noqa: BLE001
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


def _truncate_subject(subject: str) -> str:
    """Обрезает предмет до трёх слов; пустой остаётся пустым."""
    words = (subject or "").split()
    if not words:
        return ""
    return " ".join(words[:3])


def _valid_branch_slugs(
    slugs: Sequence[str],
    branches: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Оставляет только слаги из перечня, не больше трёх, порядок сохраняет.

    Пустой перечень — слаги пока не фильтруем: списка ещё нет, контекстер
    подгрузит филиалы после выбора инструмента и проверит слаги сам.
    """
    out: list[str] = []
    if not branches:
        for slug in slugs:
            text = str(slug).strip()
            if not text or text in out:
                continue
            out.append(text)
            if len(out) >= 3:
                break
        return out
    allowed = {str(b.get("slug")) for b in branches if b.get("slug")}
    for slug in slugs:
        text = str(slug).strip()
        if not text or text not in allowed or text in out:
            continue
        out.append(text)
        if len(out) >= 3:
            break
    return out


async def decide_context(
    reply: str,
    context: ConversationContext,
    tools: Sequence[ContextTool],
    *,
    branches: Sequence[Mapping[str, Any]] = (),
    agent: ContextAgent | None = None,
) -> ContextDecision:
    """Точка входа агента контекста с валидацией решения.

    Args:
        reply: реплика клиента.
        context: текущий контекст разговора.
        tools: реестр доступных инструментов.
        branches: филиалы города для отбора слагов; пусто — списка нет.
        agent: подмена для офлайн-тестов.

    Returns:
        Решение; ``tool`` вне реестра → ``need=False``. Ошибка агента
        наружу не летит — пустое решение.
    """
    worker = agent or LlmContextAgent()
    known = {t.name for t in tools}
    try:
        decision = await worker.decide(reply, context, tools, _faq_questions(context), branches)
    except Exception as exc:  # noqa: BLE001
        log.warning("Агент контекста упал: %s", exc)
        return ContextDecision()

    log.info(
        "Агент контекста: need=%s tool=%r query=%r subject=%r branch_slugs=%r реплика=%r",
        decision.need,
        decision.tool,
        decision.query,
        decision.subject,
        decision.branch_slugs,
        (reply or "")[:80],
    )

    subject = _truncate_subject(decision.subject)
    branch_slugs = _valid_branch_slugs(decision.branch_slugs, branches)

    if decision.need and (not decision.tool or decision.tool not in known):
        return ContextDecision(
            need=False,
            tool=None,
            query="",
            subject=subject,
            branch_slugs=branch_slugs,
        )

    if decision.need and decision.tool == "branches" and not branch_slugs:
        return ContextDecision(
            need=False,
            tool=None,
            query="",
            subject=subject,
            branch_slugs=[],
        )

    return ContextDecision(
        need=decision.need,
        tool=decision.tool,
        query=decision.query,
        subject=subject,
        branch_slugs=branch_slugs,
    )
