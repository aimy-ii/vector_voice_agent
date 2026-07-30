"""Реестр инструментов контекстера.

Новый источник (вектор, карты, полнотекст) = класс по ``ContextTool``
и одна строка в ``build_context_tools``. Агент выбирает инструмент по
описанию; код инструмента сам ходит в справочник.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from graph.context import ConversationContext, format_branch_static
from graph.resolvers import resolve_branch
from kb.client import vector_kb
from script.build import CompiledScript

#: Человекочитаемые факты из ``knowledge`` → потребности справочника.
#: Чего в справочнике нет — просто не найдётся, шаг отработает без чисел.
_KNOWLEDGE_TO_NEED: dict[str, str] = {
    "перечень городов сети": "city_choices",
    "автопарк города по коробке передач": "city_meta",
    "срок обучения по городу": "city_meta",
    "форматы теории в городе": "city_meta",
    "что входит в стоимость курса": "city_meta",
    "график занятий и адреса филиалов": "city_meta",
    "филиалы города с адресами": "branches",
    "стоимость обучения в городе": "price",
    "адрес выбранного филиала": "branch_meta",
    "документы для оформления": "city_meta",
    "категории обучения кроме легковой": "city_meta",
    "мессенджеры, доступные в городе": "city_meta",
}


def needs_from_knowledge(knowledge: Sequence[str]) -> list[str]:
    """Превращает список ``knowledge`` в потребности справочника для прогрева.

    Args:
        knowledge: факты, которые шаг ищет в базе знаний.

    Returns:
        Уникальный список ключей ``NeedKind`` / ``city_choices`` в стабильном порядке.
        Неизвестные факты пропускаются — в справочнике их нет.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for fact in knowledge:
        need = _KNOWLEDGE_TO_NEED.get(str(fact).strip())
        if need and need not in seen:
            seen.add(need)
            ordered.append(need)
    return ordered


class ContextTool(Protocol):
    """Инструмент контекстера: агент выбирает его по описанию."""

    name: str
    description: str

    async def run(self, query: str, context: ConversationContext) -> str:
        """Ответ инструмента текстом для динамики; пустая строка — не нашлось."""
        ...


class BranchesTool:
    """Подбор филиалов по району, улице, ориентиру или перечень «какие есть»."""

    name = "branches"
    description = (
        "Подобрать филиалы по району, улице, ориентиру или перечислить, "
        "когда клиент спрашивает «а какие есть»."
    )

    async def run(self, query: str, context: ConversationContext) -> str:
        """Отбирает до трёх филиалов города по запросу.

        Args:
            query: район, улица, ориентир или пусто для перечня.
            context: текущий контекст; нужен ``city_slug``.

        Returns:
            Строка «Филиалы под запрос: …» или пустая, если города/списка нет.
        """
        city_slug = (context.city_slug or "").strip()
        if not city_slug:
            return ""
        branches = await vector_kb.list_branches(city_slug)
        if not branches:
            return ""
        resolution = await resolve_branch(query or "", branches)
        by_slug = {str(b.get("slug")): b for b in branches if b.get("slug")}
        picked = [by_slug[s] for s in resolution.slugs if s in by_slug]
        if not picked and resolution.selected and resolution.selected in by_slug:
            picked = [by_slug[resolution.selected]]
        if not picked:
            return ""
        parts: list[str] = []
        for branch in picked:
            address = str(branch.get("address") or "").strip()
            if not address:
                continue
            landmark = str(branch.get("landmark") or "").strip()
            parts.append(f"{address} ({landmark})" if landmark else address)
        if not parts:
            return ""
        return "Филиалы под запрос: " + "; ".join(parts) + "."


class CityFaqTool:
    """Типовые вопросы по городу с готовыми ответами из меты."""

    name = "city_faq"
    description = (
        "Типовые вопросы по городу с готовыми ответами. "
        "В query передай вопрос дословно из перечня FAQ."
    )

    async def run(self, query: str, context: ConversationContext) -> str:
        """Ищет точное совпадение вопроса в ``context.city_faq``.

        Сравнение по strip + lower, без матчинга по смыслу.

        Args:
            query: вопрос дословно из перечня, который видел агент.
            context: контекст с ``city_faq``.

        Returns:
            Текст ответа или пустая строка при отсутствии совпадения.
        """
        needle = (query or "").strip().lower()
        if not needle:
            return ""
        for item in context.city_faq or []:
            question = str(item.get("question") or "").strip().lower()
            if question == needle:
                return str(item.get("answer") or "").strip()
        return ""


class BranchDetailsTool:
    """Адрес, часы и ориентир уже выбранного филиала."""

    name = "branch_details"
    description = "Адрес, часы и ориентир уже выбранного филиала."

    async def run(self, query: str, context: ConversationContext) -> str:
        """Тянет мету филиала по ``context.branch_slug``.

        Args:
            query: не используется; оставлен для единого интерфейса.
            context: контекст с непустым ``branch_slug``.

        Returns:
            Текстовый блок филиала или пустая строка без слага / меты.
        """
        _ = query
        branch_slug = (context.branch_slug or "").strip()
        if not branch_slug:
            return ""
        branch = await vector_kb.get_branch(branch_slug)
        if not branch:
            return ""
        return format_branch_static(branch, branch_slug=branch_slug)


def build_context_tools(script: CompiledScript) -> list[ContextTool]:
    """Собирает реестр инструментов контекстера.

    Филиалы и FAQ всегда; ``branch_details`` всегда (сам отсеется без слага).
    Справок скрипта в продажах нет по замыслу — ветки ``is_sales`` нет.

    Args:
        script: скомпилированный скрипт (сигнатура сохранена для вызывающих).

    Returns:
        Список инструментов для агента контекста.
    """
    _ = script  # реестр не зависит от скрипта; сигнатура наружу та же
    return [
        BranchesTool(),
        CityFaqTool(),
        BranchDetailsTool(),
    ]
