"""Реестр инструментов контекстера.

Новый источник (вектор, карты, полнотекст) = класс по ``ContextTool``
и одна строка в ``build_context_tools``. Агент выбирает инструмент по
описанию; код инструмента сам ходит в справочник. Решения о смысле
реплики инструменты не принимают.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, Sequence

from graph.context import ConversationContext, format_branch_static, merge_static
from graph.resolvers import CityResolver, resolve_city
from kb.client import vector_kb
from script.build import CompiledScript
from script.price import price_line, price_line_from_kb

log = logging.getLogger(__name__)

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

#: Потребности, которые закрывает ``FactsTool`` через ``collect_facts``.
FACT_NEEDS: frozenset[str] = frozenset(
    {"city_choices", "city_meta", "price", "branches", "branch_meta"}
)


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

    async def run(
        self,
        query: str,
        context: ConversationContext,
        *,
        slugs: Sequence[str] = (),
    ) -> str:
        """Ответ инструмента текстом для динамики; пустая строка — не нашлось."""
        ...


def _apply_merged(context: ConversationContext, merged: ConversationContext) -> None:
    """Переносит поля ``merge_static`` в переданный контекст на месте."""
    context.static_text = merged.static_text
    context.city_slug = merged.city_slug
    context.city_name = merged.city_name
    context.branch_slug = merged.branch_slug
    context.city_faq = list(merged.city_faq)
    context.frozen = merged.frozen


def _price_phrase(script: CompiledScript | None, price: Any) -> str | None:
    """Готовая фраза о цене из меты города."""
    if price is None or script is None:
        return None
    if script.is_sales:
        return price_line_from_kb(price)
    return price_line(price, script.params.price)


def _facts_to_dynamic(facts: dict[str, Any]) -> str:
    """Сериализует факты справочника в текст динамики."""
    payload = {
        key: value
        for key, value in facts.items()
        if value not in (None, "", [], {}) and key not in {"city_choices", "branches_total"}
    }
    if not payload:
        return ""
    return "Факты справочника:\n" + json.dumps(payload, ensure_ascii=False, indent=1)


class CityTool:
    """Город: перечень → резолвер → мета; статика через ``merge_static``."""

    name = "city"
    description = (
        "Зафиксировать город обучения по реплике клиента: слаг из перечня сети, "
        "мета города в статику."
    )

    def __init__(
        self,
        script: CompiledScript | None = None,
        *,
        resolver: CityResolver | None = None,
    ) -> None:
        """Сохраняет скрипт (для фразы цены) и подмену резолвера.

        Args:
            script: скомпилированный скрипт; нужен для ``price_line``.
            resolver: подмена резолвера города для офлайн-тестов.
        """
        self.script = script
        self.resolver = resolver

    async def run(
        self,
        query: str,
        context: ConversationContext,
        *,
        slugs: Sequence[str] = (),
        reply: str = "",
    ) -> str:
        """Резолвит город и кладёт статику; при районе — заметку в динамику.

        Args:
            query: название города от агента; может быть пустым.
            context: текущий контекст; поля города обновляются на месте.
            slugs: не используется; оставлен для единого интерфейса.
            reply: реплика клиента целиком; подставляется, если ``query`` пуст.

        Returns:
            Краткая строка при успехе / заметка про район; пустая — не нашлось.
        """
        _ = slugs
        text = (query or "").strip() or (reply or "").strip()
        preview = text[:60]
        if not text or context.city_slug:
            log.info(
                "CityTool: пустой запрос или город уже зафиксирован, query=%r",
                preview,
            )
            return ""
        cities = await vector_kb.list_cities()
        if not cities:
            log.info(
                "CityTool: пустой перечень городов из справочника, query=%r",
                preview,
            )
            return ""
        resolution = await resolve_city(text, cities, resolver=self.resolver)
        if resolution.is_district:
            log.info("CityTool: резолвер вернул район, query=%r", preview)
            return (
                "Клиент назвал район внутри города, а не город сети. "
                "Уточни город обучения, район городом не записывай."
            )
        if not resolution.slug or not resolution.name:
            log.info(
                "CityTool: резолвер не дал слаг или название, query=%r",
                preview,
            )
            return ""
        city_meta = await vector_kb.get_city(resolution.slug)
        if not city_meta:
            log.info(
                "CityTool: справочник не отдал мету города, query=%r slug=%r",
                preview,
                resolution.slug,
            )
            return ""
        price_phrase = _price_phrase(self.script, city_meta.get("price"))
        merged = merge_static(
            context,
            city_slug=resolution.slug,
            city_name=resolution.name,
            city_meta=city_meta,
            price_line=price_phrase,
        )
        _apply_merged(context, merged)
        log.info("CityTool: город найден, slug=%r query=%r", resolution.slug, preview)
        return f"Город: {resolution.name}."


class BranchesTool:
    """Подтверждение адресов филиалов, отобранных агентом по реплике."""

    name = "branches"
    description = (
        "Подтвердить адреса филиалов, которые ты отобрал по реплике клиента: "
        "в branch_slugs передай до трёх слагов из перечня города."
    )

    async def run(
        self,
        query: str,
        context: ConversationContext,
        *,
        slugs: Sequence[str] = (),
    ) -> str:
        """Собирает адреса филиалов строго по переданным слагам.

        Args:
            query: не используется; отбор уже сделан агентом.
            context: текущий контекст; нужен ``city_slug``.
            slugs: слаги филиалов в порядке агента, не больше трёх.

        Returns:
            Строка «Филиалы под запрос: …» или пустая, если города/слагов нет.
        """
        _ = query
        city_slug = (context.city_slug or "").strip()
        if not city_slug or not slugs:
            return ""
        branches = await vector_kb.list_branches(city_slug)
        if not branches:
            return ""
        by_slug = {str(b.get("slug")): b for b in branches if b.get("slug")}
        picked: list[Any] = []
        for slug in slugs:
            branch = by_slug.get(str(slug))
            if branch is None:
                continue
            picked.append(branch)
            if len(picked) >= 3:
                break
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

    async def run(
        self,
        query: str,
        context: ConversationContext,
        *,
        slugs: Sequence[str] = (),
    ) -> str:
        """Ищет точное совпадение вопроса в ``context.city_faq``.

        Сравнение по strip + lower, без матчинга по смыслу.

        Args:
            query: вопрос дословно из перечня, который видел агент.
            context: контекст с ``city_faq``.
            slugs: не используется; оставлен для единого интерфейса.

        Returns:
            Текст ответа или пустая строка при отсутствии совпадения.
        """
        _ = slugs
        needle = (query or "").strip().lower()
        if not needle:
            return ""
        for item in context.city_faq or []:
            question = str(item.get("question") or "").strip().lower()
            if question == needle:
                return str(item.get("answer") or "").strip()
        return ""


class BranchDetailsTool:
    """Адрес, часы и ориентир уже выбранного филиала — в статику."""

    name = "branch_details"
    description = "Адрес, часы и ориентир уже выбранного филиала."

    async def run(
        self,
        query: str,
        context: ConversationContext,
        *,
        slugs: Sequence[str] = (),
    ) -> str:
        """Тянет мету филиала и кладёт её в статику через ``merge_static``.

        Args:
            query: не используется; оставлен для единого интерфейса.
            context: контекст с непустым ``branch_slug`` (или ``slugs[0]``).
            slugs: необязательный слаг, если в контексте ещё не зафиксирован.

        Returns:
            Текстовый блок филиала или пустая строка без слага / меты.
        """
        _ = query
        branch_slug = (context.branch_slug or "").strip()
        if not branch_slug and slugs:
            branch_slug = str(slugs[0]).strip()
        if not branch_slug:
            return ""
        branch = await vector_kb.get_branch(branch_slug)
        if not branch:
            return ""
        merged = merge_static(context, branch_slug=branch_slug, branch_meta=branch)
        _apply_merged(context, merged)
        return format_branch_static(branch, branch_slug=branch_slug)


class FactsTool:
    """Факты и цена по потребностям шага через ``collect_facts``."""

    name = "facts"
    description = (
        "Факты справочника под шаг: мета города, цена, перечень филиалов. "
        "Потребности задаёт код шапки, не реплика."
    )

    def __init__(
        self,
        script: CompiledScript,
        *,
        needs: Sequence[str] = (),
    ) -> None:
        """Сохраняет скрипт и текущий набор потребностей.

        Args:
            script: скомпилированный скрипт (фразы цены).
            needs: потребности справочника; контекстер может обновить на ходу.
        """
        self.script = script
        self.needs: list[str] = list(needs)

    async def run(
        self,
        query: str,
        context: ConversationContext,
        *,
        slugs: Sequence[str] = (),
    ) -> str:
        """Зовёт ``collect_facts`` и при необходимости дописывает статику города.

        Args:
            query: не используется; потребности уже в ``self.needs``.
            context: текущий контекст (слаги города/филиала).
            slugs: не используется; оставлен для единого интерфейса.

        Returns:
            Текст фактов для динамики или пустая строка.
        """
        _ = query
        _ = slugs
        needs = [n for n in self.needs if n in FACT_NEEDS]
        if not needs:
            return ""
        from graph.facts import collect_facts

        city_slug = (context.city_slug or "").strip() or None
        branch_slug = (context.branch_slug or "").strip() or None
        want_choices = "city_choices" in needs
        try:
            facts, _journal = await collect_facts(
                vector_kb,
                script=self.script,
                needs=needs,
                city_slug=city_slug,
                branch_slug=branch_slug,
                want_city_choices=want_choices,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("FactsTool: collect_facts не удался: %s", exc)
            return ""
        if city_slug and not context.city_slug and facts.get("city"):
            raw_meta = await vector_kb.get_city(city_slug)
            if raw_meta:
                city_name = context.city_name or city_slug
                merged = merge_static(
                    context,
                    city_slug=city_slug,
                    city_name=str(city_name),
                    city_meta=raw_meta,
                    price_line=facts.get("price_line"),
                )
                _apply_merged(context, merged)
        return _facts_to_dynamic(facts)


def build_context_tools(script: CompiledScript) -> list[ContextTool]:
    """Собирает реестр инструментов контекстера.

    Город, филиалы, FAQ, детали филиала и факты шага.

    Args:
        script: скомпилированный скрипт (цена и ``FactsTool``).

    Returns:
        Список инструментов для агента и исполнения потребностей шага.
    """
    return [
        CityTool(script),
        BranchesTool(),
        CityFaqTool(),
        BranchDetailsTool(),
        FactsTool(script),
    ]
