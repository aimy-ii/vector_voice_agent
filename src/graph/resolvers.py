"""Служебные вызовы: фиксация города и филиала.

Быстрая модель, короткие схемы. Слаги проверяются по перечню на выходе —
промахнуться мимо справочника нельзя по построению. После фиксации города
вызов не повторяется. Перечень городов в промпт генератора не попадает.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from utils.llm_gen import LLMTurnFailed, astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)


class CityResolution(BaseModel):
    """Результат разбора города из свободной речи."""

    slug: str | None = Field(
        default=None,
        description="Слаг города из переданного перечня или пусто.",
    )
    name: str | None = Field(
        default=None,
        description="Читаемое название города из перечня или пусто.",
    )
    is_district: bool = Field(
        default=False,
        description=(
            "True, если назван район внутри города, а не город сети "
            "(например, «Красное Село» — район Петербурга)."
        ),
    )


class BranchResolution(BaseModel):
    """Результат отбора филиалов по свободной речи."""

    slugs: list[str] = Field(
        default_factory=list,
        description="Слаги релевантных филиалов, не больше трёх, только из списка.",
    )
    selected: str | None = Field(
        default=None,
        description="Слаг однозначно выбранного филиала или пусто, если выбора нет.",
    )


class CityResolver(Protocol):
    """Контракт резолвера города."""

    async def resolve(
        self,
        text: str,
        cities: Sequence[Mapping[str, Any]],
    ) -> CityResolution:
        """Разбирает текст в слаг и название."""


class BranchResolver(Protocol):
    """Контракт резолвера филиала."""

    async def resolve(
        self,
        text: str,
        branches: Sequence[Mapping[str, Any]],
    ) -> BranchResolution:
        """Отбирает до трёх релевантных филиалов."""


def exact_city_match(
    text: str,
    cities: Sequence[Mapping[str, Any]],
) -> CityResolution | None:
    """Быстрый путь: точное совпадение названия или слага без вызова модели.

    Args:
        text: реплика клиента.
        cities: перечень городов справочника.

    Returns:
        Фиксация или None, если точного совпадения нет.
    """
    needle = text.strip().lower().replace("ё", "е")
    if not needle:
        return None
    for city in cities:
        slug = str(city.get("slug") or "").strip()
        name = str(city.get("name") or "").strip()
        if not slug or not name:
            continue
        if needle == slug.lower() or needle == name.lower().replace("ё", "е"):
            return CityResolution(slug=slug, name=name, is_district=False)
    return None


def _validate_city(
    result: CityResolution,
    cities: Sequence[Mapping[str, Any]],
) -> CityResolution:
    """Отбрасывает слаг вне перечня."""
    if result.is_district:
        return CityResolution(slug=None, name=None, is_district=True)
    known = {str(c.get("slug")): str(c.get("name")) for c in cities if c.get("slug")}
    if result.slug and result.slug in known:
        return CityResolution(
            slug=result.slug,
            name=known[result.slug],
            is_district=False,
        )
    return CityResolution(slug=None, name=None, is_district=False)


def _validate_branches(
    result: BranchResolution,
    branches: Sequence[Mapping[str, Any]],
) -> BranchResolution:
    """Оставляет не больше трёх слагов из списка города."""
    known = {str(b.get("slug")) for b in branches if b.get("slug")}
    slugs = [slug for slug in result.slugs if slug in known][:3]
    selected = result.selected if result.selected in known else None
    if selected and selected not in slugs:
        slugs = [selected, *[s for s in slugs if s != selected]][:3]
    return BranchResolution(slugs=slugs, selected=selected)


class LlmCityResolver:
    """Резолвер города на быстрой модели."""

    async def resolve(
        self,
        text: str,
        cities: Sequence[Mapping[str, Any]],
    ) -> CityResolution:
        """Разбирает город; слаг вне перечня отвергается."""
        exact = exact_city_match(text, cities)
        if exact is not None:
            return exact
        catalogue = "\n".join(
            f"- {c.get('slug')}: {c.get('name')}" for c in cities if c.get("slug") and c.get("name")
        )
        system = (
            "Определи город обучения по реплике клиента.\n"
            "Верни слаг и название СТРОГО из перечня.\n"
            "Если назван район внутри города («Красное Село», «Люблино»), "
            "а не город сети — is_district=true, slug и name пустые.\n"
            "Если города нет в перечне — всё пустое, is_district=false."
        )
        human = f"Перечень городов:\n{catalogue}\n\nРеплика: {text}"
        schema = response_format_from(CityResolution, name="vector_city")
        try:
            async with get_llm(fast=True, temperature=0.0) as llm:
                raw = await astream_structured(
                    llm,
                    [SystemMessage(content=system), HumanMessage(content=human)],
                    schema=schema,
                    text_field=None,
                )
            return _validate_city(CityResolution.model_validate(raw), cities)
        except (LLMTurnFailed, Exception) as exc:  # noqa: BLE001
            log.warning("Резолвер города не ответил: %s", exc)
            return CityResolution()


class LlmBranchResolver:
    """Резолвер филиала на быстрой модели."""

    async def resolve(
        self,
        text: str,
        branches: Sequence[Mapping[str, Any]],
    ) -> BranchResolution:
        """Отбирает до трёх филиалов; мета не тянется здесь."""
        catalogue = "\n".join(
            "- {slug}: {address}{landmark}".format(
                slug=b.get("slug"),
                address=b.get("address") or "",
                landmark=f" ({b['landmark']})" if b.get("landmark") else "",
            )
            for b in branches
            if b.get("slug")
        )
        system = (
            "Подбери филиалы по району, улице или ориентиру из реплики.\n"
            "Верни не больше трёх слагов ТОЛЬКО из списка.\n"
            "Если филиал однозначен — заполни selected тем же слагом.\n"
            "Если человек просит перечень или район не матчится — "
            "selected пустой, в slugs до трёх кандидатов для озвучки."
        )
        human = f"Филиалы города:\n{catalogue}\n\nРеплика: {text}"
        schema = response_format_from(BranchResolution, name="vector_branch")
        try:
            async with get_llm(fast=True, temperature=0.0) as llm:
                raw = await astream_structured(
                    llm,
                    [SystemMessage(content=system), HumanMessage(content=human)],
                    schema=schema,
                    text_field=None,
                )
            return _validate_branches(BranchResolution.model_validate(raw), branches)
        except (LLMTurnFailed, Exception) as exc:  # noqa: BLE001
            log.warning("Резолвер филиала не ответил: %s", exc)
            return BranchResolution()


async def resolve_city(
    text: str,
    cities: Sequence[Mapping[str, Any]],
    *,
    resolver: CityResolver | None = None,
) -> CityResolution:
    """Точка входа резолвера города.

    Args:
        text: реплика клиента.
        cities: перечень городов.
        resolver: подмена для тестов.

    Returns:
        Результат с проверенным слагом.
    """
    exact = exact_city_match(text, cities)
    if exact is not None:
        return exact
    worker = resolver or LlmCityResolver()
    return _validate_city(await worker.resolve(text, cities), cities)


async def resolve_branch(
    text: str,
    branches: Sequence[Mapping[str, Any]],
    *,
    resolver: BranchResolver | None = None,
) -> BranchResolution:
    """Точка входа резолвера филиала.

    Args:
        text: реплика клиента.
        branches: филиалы города.
        resolver: подмена для тестов.

    Returns:
        До трёх слагов из списка города.
    """
    worker = resolver or LlmBranchResolver()
    return _validate_branches(await worker.resolve(text, branches), branches)
