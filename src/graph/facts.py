"""Факты из справочника: что принести шагу до вызова модели.

Ходим в справочник **всегда, когда положено по шагу**, а не когда модель
сочтёт нужным. Инструмент у модели означал бы два обращения к ней за ход —
решить позвать и переформулировать результат — плюс сам поход. В звонке это
лишняя секунда молчания, а главное — модель звала бы инструмент через раз и
остальные разы выдумывала.

Справочник отвечает из памяти процесса, ошибок наружу не бросает: при
недоступности отдаёт None и пустые списки. Разговор продолжается общими
словами — это хуже, чем с фактами, но лучше, чем упавший звонок.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from graph.tools_registry import needs_from_knowledge
from kb.client import VectorKBClient
from script.build import AnyStep, CompiledScript
from script.models import SalesStep
from script.price import price_facts, price_facts_from_kb

log = logging.getLogger(__name__)

#: Сколько филиалов перечисляем вслух. Районы в справочнике почти не
#: заполнены (три из двухсот тридцати пяти), поэтому филиал подбирается
#: перечислением адресов и ориентиров — но не всех сразу.
BRANCHES_TO_OFFER = 6


def city_choices(cities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Готовит перечень городов для выбора моделью.

    Слаги отдаются вместе с названиями: по одному слагу город не опознать,
    `kyrgan` это Курган, а `tagil` — Нижний Тагил.

    Args:
        cities: ответ справочника со списком городов.

    Returns:
        Список пар «слаг — название».
    """
    return [
        {"slug": c.get("slug"), "name": c.get("name")}
        for c in cities
        if c.get("slug") and c.get("name")
    ]


def branch_choices(branches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Готовит перечень филиалов для перечисления вслух.

    Args:
        branches: ответ справочника со списком филиалов города.

    Returns:
        Список филиалов со слагом, адресом и ориентиром.
    """
    picked = []
    for branch in list(branches)[:BRANCHES_TO_OFFER]:
        item: dict[str, Any] = {"slug": branch.get("slug"), "address": branch.get("address")}
        if branch.get("landmark"):
            item["landmark"] = branch["landmark"]
        picked.append(item)
    return picked


def city_summary(city: Mapping[str, Any]) -> dict[str, Any]:
    """Оставляет от меты города то, что реально произносится в разговоре.

    Цену сюда не кладём: она собирается отдельно, тремя ветками.

    Args:
        city: полная мета города из справочника.

    Returns:
        Сжатая выжимка для промпта.
    """
    vehicles = city.get("vehicles") or {}
    categories = city.get("categories") or []
    return {
        "city": city.get("name"),
        "branches_count": city.get("branches_count"),
        "categories": [
            {
                "code": c.get("code"),
                "duration": c.get("duration"),
                "start_frequency": c.get("start_frequency"),
            }
            for c in categories
        ],
        "vehicles_manual": vehicles.get("manual") or [],
        "vehicles_automatic": vehicles.get("automatic") or [],
        "fleet_age": vehicles.get("fleet_age"),
        "theory_formats": city.get("theory_formats") or [],
        "documents": city.get("documents") or [],
        "payment": city.get("payment") or {},
        "messengers": city.get("messengers") or [],
        "call_hours": city.get("call_hours"),
    }


def branch_summary(branch: Mapping[str, Any]) -> dict[str, Any]:
    """Оставляет от меты филиала то, что нужно для встречи.

    Учебный офис и автодром — разные вещи: на автодром за договором ехать не
    надо. У филиала со статусом «скоро открытие» часы пустые, записывать туда
    нельзя, но сказать, что он откроется, можно.

    Args:
        branch: полная мета филиала.

    Returns:
        Сжатая выжимка для промпта.
    """
    return {
        "address": branch.get("address"),
        "landmark": branch.get("landmark"),
        "place_type": branch.get("place_type"),
        "status": branch.get("status"),
        "working_hours": branch.get("working_hours"),
        "break_time": branch.get("break_time"),
        "can_sign_here": branch.get("place_type") != "автодром"
        and branch.get("status") != "скоро открытие",
    }


def needs_of(step: AnyStep | None) -> list[str]:
    """Что шаг просит принести из справочника.

    Старый формат — поле ``needs``. Новый — весь список ``knowledge``
    (чего в справочнике нет, то не мапится и не найдётся).

    Args:
        step: шаг этого хода или None.

    Returns:
        Список потребностей шага.
    """
    if step is None:
        return []
    if isinstance(step, SalesStep):
        return needs_from_knowledge(step.knowledge)
    return list(step.needs)


async def collect_facts(
    kb: VectorKBClient,
    *,
    script: CompiledScript,
    needs: Sequence[str],
    city_slug: str | None,
    branch_slug: str | None,
    want_city_choices: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Собирает факты справочника под потребности шага.

    Args:
        kb: клиент справочника.
        script: скомпилированный скрипт (нужен для текстов о цене).
        needs: что просит шаг.
        city_slug: подтверждённый город или None.
        branch_slug: подтверждённый филиал или None.
        want_city_choices: нужно ли положить в промпт перечень городов.

    Returns:
        Пара «факты для промпта, журнал походов».
    """
    facts: dict[str, Any] = {}
    journal: list[dict[str, Any]] = []

    if want_city_choices or "city_choices" in needs:
        cities = await kb.list_cities()
        journal.append({"call": "list_cities", "found": len(cities)})
        if cities:
            facts["city_choices"] = city_choices(cities)

    city_meta: Mapping[str, Any] | None = None
    if city_slug and ({"city_meta", "price"} & set(needs)):
        city_meta = await kb.get_city(city_slug)
        journal.append({"call": "get_city", "slug": city_slug, "ok": city_meta is not None})
        if city_meta and "city_meta" in needs:
            facts["city"] = city_summary(city_meta)

    if "price" in needs:
        price = (city_meta or {}).get("price") if city_meta else None
        if script.is_sales:
            facts["price"] = price_facts_from_kb(price)
        else:
            facts["price"] = price_facts(price, script.params.price)
        facts["price_line"] = facts["price"]["line"]

    if "branches" in needs and city_slug:
        branches = await kb.list_branches(city_slug)
        journal.append({"call": "list_branches", "slug": city_slug, "found": len(branches)})
        if branches:
            facts["branches"] = branch_choices(branches)
            facts["branches_total"] = len(branches)

    if "branch_meta" in needs and branch_slug:
        branch = await kb.get_branch(branch_slug)
        journal.append({"call": "get_branch", "slug": branch_slug, "ok": branch is not None})
        if branch:
            facts["branch"] = branch_summary(branch)
            facts["branch_address"] = branch.get("address")

    return facts, journal


async def confirm_city(kb: VectorKBClient, value: str) -> str | None:
    """Превращает названный клиентом город в слаг справочника.

    Порядок: если значение уже слаг из перечисления — берём его, промахнуться
    нельзя по построению. Иначе пробуем быстрый разбор на стороне сервиса; он
    понимает точное название и небольшой список разговорных вариантов, но не
    падежи, поэтому опорой не служит.

    Args:
        kb: клиент справочника.
        value: слаг или название города.

    Returns:
        Слаг города или None, если город не из сети.
    """
    if not value:
        return None
    candidate = value.strip().lower()
    known = await kb.cities_enum()
    if candidate in known:
        return candidate
    return await kb.resolve_city(value)


async def confirm_branch(kb: VectorKBClient, city_slug: str | None, value: str) -> str | None:
    """Проверяет слаг филиала по перечислению филиалов города.

    Args:
        kb: клиент справочника.
        city_slug: слаг города.
        value: слаг филиала, названный моделью.

    Returns:
        Слаг филиала или None, если такого филиала в городе нет.
    """
    if not value or not city_slug:
        return None
    candidate = value.strip().lower()
    known = await kb.branches_enum(city_slug)
    return candidate if candidate in known else None
