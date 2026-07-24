"""Контекст разговора: один накопительный документ.

Статика кладётся один раз и до конца разговора не меняется — её нельзя резать.
Динамика — место под следующий этап; сейчас пустая. В промпт документ
подшивается целиком.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel


class ConversationContext(BaseModel):
    """Единый контекст разговора.

    Attributes:
        static_text: запечённая статика города и филиала.
        dynamic_text: динамическая часть (пока пусто — следующий этап).
        city_slug: слаг города после фиксации.
        city_name: читаемое название города.
        branch_slug: слаг выбранного филиала.
        frozen: статика уже зафиксирована и не пересобирается.
    """

    static_text: str = ""
    dynamic_text: str = ""
    city_slug: str | None = None
    city_name: str | None = None
    branch_slug: str | None = None
    frozen: bool = False

    def render(self) -> str:
        """Собирает документ для промпта: статика, затем динамика.

        Returns:
            Текст контекста; пустая строка, если ещё нечего класть.
        """
        parts: list[str] = []
        if self.static_text.strip():
            parts.append(self.static_text.strip())
        if self.dynamic_text.strip():
            parts.append(self.dynamic_text.strip())
        return "\n\n".join(parts)


def _line(label: str, value: Any) -> str | None:
    """Строка «метка: значение», если значение непустое."""
    if value is None or value == "" or value == [] or value == {}:
        return None
    if isinstance(value, list):
        text = ", ".join(str(item) for item in value)
    elif isinstance(value, dict):
        text = ", ".join(f"{k}={v}" for k, v in value.items() if v not in (None, "", [], {}))
    else:
        text = str(value)
    if not text.strip():
        return None
    return f"{label}: {text}"


def format_city_static(
    *,
    city_slug: str,
    city_name: str,
    city_meta: Mapping[str, Any],
    price_line: str | None = None,
) -> str:
    """Собирает статику города без списка филиалов и сырых полей цены.

    Args:
        city_slug: слаг города.
        city_name: читаемое название.
        city_meta: мета города из справочника.
        price_line: готовая фраза о цене, если уже известна.

    Returns:
        Текстовый блок статики города.
    """
    vehicles = city_meta.get("vehicles") or {}
    categories = city_meta.get("categories") or []
    lines = [
        "Статика разговора (не меняется до конца звонка):",
        f"Город: {city_name} (слаг {city_slug}).",
    ]
    if categories:
        cat_parts = []
        for item in categories:
            code = item.get("code") or "?"
            duration = item.get("duration") or ""
            freq = item.get("start_frequency") or ""
            piece = f"{code}"
            if duration:
                piece += f" — {duration}"
            if freq:
                piece += f", набор: {freq}"
            cat_parts.append(piece)
        lines.append("Категории: " + "; ".join(cat_parts) + ".")
    manual = vehicles.get("manual") or []
    auto = vehicles.get("automatic") or []
    if manual or auto:
        fleet = []
        if manual:
            fleet.append("механика: " + ", ".join(str(x) for x in manual))
        if auto:
            fleet.append("автомат: " + ", ".join(str(x) for x in auto))
        if vehicles.get("fleet_age"):
            fleet.append(f"возраст парка: {vehicles['fleet_age']}")
        lines.append("Автопарк: " + "; ".join(fleet) + ".")
    for label, key in (
        ("Форматы теории", "theory_formats"),
        ("Документы", "documents"),
        ("Мессенджеры", "messengers"),
    ):
        row = _line(label, city_meta.get(key))
        if row:
            lines.append(row + ".")
    payment = city_meta.get("payment") or {}
    if payment:
        row = _line("Оплата", payment)
        if row:
            lines.append(row + ".")
    if city_meta.get("call_hours"):
        lines.append(f"Часы колл-центра: {city_meta['call_hours']}.")
    contacts = city_meta.get("contacts") or city_meta.get("phones") or {}
    if contacts:
        row = _line("Контакты", contacts)
        if row:
            lines.append(row + ".")
    if price_line:
        lines.append(f"Цена (готовая фраза, произносить только так): {price_line}")
    lines.append("Список филиалов города в контекст не входит.")
    return "\n".join(lines)


def format_branch_static(branch: Mapping[str, Any], *, branch_slug: str) -> str:
    """Собирает статику выбранного филиала.

    Args:
        branch: мета филиала.
        branch_slug: слаг филиала.

    Returns:
        Текстовый блок филиала.
    """
    lines = [f"Выбранный филиал (слаг {branch_slug}):"]
    for label, key in (
        ("Адрес", "address"),
        ("Ориентир", "landmark"),
        ("Тип", "place_type"),
        ("Статус", "status"),
        ("Часы", "working_hours"),
    ):
        value = branch.get(key)
        if value:
            lines.append(f"{label}: {value}.")
    return "\n".join(lines)


def merge_static(
    context: ConversationContext,
    *,
    city_slug: str | None = None,
    city_name: str | None = None,
    city_meta: Mapping[str, Any] | None = None,
    price_line: str | None = None,
    branch_slug: str | None = None,
    branch_meta: Mapping[str, Any] | None = None,
) -> ConversationContext:
    """Дописывает статику один раз: город, затем филиал.

    Args:
        context: текущий контекст.
        city_slug: слаг города.
        city_name: читаемое название.
        city_meta: мета города.
        price_line: готовая фраза о цене.
        branch_slug: слаг филиала.
        branch_meta: мета филиала.

    Returns:
        Обновлённый контекст; если статика уже заморожена целиком — без изменений.
    """
    updated = context.model_copy(deep=True)
    if city_slug and city_name and city_meta is not None and not updated.city_slug:
        updated.city_slug = city_slug
        updated.city_name = city_name
        updated.static_text = format_city_static(
            city_slug=city_slug,
            city_name=city_name,
            city_meta=city_meta,
            price_line=price_line,
        )
    if branch_slug and branch_meta is not None and updated.city_slug and not updated.branch_slug:
        updated.branch_slug = branch_slug
        branch_block = format_branch_static(branch_meta, branch_slug=branch_slug)
        updated.static_text = (updated.static_text + "\n\n" + branch_block).strip()
        updated.frozen = True
    return updated


class ContextState(BaseModel):
    """Сериализуемое представление контекста в состоянии треда."""

    static_text: str = ""
    dynamic_text: str = ""
    city_slug: str | None = None
    city_name: str | None = None
    branch_slug: str | None = None
    frozen: bool = False

    def to_context(self) -> ConversationContext:
        """Преобразует в рабочий объект контекста."""
        return ConversationContext.model_validate(self.model_dump())


def context_from_state(data: Mapping[str, Any] | None) -> ConversationContext:
    """Достаёт контекст из состояния треда.

    Args:
        data: словарь ``conversation_context`` или None.

    Returns:
        Контекст разговора.
    """
    if not data:
        return ConversationContext()
    return ConversationContext.model_validate(dict(data))
