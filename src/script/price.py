"""Разговор о цене: три ветки, все — по полям ответа справочника.

Ветка выбирается **по данным**, а не по списку городов и не по константам в
коде. Заказчики ведут базу знаний сами; когда там появятся настоящие цены и
`reliable` станет `true`, агент обязан заговорить иначе без единой правки кода.
Третья ветка поэтому реализована сейчас, хотя сегодня не срабатывает ни разу,
и покрыта офлайн-тестом.

Поле `note` из API вслух не произносится: оно написано под передачу менеджеру,
а бот и есть менеджер. Используем его только как признак. Истина — `amount` и
`reliable`, формулировка — наша, из данных скрипта.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from script.models import PriceTexts

#: Какая из трёх веток разговора о цене выбрана.
PriceBranch = Literal["no_amount", "unreliable", "reliable"]


def pick_branch(price: Mapping[str, Any] | None) -> PriceBranch:
    """Выбирает ветку разговора о цене по ответу справочника.

    * суммы нет → цену не называем;
    * сумма есть, не подтверждена → называем как примерную «от»;
    * сумма есть и подтверждена → называем как точную.

    Args:
        price: раздел `price` из меты города или None, если справочник молчит.

    Returns:
        Идентификатор ветки.
    """
    if not price:
        return "no_amount"
    amount = price.get("amount")
    if not isinstance(amount, (int, float)) or amount <= 0:
        return "no_amount"
    return "reliable" if bool(price.get("reliable")) else "unreliable"


def format_amount(amount: int) -> str:
    """Форматирует сумму так, чтобы её было удобно произносить.

    Пробелы-разделители в озвучке мешают, поэтому число отдаётся слитно.

    Args:
        amount: сумма в рублях.

    Returns:
        Строка вида «44900».
    """
    return str(int(amount))


def price_line(price: Mapping[str, Any] | None, texts: PriceTexts) -> str:
    """Собирает готовую фразу о цене.

    Текст берётся из данных скрипта, число — из справочника. В шаблоне
    доступны подстановки `{amount}` и `{package}`.

    Args:
        price: раздел `price` из меты города.
        texts: тексты трёх веток из параметров скрипта.

    Returns:
        Фраза, готовая к произнесению.
    """
    branch = pick_branch(price)
    template = getattr(texts, branch)
    if branch == "no_amount":
        return template

    data = price or {}
    return template.format(
        amount=format_amount(int(data["amount"])),
        package=str(data.get("package") or "").strip(),
    ).strip()


def price_facts(price: Mapping[str, Any] | None, texts: PriceTexts) -> dict[str, Any]:
    """Готовит блок о цене для промпта модели.

    Модель получает уже выбранную ветку и готовую фразу — решать, называть ли
    число, ей не нужно и нельзя.

    Args:
        price: раздел `price` из меты города.
        texts: тексты трёх веток из параметров скрипта.

    Returns:
        Словарь с веткой, фразой и признаком «число называть можно».
    """
    branch = pick_branch(price)
    return {
        "branch": branch,
        "line": price_line(price, texts),
        "may_name_amount": branch != "no_amount",
    }


def price_line_from_kb(price: Mapping[str, Any] | None) -> str:
    """Готовая фраза о цене из ответа справочника (формат продаж).

    Если в ответе уже есть ``phrase`` / ``line`` — берём её. Иначе собираем
    короткую фразу из суммы и признака ``reliable``.

    Args:
        price: раздел ``price`` из меты города.

    Returns:
        Фраза для произнесения.
    """
    if not price:
        return "Точную сумму зафиксируем при оформлении."
    for key in ("phrase", "line", "text"):
        ready = str(price.get(key) or "").strip()
        if ready:
            return ready
    branch = pick_branch(price)
    if branch == "no_amount":
        return "Точную сумму зафиксируем при оформлении."
    amount = format_amount(int(price["amount"]))
    if branch == "reliable":
        return f"Стоимость обучения — {amount} рублей."
    return f"Стоимость обучения — от {amount} рублей."


def price_facts_from_kb(price: Mapping[str, Any] | None) -> dict[str, Any]:
    """Блок о цене для промпта, когда формулировка приходит из базы.

    Args:
        price: раздел ``price`` из меты города.

    Returns:
        Словарь с веткой, фразой и признаком «число называть можно».
    """
    branch = pick_branch(price)
    return {
        "branch": branch,
        "line": price_line_from_kb(price),
        "may_name_amount": branch != "no_amount",
    }
