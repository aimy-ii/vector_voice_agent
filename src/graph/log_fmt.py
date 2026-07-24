"""Чистые форматтеры служебных логов хода.

Сами ``stage`` / ``log.info`` здесь не вызываются — только строки,
которые узлы отдают в ``graph.progress.stage``. Так логи покрываются
тестами без подмены логгера.
"""

from __future__ import annotations

from typing import Mapping, Sequence

#: Сколько символов произнесённого показывать в логе commit.
SPOKEN_PREVIEW_LEN = 60


def format_lookup_done(turn_calls: Sequence[Mapping[str, object]]) -> str:
    """Итог lookup за ход: сколько вызовов и какие.

    Args:
        turn_calls: записи журнала только этого хода (не накопительного).

    Returns:
        Строка вида ``2 вызова: list_cities, resolve_city``.
    """
    if not turn_calls:
        return "обращений к справочнику нет"
    names: list[str] = []
    for entry in turn_calls:
        call = entry.get("call")
        if call:
            names.append(str(call))
    label = "вызов" if len(names) == 1 else "вызова" if 2 <= len(names) <= 4 else "вызовов"
    return f"{len(names)} {label}: {', '.join(names)}"


def format_check_done(closures: Sequence[tuple[str, str]]) -> str:
    """Итог чекера: что закрылось и на каком основании.

    Args:
        closures: пары ``(step_id, основание)``; основание —
            ``диалог`` / ``счётчик`` / ``доставка``.

    Returns:
        Строка для ``[check|done]``.
    """
    if not closures:
        return "ничего не закрылось"
    parts = [f"закрыт {step_id} ({reason})" for step_id, reason in closures]
    return ", ".join(parts)


def format_plan_done(
    *,
    step_id: str | None,
    route: str,
    head: Sequence[tuple[str, int]],
    city_slug: str | None,
    branch_slug: str | None,
) -> str:
    """Итог plan: шаг, маршрут, счётчики шапки, фиксации.

    Args:
        step_id: ведущий шаг или None.
        route: выбранный маршрут.
        head: пары ``(step_id, счётчик_терпения)`` в порядке шапки.
        city_slug: зафиксированный город.
        branch_slug: зафиксированный филиал.

    Returns:
        Строка для ``[plan|done]``.
    """
    if head:
        head_bits = ", ".join(f"{sid}({count})" for sid, count in head)
        head_text = f"[{head_bits}]"
    else:
        head_text = "[]"
    city = city_slug or "—"
    branch = branch_slug or "—"
    return (
        f"шаг {step_id or '—'}, маршрут {route}, шапка {head_text}, город={city}, филиал={branch}"
    )


def format_spoken_preview(text: str, *, limit: int = SPOKEN_PREVIEW_LEN) -> str:
    """Обрезает произнесённое для лога commit.

    Args:
        text: полный текст реплики.
        limit: максимум символов в превью.

    Returns:
        Превью без кавычек; длиннее ``limit`` — с многоточием.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"
