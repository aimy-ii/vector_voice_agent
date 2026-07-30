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
    call_id: str | None = None,
) -> str:
    """Итог plan: шаг, маршрут, счётчики шапки, звонок, фиксации.

    Args:
        step_id: ведущий шаг или None.
        route: выбранный маршрут.
        head: пары ``(step_id, счётчик_попыток)`` в порядке шапки.
        city_slug: зафиксированный город.
        branch_slug: зафиксированный филиал.
        call_id: идентификатор звонка для ключа прогресса.

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
    call = call_id or "—"
    return (
        f"шаг {step_id or '—'}, маршрут {route}, шапка {head_text}, "
        f"звонок {call}, город={city}, филиал={branch}"
    )


def format_live_check_state(
    *,
    attempts: Mapping[str, int],
    status: Mapping[str, str],
    profile: Mapping[str, str],
) -> str:
    """Снимок прогресса, который увидел лайв-чекер после загрузки.

    Args:
        attempts: счётчики попыток шагов.
        status: статусы шагов.
        profile: профиль; в лог — только имена заполненных полей.

    Returns:
        Строка для ``[live-check|state]``.
    """
    filled = sorted(key for key, value in profile.items() if value and str(value).strip())
    profile_text = ", ".join(filled) if filled else "—"
    return f"счётчики {dict(attempts)}, статусы {dict(status)}, профиль: {profile_text}"


def format_check_pending(
    *,
    pending: Sequence[tuple[str, int]],
    rejected: Sequence[tuple[str, str]],
    available: Sequence[tuple[str, int]] | None = None,
) -> str:
    """Состав висящих перед вызовом модели и причины отсева.

    Args:
        pending: пары ``(step_id, счётчик)`` на проверку.
        rejected: пары ``(step_id, причина)`` — почему шаг не попал.
        available: при пустом ``pending`` — доступные шаги со счётчиками.

    Returns:
        Строка для ``[check|pending]``.
    """
    if pending:
        pending_bits = ", ".join(f"{sid}({count})" for sid, count in pending)
        pending_text = f"[{pending_bits}]"
    else:
        pending_text = "пусто"
    if rejected:
        rejected_text = ", ".join(f"{sid} — {reason}" for sid, reason in rejected)
    else:
        rejected_text = "—"
    text = f"на проверку: {pending_text}; отсеяно: {rejected_text}"
    if not pending and available is not None:
        if available:
            avail_bits = ", ".join(f"{sid}({count})" for sid, count in available)
            text += f"; доступны: [{avail_bits}]"
        else:
            text += "; доступны: []"
    return text


def format_contexter_done(
    *,
    tool: str | None,
    subject: str,
    status: str,
    elapsed_ms: int,
    needed: bool = True,
    branch_slugs_count: int | None = None,
) -> str:
    """Итог контекстера: решение агента и статус динамики.

    Args:
        tool: имя выбранного инструмента или None.
        subject: предмет для заглушки.
        status: итоговый ``dynamic_status``.
        elapsed_ms: длительность работы в миллисекундах.
        needed: False — агент решил, что контекст не нужен.
        branch_slugs_count: число отобранных слагов при инструменте ``branches``.

    Returns:
        Строка для ``[contexter|done]``.
    """
    if not needed:
        return "решение: контекст не нужен"
    tool_text = tool or "—"
    subject_text = (subject or "").strip()
    slugs_part = ""
    if branch_slugs_count is not None:
        slugs_part = f", слагов {branch_slugs_count}"
    return (
        f"решение: инструмент {tool_text}, предмет «{subject_text}»{slugs_part}, "
        f"{elapsed_ms} мс; статус {status}"
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


def format_reply_integrity(*, streamed: str, final: str) -> str | None:
    """Сверяет отданное в поток с финальным текстом модели.

    Args:
        streamed: склеенные дельты этого хода, ушедшие в эфир.
        final: поле ``reply`` из ответа модели.

    Returns:
        Строка для лога, если тексты расходятся; иначе ``None``.
    """
    if streamed == final:
        return None
    tail = 40
    stream_tail = streamed[-tail:] if streamed else ""
    final_tail = final[-tail:] if final else ""
    return (
        f"расхождение реплики: поток {len(streamed)} симв. «…{stream_tail}», "
        f"финал {len(final)} симв. «…{final_tail}»"
    )
