"""Писатель в поток: что уходит в трубку, а что только в логи.

Бот подключён плагином в режиме `stream_mode="custom"`, строкой,
`subgraphs=False`. Плагин озвучивает то, что писатель отдал строкой либо
словарём с ключом `content`; словарь без `content` он молча отбрасывает.

Отсюда два вида полезной нагрузки:

* `say("текст")` — звучит в трубке;
* `stage(...)` — словарь `{"stage", "phase", "text"}` без `content`:
  виден в логах и в Studio, в трубку не попадает.

Служебный прогресс поэтому летит тем же потоком и не мешает разговору —
отдельного канала заводить не нужно.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _writer() -> Any | None:
    """Достаёт писатель потока или None, если код исполняется вне графа."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # noqa: BLE001
        return None


def say(text: str) -> None:
    """Отдаёт кусок реплики в эфир.

    Строка озвучивается плагином как есть. Пустые куски не шлём: они не несут
    звука, но добавляют событий в поток.

    Args:
        text: кусок текста реплики.
    """
    if not text:
        return
    writer = _writer()
    if writer is None:
        return
    try:
        writer(text)
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось отдать текст в поток: %s", exc)


def stage(name: str, text: str, phase: str = "done", **extra: Any) -> None:
    """Публикует служебное событие: в поток и в лог, но не в трубку.

    Args:
        name: этап («lookup», «respond», «commit»).
        text: что произошло, человеческим языком.
        phase: `start` — начали, `done` — закончили.
        **extra: дополнительные поля события для логов и Studio.
    """
    log.info("[%s|%s] %s", name, phase, text)
    writer = _writer()
    if writer is None:
        return
    payload: dict[str, Any] = {"stage": name, "phase": phase, "text": text}
    payload.update(extra)
    try:
        # Ключа `content` здесь нет намеренно — плагин вернёт None и не озвучит.
        writer(payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось отдать событие в поток: %s", exc)
