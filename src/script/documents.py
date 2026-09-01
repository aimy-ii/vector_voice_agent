"""Документы разговора рядом со скриптом: возражения и варианты полей анкеты.

Оба перечня — данные заказчика, и оба переезжают в админку тем же путём,
что скрипт: сначала спрашиваем справочник, при любой беде читаем файл из
образа. Формат ответа совпадает с файлом поле в поле, поэтому разбирающий
код у обоих источников один и тот же.

Читаются перечни один раз при старте процесса — там же, где читались
раньше. Значит, правки из админки подхватываются перезапуском мозга, а не
на лету. Для возражений это не срочно: они меняются редко, а перезапуск
рвёт идущие звонки.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from core.config import settings

log = logging.getLogger(__name__)

#: Ручки справочника по видам документов.
PATHS: dict[str, str] = {
    "objections": "/api/objections/{doc_id}",
    "field_choices": "/api/field_choices/{doc_id}",
}


def _from_http(kind: str, doc_id: str) -> dict[str, Any] | None:
    """Спрашивает документ у админки справочника.

    Args:
        kind: вид документа: ``objections`` или ``field_choices``.
        doc_id: идентификатор документа, обычно ``vector_ru``.

    Returns:
        Разобранный документ или None, если админка не ответила или
        ответила не тем. Исключений наружу нет: перечень — не повод
        ронять запуск.
    """
    url = settings.vector_kb_url.rstrip("/") + PATHS[kind].format(doc_id=doc_id)
    try:
        response = httpx.get(url, timeout=settings.script_http_timeout)
    except httpx.HTTPError as exc:
        log.warning("Админка недоступна (%s), %s берём из файла", exc, kind)
        return None

    if response.status_code != 200:
        log.warning(
            "Админка ответила HTTP %s на %s, берём из файла",
            response.status_code,
            kind,
        )
        return None

    try:
        payload = response.json()
    except ValueError as exc:
        log.warning("Админка отдала не JSON (%s), %s берём из файла", exc, kind)
        return None
    if not isinstance(payload, dict):
        log.warning("Админка отдала не объект, %s берём из файла", kind)
        return None
    return payload


def _from_file(path: Path) -> dict[str, Any] | None:
    """Читает документ из файла образа.

    Args:
        path: путь к JSON-файлу.

    Returns:
        Разобранный документ или None, если файла нет или он битый.
        Отсутствие файла не ошибка: без перечня бот работает как работал.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Файл %s не читается: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def load_document(kind: str, path: Path, doc_id: str | None = None) -> dict[str, Any] | None:
    """Читает документ из источника, заданного настройкой ``SCRIPT_SOURCE``.

    Args:
        kind: вид документа: ``objections`` или ``field_choices``.
        path: файл документа в образе — он же запасной вариант.
        doc_id: идентификатор документа; None — из настроек.

    Returns:
        Разобранный документ или None, если его нет ни там, ни там.
    """
    if settings.script_source.strip().lower() == "http":
        payload = _from_http(kind, doc_id or settings.script_id)
        if payload is not None:
            log.info("%s взяты из админки справочника", kind)
            return payload
    return _from_file(path)
