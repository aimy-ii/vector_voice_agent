"""Приведение значений анкеты к виду записи, а не к виду реплики.

Агент профиля возвращает значение словами человека, и в разговоре это
верно: «механика», «вечером после семи» — так и надо записать. Но человек
отвечает предложением, а не заполняет поле, и в анкету попадает «В
Санкт-Петербурге», «у метро Проспект Просвещения», «Ну, механика.». Для
чтения это терпимо, для выгрузки в работу — нет: одно и то же место
записывается по-разному, и поля не сравнить между звонками.

Приведение здесь детерминированное, без модели: пробелы, крайние знаки,
заглавная буква и ведущие предлоги у полей о месте. Падежи не трогаем —
без морфологии это гадание; город берётся из справочника отдельно, там он
уже в именительном.
"""

from __future__ import annotations

from graph.nearby import PLACE_PREFIXES

#: Ведущие предлоги для полей анкеты.
#:
#: Перечень геокодера плюс голое «в». В запросе к геокодеру его нет
#: намеренно — там список подбирался под поиск, — а в записи «В
#: Санкт-Петербурге» предлог лишний.
_PROFILE_PREFIXES: tuple[str, ...] = (*PLACE_PREFIXES, "в")

#: Поля, где ведущий предлог — часть речи, а не часть значения.
#:
#: «у метро Проспект Просвещения» и «метро Проспект Просвещения» — одно
#: место. У остальных полей предлог осмыслен: «в классе» это ответ про
#: формат теории, и «Классе» вместо него было бы порчей.
PLACE_KEYS = frozenset({"location_hint", "branch", "city"})

#: Вводные слова, с которых человек начинает ответ вслух.
#:
#: «Ну, механика», «Это Приморский район», «Да, механика» — само значение
#: начинается после них.
_OPENERS: tuple[str, ...] = ("ну", "это", "да", "а", "вот", "так")

#: Знаки, которые в поле анкеты не несут смысла.
_TRAILING = " .,;:!"


def _strip_openers(text: str) -> str:
    """Снимает вводные слова в начале.

    Args:
        text: значение поля.

    Returns:
        Значение без ведущих вводных слов.
    """
    body = text
    changed = True
    while changed and body:
        changed = False
        head = body.split(maxsplit=1)
        if not head:
            break
        first = head[0].strip(_TRAILING).lower()
        if first in _OPENERS:
            body = head[1].strip() if len(head) > 1 else ""
            changed = True
    return body


def _strip_place_prefix(text: str) -> str:
    """Снимает ведущий предлог места, сохраняя регистр.

    ``normalize_place`` делает то же, но приводит строку к нижнему
    регистру: она собирает ключ поиска, а не значение для записи.

    Args:
        text: значение поля о месте.

    Returns:
        Значение без ведущего предлога.
    """
    body = text
    changed = True
    while changed and body:
        changed = False
        lowered = body.lower()
        for prefix in _PROFILE_PREFIXES:
            head = f"{prefix} "
            if lowered.startswith(head):
                body = body[len(head) :].strip()
                changed = True
                break
            if lowered == prefix:
                return ""
    return body


def tidy_value(key: str, value: str) -> str:
    """Приводит значение поля к виду записи.

    Args:
        key: ключ поля анкеты.
        value: значение так, как его вернул агент профиля.

    Returns:
        Приведённое значение; пустая строка, если после чистки ничего не
        осталось.

    Examples:
        ``tidy_value("city", "В Санкт-Петербурге")`` → «Санкт-Петербурге»;
        ``tidy_value("location_hint", "у метро Проспект Просвещения")`` →
        «Метро Проспект Просвещения»;
        ``tidy_value("transmission", "Ну, механика.")`` → «Механика».
    """
    body = " ".join((value or "").split()).strip(_TRAILING)
    if not body:
        return ""
    body = _strip_openers(body).strip(_TRAILING)
    if key in PLACE_KEYS:
        body = _strip_place_prefix(body).strip(_TRAILING)
    if not body:
        return ""
    return body[0].upper() + body[1:]
