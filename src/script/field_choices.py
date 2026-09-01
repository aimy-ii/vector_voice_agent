"""Поля анкеты с заранее заданной формой: варианты как данные, а не код.

Агент профиля возвращает значение словами человека, и для большинства
полей это верно. Но у некоторых полей форма задана скриптом: формат
теории — это «очно», «дистанционно» или «комбинированно», а не пересказ
реплики. На прогонах в поле ложилось «Дома», «Очно дома», «В теории
лучше очно дома отвлекается» — читать можно, работать нельзя.

Варианты названы в требованиях шагов прозой, отдельного поля под них в
``SalesStep`` нет и не будет: технических полей в скрипте нет намеренно.
Поэтому перечень правится без кода: он лежит в админке справочника, а файл
``field_choices_ru.json`` рядом со скриптом остаётся запасным вариантом —
как и у перечня возражений.

Подбор здесь детерминированный, без модели: совпадение по словам-приметам.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from script.documents import load_document

#: Файл с вариантами полей рядом с данными скрипта.
DEFAULT_FILE = Path(__file__).resolve().parent / "data" / "field_choices_ru.json"


@dataclass(frozen=True)
class FieldChoice:
    """Один вариант значения поля.

    Attributes:
        value: как значение должно выглядеть в анкете.
        triggers: слова и обороты, по которым вариант узнаётся.
    """

    value: str
    triggers: tuple[str, ...] = ()


def _normalize(text: str) -> str:
    """Приводит текст к виду для сравнения: нижний регистр, «ё» → «е»."""
    return (text or "").lower().replace("ё", "е")


def load_field_choices(path: str | Path | None = None) -> dict[str, tuple[FieldChoice, ...]]:
    """Читает варианты полей из файла.

    Args:
        path: путь к файлу; по умолчанию ``DEFAULT_FILE``.

    Returns:
        Варианты по ключу поля в порядке файла; пустой словарь, если файла
        нет. Отсутствие файла не ошибка: без него значение попадает в
        анкету как есть, то есть как было до перечня.
    """
    source = Path(path or DEFAULT_FILE)
    raw = load_document("field_choices", source)
    if not raw:
        return {}
    fields: Mapping[str, Sequence[Mapping[str, Any]]] = raw.get("fields") or {}
    out: dict[str, tuple[FieldChoice, ...]] = {}
    for key, items in fields.items():
        choices = tuple(
            FieldChoice(
                value=str(item.get("value") or "").strip(),
                triggers=tuple(
                    str(t).strip() for t in item.get("triggers") or () if str(t).strip()
                ),
            )
            for item in items or ()
            if str(item.get("value") or "").strip()
        )
        if choices:
            out[str(key).strip()] = choices
    return out


def match_choice(value: str, choices: Sequence[FieldChoice]) -> str:
    """Сводит сказанное к одному из вариантов поля.

    Если реплика подходит под несколько вариантов, побеждает тот, что
    ниже в файле: порядок задаёт заказчик. Так «очно дома» становится
    «Очно» — человек говорит про очные занятия рядом с домом, — а «дома»
    без «очно» остаётся дистанционным форматом.

    Args:
        value: значение поля так, как его вернул агент профиля.
        choices: варианты этого поля в порядке файла.

    Returns:
        Значение варианта либо пустая строка, если ни один не узнан.
    """
    text = _normalize(value)
    if not text.strip():
        return ""
    found = ""
    for choice in choices:
        if any(_normalize(trigger) in text for trigger in choice.triggers):
            found = choice.value
    return found
