"""Возражения клиента и доводы к ним: данные, а не код.

В скрипте продаж двадцать два шага и ни одного под отработку возражений.
На прогонах бот отбивал «дорого» и «почему предоплата» из общего промпта:
получалось складно, но неуправляемо — заказчик не мог задать, чем именно
отвечать, и от звонка к звонку доводы менялись.

Перечень лежит рядом со скриптом, в ``objections_ru.json``, и правится без
кода. Подбор здесь детерминированный: совпадение по словам-приметам, без
модели. Модель получает уже выбранные доводы и говорит их своими словами —
решение о том, что сказать, принимает файл, а не она.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Файл с перечнем возражений рядом с данными скрипта.
DEFAULT_FILE = Path(__file__).resolve().parent / "data" / "objections_ru.json"

#: Сколько слов-примет должно совпасть, чтобы считать возражение узнанным.
#:
#: Одной приметы достаточно: они подобраны так, что каждое слово само по
#: себе указывает на возражение — «дороговато», «предоплата», «посоветоваться».
MIN_HITS = 1

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Objection:
    """Одно возражение и доводы к нему.

    Attributes:
        id: слаг возражения.
        name: человеческое название для журнала.
        triggers: слова и обороты, по которым возражение узнаётся.
        arguments: доводы в порядке важности.
        ask: чем закончить реплику, чтобы разговор шёл дальше.
    """

    id: str
    name: str
    triggers: tuple[str, ...] = ()
    arguments: tuple[str, ...] = ()
    ask: str = ""
    hits: tuple[str, ...] = field(default=(), compare=False)


def _normalize(text: str) -> str:
    """Приводит текст к виду для сравнения: нижний регистр, «ё» → «е»."""
    return " ".join(_WORD.findall((text or "").lower().replace("ё", "е")))


def load_objections(path: str | Path | None = None) -> tuple[Objection, ...]:
    """Читает перечень возражений из файла.

    Args:
        path: путь к файлу; по умолчанию ``DEFAULT_FILE``.

    Returns:
        Возражения в порядке файла; пустой кортеж, если файла нет или он
        пуст. Отсутствие файла не ошибка: без него бот работает как
        работал, из общего промпта.
    """
    source = Path(path or DEFAULT_FILE)
    if not source.exists():
        return ()
    raw: Mapping[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    items: Sequence[Mapping[str, Any]] = raw.get("objections") or ()
    out: list[Objection] = []
    for item in items:
        slug = str(item.get("id") or "").strip()
        if not slug:
            continue
        out.append(
            Objection(
                id=slug,
                name=str(item.get("name") or slug).strip(),
                triggers=tuple(
                    str(t).strip() for t in item.get("triggers") or () if str(t).strip()
                ),
                arguments=tuple(
                    str(a).strip() for a in item.get("arguments") or () if str(a).strip()
                ),
                ask=str(item.get("ask") or "").strip(),
            )
        )
    return tuple(out)


def match_objection(reply: str, objections: Sequence[Objection]) -> Objection | None:
    """Подбирает возражение по реплике клиента.

    Считает, сколько слов-примет возражения встретилось в реплике.
    Побеждает возражение с наибольшим числом совпадений; при равенстве —
    то, что стоит раньше в файле: порядок задаёт заказчик.

    Args:
        reply: реплика клиента.
        objections: перечень возражений.

    Returns:
        Возражение с заполненным ``hits`` или ``None``, если ничего не
        совпало.
    """
    text = _normalize(reply)
    if not text:
        return None
    best: Objection | None = None
    best_hits = 0
    for item in objections:
        hits = tuple(t for t in item.triggers if _normalize(t) and _normalize(t) in text)
        if len(hits) > best_hits and len(hits) >= MIN_HITS:
            best = Objection(
                id=item.id,
                name=item.name,
                triggers=item.triggers,
                arguments=item.arguments,
                ask=item.ask,
                hits=hits,
            )
            best_hits = len(hits)
    return best


def format_objection(objection: Objection) -> str:
    """Собирает блок для контекста: чем отвечать на возражение.

    Не готовая реплика, а доводы и вопрос: говорит их модель своими
    словами, как и всё остальное в разговоре.

    Args:
        objection: подобранное возражение.

    Returns:
        Текст блока.
    """
    lines = [f"Возражение «{objection.name}». Отвечать этими доводами, по порядку:"]
    lines.extend(f"- {argument}" for argument in objection.arguments)
    if objection.ask:
        lines.append(f"Закончить обращением к человеку: {objection.ask}")
    lines.append("Доводов сверх перечисленных не придумывать.")
    return "\n".join(lines)
