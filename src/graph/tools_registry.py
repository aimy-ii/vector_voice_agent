"""Реестр инструментов контекстера.

Новый источник (вектор, карты, полнотекст) = класс по ``ContextTool``
и одна строка в ``build_context_tools``. Контекстер реестр только
перебирает — сам про источники не знает.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from graph.context import ConversationContext
from graph.history import find_aside, matches_triggers
from script.build import CompiledScript
from script.models import Help

#: Признак: инструмент подошёл, но без города клиента ответить нельзя.
NEED_CITY_SIGNAL = "\0need_city"


class ContextTool(Protocol):
    """Инструмент контекстера.

    Единый интерфейс для всех источников — справки, FAQ города, в будущем
    вектор, полнотекст, карты.
    """

    name: str

    async def try_answer(self, reply: str, context: ConversationContext) -> str | None:
        """Ответ инструмента на реплику, либо None, если не по адресу.

        None — инструмент не подходит под этот вопрос, пробуем следующий.
        Пустая строка — подошёл, но ответа нет (статус «не нашлось»).
        ``NEED_CITY_SIGNAL`` — подошёл, но нужен город клиента.
        """
        ...


class HelpsTool:
    """Справки из скрипта — приоритетнее FAQ города."""

    name = "helps"

    def __init__(self, helps: Mapping[str, Help] | CompiledScript) -> None:
        """Принимает скрипт или готовый словарь справок.

        Args:
            helps: скомпилированный скрипт либо ``id → Help``.
        """
        if isinstance(helps, CompiledScript):
            self._helps: Mapping[str, Help] = helps.helps
        else:
            self._helps = helps

    async def try_answer(self, reply: str, context: ConversationContext) -> str | None:
        """Ищет справку по триггерам; ``None`` — реплика не про справки."""
        if not self._helps:
            return None
        catalogue: dict[str, Sequence[str]] = {
            item_id: item.triggers for item_id, item in self._helps.items()
        }
        help_id = find_aside(reply, catalogue)
        if help_id is None:
            return None
        item = self._helps.get(help_id)
        return (item.text if item else "").strip()


class FaqTool:
    """FAQ из меты города — резерв после справок скрипта."""

    name = "faq"

    async def try_answer(self, reply: str, context: ConversationContext) -> str | None:
        """Ищет ответ в FAQ города.

        Без города — ``NEED_CITY_SIGNAL``, если реплика похожа на вопрос
        из типичного FAQ (есть «?» или вопросительные маркеры). Пустой
        FAQ при известном городе — ``None``. Совпадение без текста ответа —
        пустая строка.
        """
        text = (reply or "").strip()
        if not text:
            return None

        if not context.city_slug:
            if self._looks_like_faq_question(text):
                return NEED_CITY_SIGNAL
            return None

        faq = list(context.city_faq or [])
        if not faq:
            return None

        catalogue: dict[str, Sequence[str]] = {}
        answers: dict[str, str] = {}
        for index, item in enumerate(faq):
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            item_id = f"faq_{index}"
            # Признаки — слова вопроса длиннее трёх букв.
            triggers = [w for w in question.lower().split() if len(w) >= 4]
            if not triggers:
                triggers = [question.lower()]
            catalogue[item_id] = triggers
            answers[item_id] = str(item.get("answer") or "").strip()

        if not catalogue:
            return None

        # Сначала точное вхождение вопроса / триггеров через find_aside.
        matched = find_aside(text, catalogue)
        if matched is None:
            # Запас: реплика содержит существенную часть вопроса.
            matched = self._match_by_overlap(text, faq)
            if matched is None:
                return None
            return matched

        return answers.get(matched, "")

    @staticmethod
    def _looks_like_faq_question(reply: str) -> bool:
        """Грубая проверка: реплика похожа на справочный вопрос без города."""
        lowered = reply.lower()
        if "?" in reply:
            return True
        markers = (
            "сколько",
            "как ",
            "где ",
            "какой",
            "какая",
            "какие",
            "что ",
            "есть ли",
            "нужн",
            "документ",
            "рассроч",
            "стоим",
            "цена",
            "оплат",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _match_by_overlap(reply: str, faq: Sequence[Mapping[str, str]]) -> str | None:
        """Совпадение по пересечению токенов с вопросом FAQ.

        Returns:
            Текст ответа, пустая строка (вопрос похож, ответа нет) или None.
        """
        reply_tokens = {t for t in reply.lower().replace("?", " ").split() if len(t) >= 4}
        if not reply_tokens:
            return None
        best_id: int | None = None
        best_score = 0
        for index, item in enumerate(faq):
            question = str(item.get("question") or "").lower()
            q_tokens = {t for t in question.replace("?", " ").split() if len(t) >= 4}
            score = len(reply_tokens & q_tokens)
            if score > best_score and score >= 2:
                best_score = score
                best_id = index
        if best_id is None:
            # Один сильный триггер из вопроса целиком в реплике.
            for item in faq:
                question = str(item.get("question") or "").strip()
                if question and matches_triggers(reply, [question]):
                    return str(item.get("answer") or "").strip()
            return None
        return str(faq[best_id].get("answer") or "").strip()


def build_context_tools(script: CompiledScript) -> list[ContextTool]:
    """Реестр инструментов контекстера по приоритету.

    Добавить новый инструмент = дописать его в этот список. Контекстер
    перебирает реестр по порядку, первый подходящий отвечает.
    """
    return [
        HelpsTool(script),  # справки скрипта — главнее
        FaqTool(),  # FAQ города — резерв
    ]
