"""Реестр инструментов контекстера.

Новый источник (вектор, карты, полнотекст) = класс по ``ContextTool``
и одна строка в ``build_context_tools``. Контекстер реестр только
перебирает — сам про источники не знает.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from graph.context import ConversationContext
from graph.history import find_aside
from script.build import CompiledScript
from script.models import Help


class ContextTool(Protocol):
    """Инструмент контекстера.

    Единый интерфейс для всех источников — справки, в будущем вектор,
    полнотекст, карты.
    """

    name: str

    async def try_answer(self, reply: str, context: ConversationContext) -> str | None:
        """Ответ инструмента на реплику, либо None, если не по адресу.

        None — инструмент не подходит под этот вопрос, пробуем следующий.
        Пустая строка — подошёл, но ответа нет (статус «не нашлось»).
        """
        ...


class HelpsTool:
    """Справки из скрипта — единственный инструмент сегодня."""

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


def build_context_tools(script: CompiledScript) -> list[ContextTool]:
    """Реестр инструментов контекстера по приоритету.

    Добавить новый инструмент = дописать его в этот список. Контекстер
    перебирает реестр по порядку, первый подходящий отвечает.
    """
    return [
        HelpsTool(script),  # справки из скрипта — единственный на сегодня
        # сюда добавляются будущие: VectorTool(...), MapsTool(...), ...
    ]
