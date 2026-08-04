"""Агент конца разговора: попрощался ли собеседник.

Быстрая модель, одно булево поле. Решение о завершении — не обязанность
генератора: ошибка или таймаут → ``None``, флаг в кеше не трогаем,
лайв-канал не роняем. Признак ставится только на прозвучавшее прощание;
отказ от предложения концом разговора не считается.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from utils.llm_gen import astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)

#: Сколько последних сообщений отдаём агенту как хвост диалога.
_HISTORY_TAIL = 8

#: Системное сообщение агента конца разговора.
#:
#: Прощание засчитывается от любой стороны, но только в паре с ответом или
#: молчанием второй: одно прощание бота само по себе — не конец. Благодарность
#: намеренно убрана из исключений — блокировала «спасибо, до свидания». Отказ
#: от предложения остаётся в исключениях: на нём уже обжигались. Конкретные
#: реплики-образцы в текст не добавлять — модель зачитывает их дословно в
#: собственный ответ.
FAREWELL_SYSTEM = (
    "Ты решаешь, закончен ли разговор.\n"
    "Разговор закончен, когда прощание прозвучало и добавить больше нечего: "
    "попрощался собеседник, либо попрощался бот, а собеседник ответил тем же "
    "или молчит.\n"
    "Одно прощание бота само по себе — не конец: он мог пожелать хорошего дня "
    "посреди разговора.\n"
    "Не конец разговора: договорённость о встрече, запись, обещание прислать "
    "документы, вопрос собеседника, отказ от предложения и любой короткий "
    "ответ, после которого разговор может продолжиться.\n"
    "Сомневаешься — не конец."
)


class FarewellDecision(BaseModel):
    """Решение: закончен ли разговор."""

    conversation_ended: bool = Field(
        default=False,
        description="True, только если прощание прозвучало и разговор исчерпан.",
    )


class FarewellAgent(Protocol):
    """Контракт агента конца разговора."""

    async def decide(
        self,
        reply: str,
        history: Sequence[BaseMessage] = (),
    ) -> FarewellDecision:
        """Решает по реплике и хвосту, закончен ли разговор."""


def _format_history(history: Sequence[BaseMessage], *, limit: int = _HISTORY_TAIL) -> str:
    """Хвост диалога текстом для промпта агента."""
    if not history:
        return "— пусто"
    tail = list(history)[-max(1, limit) :]
    lines: list[str] = []
    for message in tail:
        role = "бот" if message.type == "ai" else "клиент"
        text = str(getattr(message, "content", "") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines) if lines else "— пусто"


class LlmFarewellAgent:
    """Агент конца разговора на быстрой модели."""

    async def decide(
        self,
        reply: str,
        history: Sequence[BaseMessage] = (),
    ) -> FarewellDecision:
        """Один вызов быстрой модели; исключение — наверх вызывающему."""
        reply_text = (reply or "").strip()
        if not reply_text and not history:
            return FarewellDecision(conversation_ended=False)

        history_block = _format_history(history)
        current = "реплики человека не было — он молчит" if not reply_text else reply
        human = f"Хвост диалога:\n{history_block}\nТекущая реплика собеседника: {current}"
        schema = response_format_from(FarewellDecision, name="vector_farewell")
        async with get_llm(fast=True, temperature=0.0) as llm:
            raw = await astream_structured(
                llm,
                [SystemMessage(content=FAREWELL_SYSTEM), HumanMessage(content=human)],
                schema=schema,
                text_field=None,
                purpose="прощание",
            )
        return FarewellDecision.model_validate(raw)


async def decide_farewell(
    reply: str,
    *,
    history: Sequence[BaseMessage] = (),
    agent: FarewellAgent | None = None,
) -> FarewellDecision | None:
    """Точка входа агента конца разговора.

    Args:
        reply: реплика собеседника.
        history: хвост истории разговора.
        agent: подмена для офлайн-тестов.

    Returns:
        Решение с одним булевым полем. При ошибке агента — ``None``:
        флаг в кеше не трогаем, исключение наружу не летит.
    """
    worker = agent or LlmFarewellAgent()
    try:
        return await worker.decide(reply, history)
    except Exception as exc:  # noqa: BLE001
        log.warning("Агент конца разговора упал: %s", exc)
        return None
