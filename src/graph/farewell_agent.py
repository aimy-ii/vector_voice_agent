"""Агент конца разговора: сказал ли собеседник, что разговор окончен.

Быстрая модель, одно булево поле. Решение о завершении — не обязанность
генератора: ошибка или таймаут → ``None``, флаг в кеше не трогаем,
лайв-канал не роняем.

Конец — только когда собеседник сам словами закончил разговор: попрощался
или сказал, что дальше говорить не будет. Ответы вроде «вопросов нет»
концом не считаются. Оборвать живой звонок дороже лишней реплики,
поэтому при сомнении признак не поднимаем.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from utils.llm_gen import astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)

#: Сколько последних сообщений отдаём агенту как хвост диалога.
#:
#: Прощание парное: модели нужен хвост, чтобы увидеть, ответил ли собеседник
#: прощанием на прощание бота. Двенадцать сообщений закрывают несколько
#: обменов и не меняют стоимость вызова.
_HISTORY_TAIL = 12

#: Системное сообщение агента конца разговора.
#:
#: Конец — факт сказанного, не домысел о состоянии. Два случая: собеседник
#: попрощался или сказал, что дальше говорить не будет. Образцы в кавычках —
#: признаки, не список для дословного сравнения. Формулировки вроде «отвечает
#: односложно», «не подхватывает темы», «всё уже узнал» в текст не входят:
#: они поднимали признак на живом звонке. Сколько тем осталось у бота —
#: не критерий. Оборвать живой разговор дороже, чем пропустить прощание.
FAREWELL_SYSTEM = (
    "Ты решаешь, закончил ли собеседник разговор.\n"
    "\n"
    "Конец разговора — это когда собеседник сам, словами, закончил разговор. "
    "Два случая, других нет:\n"
    "1. Он попрощался: «до свидания», «всего доброго», «до связи», «пока», "
    "«спасибо, всего хорошего».\n"
    "2. Он сказал, что дальше говорить не будет: «мне пора», «неудобно "
    "говорить», «я на работе», «давайте потом», «перезвоните позже», "
    "«хватит», «я уже отвечал».\n"
    "Это признаки, а не список для дословного сравнения: те же мысли другими "
    "словами — тоже конец.\n"
    "\n"
    "Прощание в телефонном разговоре парное. Бот попрощался, а собеседник "
    "ответил прощанием — конец. Бот попрощался, а собеседник молчит или "
    "говорит о другом — не конец: разговор продолжается.\n"
    "\n"
    "Не конец разговора, даже если звучит похоже:\n"
    "— «вопросов нет», «нет вопросов», «всё понятно», «понял», «ясно», "
    "«хорошо», «ага», «спасибо» — это ответы на вопрос бота, а не завершение "
    "разговора;\n"
    "— любой короткий или односложный ответ;\n"
    "— молчание;\n"
    "— отказ от предложения: не нужна вторая категория, не подходит филиал, "
    "не хочу рассрочку;\n"
    "— договорённость о встрече, согласие на запись, обещание прислать "
    "документы;\n"
    "— вопрос собеседника;\n"
    "— «ну ладно», «ну всё», «в общем» без прощания: после этого разговор "
    "часто продолжается.\n"
    "\n"
    "Сколько тем у бота осталось впереди — не твоё дело. Твоё дело одно: "
    "сказал ли собеседник, что разговор окончен.\n"
    "\n"
    "Оборвать живой разговор дороже, чем пропустить прощание. "
    "Сомневаешься — не конец."
)


class FarewellDecision(BaseModel):
    """Решение: закончен ли разговор."""

    conversation_ended: bool = Field(
        default=False,
        description=(
            "True, только когда разговор подошёл к концу: прощание прозвучало "
            "либо разговор исчерпан и собеседник это показывает."
        ),
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
