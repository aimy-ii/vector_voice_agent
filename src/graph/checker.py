"""Чекер шагов: единственная точка, где шаг уходит из скрипта.

На этом этапе вызывается синхронно в начале хода по последней реплике
клиента. Контракт входа не знает источника текста — переезд на куски речи
не потребует переписывания.

Закрывает по двум основаниям: ИИ видит, что шаг закрылся по диалогу; код
видит, что счётчик исчерпан. Модель не ответила — пропускаем и идём дальше.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.config import settings
from graph.history import last_user_text
from script.build import CompiledScript
from script.models import Step
from script.planner import exhausted, iter_available, render_step_text
from script.store import ScriptProgress
from utils.llm_gen import LLMTurnFailed, astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)

#: Потолок ходов в срезе истории для уже заданных шагов.
HISTORY_TURN_CAP = 12


class CheckerVerdict(BaseModel):
    """Ответ модели по одному шагу.

    Идентификатор шага модель не называет — код знает, что передал.
    """

    reply_usable: bool = Field(
        description=(
            "Годится ли реплика клиента для анализа: это законченный ответ, а не обрывок или шум."
        ),
    )
    step_closed: bool = Field(
        description=(
            "Закрывается ли шаг: по диалогу видно, что задача шага решена. "
            "«Потом скажу» — ответ есть, шаг не закрыт."
        ),
    )


class CheckerClient(Protocol):
    """Контракт вызова модели чекера — подменяется в тестах."""

    async def judge(
        self,
        *,
        history_slice: str,
        client_reply: str,
        step: Step,
        step_text: str | None,
    ) -> CheckerVerdict | None:
        """Оценивает один шаг. None — модель не ответила."""


class LlmCheckerClient:
    """Чекер на быстрой модели с короткой схемой."""

    async def judge(
        self,
        *,
        history_slice: str,
        client_reply: str,
        step: Step,
        step_text: str | None,
    ) -> CheckerVerdict | None:
        """Вызывает модель; при сбое возвращает None."""
        system = (
            "Ты проверяешь, закрылся ли шаг скрипта телефонного разговора.\n"
            "Тебе даны ТРИ РАЗДЕЛЬНЫХ блока: срез истории, реплика клиента и шаг.\n"
            "Не склеивай их мысленно как один текст: реплика — отдельный блок.\n"
            "Ответь только по схеме: reply_usable и step_closed.\n"
            "step_closed=true только если задача шага решена. "
            "«Потом скажу», уклонение, шутка — step_closed=false.\n"
            "Идентификатор шага не возвращай."
        )
        human = (
            f"### Срез истории\n{history_slice or '(пусто)'}\n\n"
            f"### Реплика клиента\n{client_reply or '(пусто)'}\n\n"
            f"### Шаг\n"
            f"id (служебно, не возвращай): {step.id}\n"
            f"Задача: {step.goal}\n"
            f"Формулировка: {step_text or step.text or '—'}"
        )
        schema = response_format_from(CheckerVerdict, name="vector_checker")
        try:
            async with get_llm(fast=True, temperature=0.0) as llm:
                raw = await astream_structured(
                    llm,
                    [SystemMessage(content=system), HumanMessage(content=human)],
                    schema=schema,
                    text_field=None,
                )
            return CheckerVerdict.model_validate(raw)
        except LLMTurnFailed as exc:
            log.warning("Чекер: модель не ответила: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("Чекер: сбой разбора ответа: %s", exc)
            return None


def _format_history(messages: Sequence[BaseMessage]) -> str:
    """Печатает историю без склейки с текущей репликой."""
    lines: list[str] = []
    for message in messages:
        role = "клиент" if message.type == "human" else "агент"
        content = message.content if isinstance(message.content, str) else str(message.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def history_slice_for(
    messages: Sequence[BaseMessage],
    *,
    steps: Sequence[Step],
    progress: ScriptProgress,
    turn: int,
) -> list[BaseMessage]:
    """Срез истории под проверяемые шаги.

    От взятия самого старого шага со счётчиком > 0; для шагов со счётчиком
    ноль — с начала звонка. Сверху — потолок по числу сообщений.

    Args:
        messages: полная история хода.
        steps: шаги, которые сейчас проверяем.
        progress: прогресс скрипта.
        turn: номер текущего хода.

    Returns:
        Срез сообщений без последней реплики клиента (она идёт отдельно).
    """
    if not messages:
        return []
    body = list(messages)
    # Последняя реплика клиента уходит отдельным блоком.
    if body and body[-1].type == "human":
        body = body[:-1]

    has_zero = any(int(progress.attempts.get(step.id, 0)) == 0 for step in steps)
    if has_zero:
        start_turn = 0
    else:
        taken_turns = [
            int(progress.taken_turn.get(step.id, turn))
            for step in steps
            if int(progress.attempts.get(step.id, 0)) > 0
        ]
        start_turn = min(taken_turns) if taken_turns else 0

    # Грубая оценка: 2 сообщения на ход (клиент + агент).
    keep_from = max(0, len(body) - HISTORY_TURN_CAP * 2)
    if start_turn > 0:
        # taken_turn — номер хода; оставляем хвост с запасом от начала взятия.
        approx = max(0, (start_turn - 1) * 2)
        keep_from = max(keep_from, min(approx, len(body)))
    return body[keep_from:]


async def run_checker(
    *,
    script: CompiledScript,
    progress: ScriptProgress,
    messages: Sequence[BaseMessage],
    profile: dict[str, str],
    turn: int,
    client: CheckerClient | None = None,
    attempt_limit: int | None = None,
) -> ScriptProgress:
    """Закрывает шаги по счётчику и по диалогу.

    Args:
        script: скомпилированный скрипт.
        progress: текущий прогресс (будет изменён копией).
        messages: история звонка.
        profile: профиль для доступности шагов.
        turn: номер хода.
        client: клиент модели; пусто — боевой.
        attempt_limit: порог попыток; пусто — из настроек.

    Returns:
        Обновлённый прогресс.
    """
    updated = ScriptProgress.from_mapping(progress.to_dict())
    limit = attempt_limit if attempt_limit is not None else settings.step_attempt_limit
    reply = last_user_text(list(messages))

    # 1. Счётчик исчерпан — без модели.
    for step in iter_available(script, status=updated.status, profile=profile):
        if exhausted(step, updated.attempts, limit=limit):
            updated.status[step.id] = "closed"

    pending = [
        step
        for step in iter_available(script, status=updated.status, profile=profile)
        if int(updated.attempts.get(step.id, 0)) > 0
        and not exhausted(step, updated.attempts, limit=limit)
    ]
    if not pending or not reply.strip():
        return updated

    judge = client or LlmCheckerClient()
    history = history_slice_for(messages, steps=pending, progress=updated, turn=turn)
    history_text = _format_history(history)

    for step in pending:
        verdict = await judge.judge(
            history_slice=history_text,
            client_reply=reply,
            step=step,
            step_text=render_step_text(step, profile),
        )
        if verdict is None:
            # Модель не ответила — шаги не трогаем, ход продолжается.
            break
        if not verdict.reply_usable:
            break
        if verdict.step_closed:
            updated.status[step.id] = "closed"
            continue
        break

    return updated


def close_delivered_inform(
    *,
    script: CompiledScript,
    progress: ScriptProgress,
    pending_step: str | None,
    delivered: bool,
) -> ScriptProgress:
    """Закрывает дословный/информирующий шаг после успешной доставки.

    Args:
        script: скомпилированный скрипт.
        progress: прогресс.
        pending_step: шаг прошлого хода.
        delivered: дослушали ли реплику.

    Returns:
        Обновлённый прогресс.
    """
    updated = ScriptProgress.from_mapping(progress.to_dict())
    if not pending_step or not delivered:
        return updated
    step = script.steps.get(pending_step)
    if step is None:
        return updated
    if step.kind in ("inform", "inform_check") and int(updated.attempts.get(step.id, 0)) > 0:
        updated.status[step.id] = "closed"
    return updated
