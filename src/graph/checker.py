"""Чекер шагов: единственная точка, где шаг уходит из скрипта.

Контракт входа не знает источника текста: полная реплика основного хода
или накопленный ``partial_reply`` служебного графа — одна и та же механика.
Три части в промпт идут раздельно: срез истории, реплика, шаг.

Закрывает по двум основаниям: ИИ видит, что шаг закрылся по диалогу; код
видит, что счётчик исчерпан. ``inform`` чекеру не отдаём — его закрывает
``close_delivered_inform`` по факту доставки. Модель не ответила —
пропускаем и идём дальше.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.config import settings
from graph.history import last_user_text
from script.build import AnyStep, CompiledScript
from script.models import SalesStep, Step
from script.planner import exhausted, is_closed, iter_available, profile_has, render_step_text
from script.source import registry
from script.store import ScriptProgress, progress_from_state
from utils.llm_gen import LLMTurnFailed, astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)

#: Потолок ходов в срезе истории для уже заданных шагов.
HISTORY_TURN_CAP = 12

#: Критерий закрытия по виду шага — и в промпт, и в код.
_KIND_CRITERIA: dict[str, str] = {
    "question": "клиент ответил по существу вопроса шага",
    "inform": "содержание дошло до клиента",
    "inform_check": "клиент ответил на проверочный вопрос",
    "action": "результат виден в диалоге: время встречи, анкета, удержанное место",
}


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
            "Закрывается ли шаг по критерию своего вида. «Потом скажу» — ответ есть, шаг не закрыт."
        ),
    )
    client_asks_inform: bool = Field(
        default=False,
        description=(
            "Просит ли клиент рассказать про обучение, условия, стоимость, "
            "состав пакета или как проходит учёба — то есть нужен информирующий блок."
        ),
    )


class CheckerClient(Protocol):
    """Контракт вызова модели чекера — подменяется в тестах."""

    async def judge(
        self,
        *,
        history_slice: str,
        client_reply: str,
        step: AnyStep,
        step_text: str | None,
    ) -> CheckerVerdict | None:
        """Оценивает один шаг. None — модель не ответила."""


def closure_criterion(step: AnyStep) -> str:
    """Человекочитаемый критерий закрытия для вида шага."""
    if isinstance(step, SalesStep):
        return "требования шага выполнены по диалогу"
    return _KIND_CRITERIA.get(step.kind, "задача шага решена")


class LlmCheckerClient:
    """Чекер на быстрой модели с короткой схемой."""

    async def judge(
        self,
        *,
        history_slice: str,
        client_reply: str,
        step: AnyStep,
        step_text: str | None,
    ) -> CheckerVerdict | None:
        """Вызывает модель; при сбое возвращает None."""
        criterion = closure_criterion(step)
        system = (
            "Ты проверяешь, закрылся ли шаг скрипта телефонного разговора.\n"
            "Тебе даны ТРИ РАЗДЕЛЬНЫХ блока: срез истории, реплика клиента и шаг.\n"
            "Не склеивай их мысленно как один текст: реплика — отдельный блок.\n"
            "Ответь только по схеме: reply_usable, step_closed и client_asks_inform.\n"
            "step_closed=true только если выполнен критерий закрытия для вида шага.\n"
            "Критерии по виду:\n"
            "- question — клиент ответил по существу вопроса шага;\n"
            "- inform — содержание дошло до клиента;\n"
            "- inform_check — клиент ответил на проверочный вопрос;\n"
            "- action — результат виден в диалоге: время встречи, анкета,"
            " удержанное место.\n"
            "- шаг продаж (requirements) — требования шага выполнены по диалогу.\n"
            "«Потом скажу», уклонение, шутка, ответ не по теме шага —"
            " step_closed=false.\n"
            "client_asks_inform=true, если клиент просит рассказать про обучение,"
            " условия, стоимость, сроки, теорию/практику или состав пакета.\n"
            "Идентификатор шага не возвращай."
        )
        if isinstance(step, SalesStep):
            human_parts = [
                f"### Срез истории\n{history_slice or '(пусто)'}",
                f"### Реплика клиента\n{client_reply or '(пусто)'}",
                "### Шаг",
                f"id (служебно, не возвращай): {step.id}",
                f"Название: {step.name}",
                f"Критерий закрытия: {criterion}",
                f"Требования:\n{step.requirements}",
            ]
        else:
            human_parts = [
                f"### Срез истории\n{history_slice or '(пусто)'}",
                f"### Реплика клиента\n{client_reply or '(пусто)'}",
                "### Шаг",
                f"id (служебно, не возвращай): {step.id}",
                f"Вид: {step.kind}",
                f"Критерий закрытия: {criterion}",
                f"Задача: {step.goal}",
                f"Формулировка: {step_text or step.text or '—'}",
            ]
            if step.kind == "inform_check" and step.check_question:
                human_parts.append(f"Проверочный вопрос: {step.check_question}")
        human = "\n".join(human_parts)
        schema = response_format_from(CheckerVerdict, name="vector_checker")
        try:
            async with get_llm(fast=True, temperature=0.0) as llm:
                raw = await astream_structured(
                    llm,
                    [SystemMessage(content=system), HumanMessage(content=human)],
                    schema=schema,
                    text_field=None,
                    purpose="чекер",
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


def _message_text(message: BaseMessage) -> str:
    """Текст сообщения одной строкой."""
    return message.content if isinstance(message.content, str) else str(message.content)


def history_slice_for(
    messages: Sequence[BaseMessage],
    *,
    steps: Sequence[AnyStep],
    progress: ScriptProgress,
    turn: int,
    reply: str | None = None,
) -> list[BaseMessage]:
    """Срез истории под проверяемые шаги.

    От взятия самого старого шага со счётчиком > 0; для шагов со счётчиком
    ноль — с начала звонка. Сверху — потолок по числу сообщений.

    Текущая реплика (полная или накопленный partial) в срез не входит:
    она уходит отдельным полем ``client_reply``. Из хвоста истории
    убираем последнее human-сообщение только если его текст совпадает
    с проверяемой репликой — иначе прошлый ответ клиента остаётся в срезе.

    Args:
        messages: полная история хода.
        steps: шаги, которые сейчас проверяем.
        progress: прогресс скрипта.
        turn: номер текущего хода.
        reply: текст реплики, который уходит отдельным блоком; ``None`` —
            срез как раньше (отрезать хвостовой human).

    Returns:
        Срез сообщений без проверяемой реплики.
    """
    if not messages:
        return []
    body = list(messages)
    if body and body[-1].type == "human":
        last_text = _message_text(body[-1])
        if reply is None or last_text == reply:
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


def _script_from_state(state: Mapping[str, Any]) -> CompiledScript:
    """Достаёт скрипт звонка из состояния или дефолтов настроек."""
    return registry.get(
        state.get("script_id") or settings.script_id,
        state.get("script_version") or settings.script_version,
    )


async def check_pass(
    state: Mapping[str, Any],
    *,
    reply: str,
    judge: CheckerClient | None = None,
    progress: ScriptProgress | None = None,
    attempt_limit: int | None = None,
) -> tuple[ScriptProgress, list[tuple[str, str]], bool]:
    """Один проход чекера по заданному тексту реплики.

    Общее ядро для синхронного узла основного хода и служебного графа.
    Источник текста (полная реплика или накопленный partial) роли не играет.

    Порядок: сначала модельный проход (и закрытие по fills), затем
    счётчиковое закрытие — чтобы шаг на последней попытке закрылся
    основанием «диалог», а не «счётчик».

    Args:
        state: состояние звонка (скрипт, профиль, история, turn).
        reply: текст реплики для анализа.
        judge: клиент модели; пусто — боевой.
        progress: прогресс; пусто — из зеркала состояния.
        attempt_limit: порог попыток; пусто — из настроек.

    Returns:
        Обновлённый прогресс, список закрытий ``(step_id, основание)``
        и признак «клиент просит рассказать про обучение».
    """
    script = _script_from_state(state)
    updated = ScriptProgress.from_mapping(
        (progress if progress is not None else progress_from_state(state)).to_dict()
    )
    profile = dict(state.get("profile") or {})
    # Профиль из кеша прогресса — для закрытия по fills.
    for key, value in updated.profile.items():
        if value and key not in profile:
            profile[key] = value
    turn = int(state.get("turn") or 0)
    messages = list(state.get("messages") or [])
    limit = attempt_limit if attempt_limit is not None else settings.step_attempt_limit
    closures: list[tuple[str, str]] = []
    asks_inform = False

    # 1. question: fills уже в профиле — закрываем кодом до модели.
    for step_id in script.step_order:
        step = script.step(step_id)
        if is_closed(updated.status.get(step.id)):
            continue
        if (
            isinstance(step, Step)
            and step.kind == "question"
            and step.fills
            and int(updated.attempts.get(step.id, 0)) > 0
            and any(profile_has(profile, key) for key in step.fills)
        ):
            updated.status[step.id] = "closed"
            closures.append((step.id, "диалог"))

    # inform закрывает код по доставке — в pending чекера не отдаём.
    # Включаем и шаги на пороге счётчика: модель смотрит раньше счётчика.
    # В формате продаж видов нет — все висящие идут в чекер.
    pending = []
    for step in iter_available(script, status=updated.status, profile=profile):
        if int(updated.attempts.get(step.id, 0)) <= 0:
            continue
        if isinstance(step, Step) and step.kind == "inform":
            continue
        pending.append(step)
    if pending and reply.strip():
        client = judge or LlmCheckerClient()
        history = history_slice_for(
            messages,
            steps=pending,
            progress=updated,
            turn=turn,
            reply=reply,
        )
        history_text = _format_history(history)

        for step in pending:
            verdict = await client.judge(
                history_slice=history_text,
                client_reply=reply,
                step=step,
                step_text=render_step_text(step, profile),
            )
            if verdict is None:
                # Модель не ответила — шаги не трогаем, ход продолжается.
                break
            if verdict.client_asks_inform:
                asks_inform = True
            if not verdict.reply_usable:
                break
            if verdict.step_closed:
                updated.status[step.id] = "closed"
                closures.append((step.id, "диалог"))
                continue
            break

    # 2. Счётчик — после модели, только ещё открытые.
    for step in iter_available(script, status=updated.status, profile=profile):
        if exhausted(step, updated.attempts, limit=limit):
            updated.status[step.id] = "closed"
            closures.append((step.id, "счётчик"))

    return updated, closures, asks_inform


async def run_checker(
    *,
    script: CompiledScript,
    progress: ScriptProgress,
    messages: Sequence[BaseMessage],
    profile: dict[str, str],
    turn: int,
    client: CheckerClient | None = None,
    attempt_limit: int | None = None,
) -> tuple[ScriptProgress, list[tuple[str, str]]]:
    """Закрывает шаги по счётчику и по диалогу.

    Обёртка над ``check_pass``: текст реплики берётся из хвоста истории.
    Сигнатура сохранена для синхронного узла и существующих тестов.

    Args:
        script: скомпилированный скрипт.
        progress: текущий прогресс (будет изменён копией).
        messages: история звонка.
        profile: профиль для доступности шагов.
        turn: номер хода.
        client: клиент модели; пусто — боевой.
        attempt_limit: порог попыток задать шаг; пусто — из настроек.

    Returns:
        Обновлённый прогресс и список закрытий ``(step_id, основание)``.
    """
    reply = last_user_text(list(messages))
    state: dict[str, Any] = {
        "script_id": script.id,
        "script_version": script.version,
        "messages": list(messages),
        "profile": profile,
        "turn": turn,
    }
    updated, closures, _asks = await check_pass(
        state,
        reply=reply,
        judge=client,
        progress=progress,
        attempt_limit=attempt_limit,
    )
    return updated, closures


def close_delivered_inform(
    *,
    script: CompiledScript,
    progress: ScriptProgress,
    pending_step: str | None,
    delivered: bool,
) -> ScriptProgress:
    """Закрывает шаг ``inform`` после успешной доставки реплики.

    ``inform_check`` сюда не входит: его закрывает чекер по ответу на
    проверочный вопрос, а не факт произнесения блока.

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
    if (
        isinstance(step, Step)
        and step.kind == "inform"
        and int(updated.attempts.get(step.id, 0)) > 0
    ):
        updated.status[step.id] = "closed"
    return updated
