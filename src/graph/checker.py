"""Чекер шагов: единственная точка, где шаг уходит из скрипта.

Контракт входа не знает источника текста: полная реплика основного хода
или накопленный ``partial_reply`` служебного графа — одна и та же механика.
Три части в промпт идут раздельно: срез истории, реплика, шаг.

Закрывает только судья: по диалогу или потому что дальше спрашивать
бессмысленно. ``inform`` чекеру не отдаём — его закрывает
``close_delivered_inform`` по факту доставки. Модель не ответила —
пропускаем и идём дальше.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.config import settings
from graph.history import last_user_text, normalize
from graph.log_fmt import format_check_pending
from script.build import AnyStep, CompiledScript
from script.models import SalesStep, Step
from script.planner import is_closed, iter_available, profile_has, render_step_text
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
    asking_pointless: bool = Field(
        default=False,
        description=(
            "Дальше спрашивать бессмысленно: шаг безнадёжно висит — пора закрыть "
            "без ответа по существу. Не путать с обычным уклонением на один ход."
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
        attempts: int = 0,
        age: int = 0,
        in_work: bool = False,
    ) -> CheckerVerdict | None:
        """Оценивает один шаг. None — модель не ответила."""


def checker_system_prompt(*, in_work: bool) -> str:
    """Системная часть промпта чекера.

    Args:
        in_work: шаг уже отдан генератору (взят в работу).

    Returns:
        Текст системного сообщения под ветку.
    """
    lines = [
        "Ты проверяешь, закрылся ли шаг скрипта телефонного разговора.",
        "Тебе даны ТРИ РАЗДЕЛЬНЫХ блока: срез истории, реплика клиента и шаг.",
        "Не склеивай их мысленно как один текст: реплика — отдельный блок.",
        (
            "Ответь только по схеме: reply_usable, step_closed,"
            " asking_pointless и client_asks_inform."
        ),
        "step_closed=true только если выполнен критерий закрытия для вида шага.",
        "Критерии по виду:",
        "- question — клиент ответил по существу вопроса шага;",
        "- inform — содержание дошло до клиента;",
        "- inform_check — клиент ответил на проверочный вопрос;",
        ("- action — результат виден в диалоге: время встречи, анкета, удержанное место."),
        "- шаг продаж (requirements) — требования шага выполнены по диалогу.",
        ("«Потом скажу», уклонение, шутка, ответ не по теме шага — step_closed=false."),
        (
            "Просьба повторить сказанное, переспрос агента или сообщение,"
            " что клиент не расслышал, — не ответ по существу шага:"
            " step_closed=false, даже если нужное уже звучало в срезе"
            " истории раньше."
        ),
        (
            "Срез истории даёт контекст разговора, но закрытие определяет"
            " реакция клиента на шаг, а не то, что агент успел рассказать;"
            " для inform критерий по-прежнему — содержание дошло до клиента."
        ),
        (
            "client_asks_inform=true, если клиент просит рассказать про обучение,"
            " условия, стоимость, сроки, теорию/практику или состав пакета."
        ),
        "Идентификатор шага не возвращай.",
    ]
    if in_work:
        lines.extend(
            [
                (
                    "Шаг взят в работу генератором. В блоке шага указан возраст —"
                    " сколько ходов прошло с момента взятия."
                ),
                (
                    "Реши: закрыт ли шаг ответами в диалоге (step_closed=true);"
                    " ещё висит и имеет смысл спрашивать снова"
                    " (step_closed=false, asking_pointless=false);"
                    " или висит безнадёжно и дальше спрашивать бессмысленно"
                    " (asking_pointless=true)."
                ),
                (
                    "Порога попыток и автоматики нет — решение только по смыслу"
                    " диалога и возрасту шага."
                ),
                (
                    "asking_pointless=true только когда продолжать спрашивать"
                    " уже бессмысленно; обычное «потом скажу» на один ход"
                    " этого не требует."
                ),
            ]
        )
    else:
        lines.append(
            "Вопрос шага ещё не задавали. Клиент мог назвать нужное сам"
            " в любой момент разговора — засчитывай упоминание по смыслу"
            " где угодно в срезе истории."
        )
    return "\n".join(lines)


def closure_criterion(step: AnyStep) -> str:
    """Человекочитаемый критерий закрытия для вида шага."""
    if isinstance(step, SalesStep):
        return "требования шага выполнены по диалогу"
    return _KIND_CRITERIA.get(step.kind, "задача шага решена")


def _age_line(*, in_work: bool, age: int, attempts: int) -> str:
    """Строка блока шага: возраст или пометка, что шаг ещё не брали."""
    if in_work:
        return f"Возраст шага (ходов с взятия): {age}"
    if attempts > 0:
        return f"Задавали: да, {attempts} раза"
    return "Задавали: нет"


def _step_age(progress: ScriptProgress, step_id: str, turn: int) -> int:
    """Возраст шага: текущий ход минус ход взятия."""
    taken = int(progress.taken_turn.get(step_id, turn))
    return max(0, turn - taken)


class LlmCheckerClient:
    """Чекер на быстрой модели с короткой схемой."""

    async def judge(
        self,
        *,
        history_slice: str,
        client_reply: str,
        step: AnyStep,
        step_text: str | None,
        attempts: int = 0,
        age: int = 0,
        in_work: bool = False,
    ) -> CheckerVerdict | None:
        """Вызывает модель; при сбое возвращает None."""
        criterion = closure_criterion(step)
        system = checker_system_prompt(in_work=in_work)
        age_line = _age_line(in_work=in_work, age=age, attempts=attempts)
        if isinstance(step, SalesStep):
            human_parts = [
                f"### Срез истории\n{history_slice or '(пусто)'}",
                f"### Реплика клиента\n{client_reply or '(пусто)'}",
                "### Шаг",
                f"id (служебно, не возвращай): {step.id}",
                f"Название: {step.name}",
                f"Критерий закрытия: {criterion}",
                age_line,
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
                age_line,
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

    От взятия самого старого шага в работе; для шагов не в работе —
    с начала звонка. Сверху — потолок по числу сообщений.

    Текущая реплика (полная или накопленный partial) в срез не входит:
    она уходит отдельным полем ``client_reply``. Из хвоста истории
    убираем последнее human-сообщение только если его текст после
    ``normalize`` совпадает с проверяемой репликой — иначе прошлый
    ответ клиента остаётся в срезе. Сами тексты сообщений в срезе
    не меняем: судье нужны знаки препинания.

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
        if reply is None or normalize(last_text) == normalize(reply):
            body = body[:-1]

    work_ids = set(progress.in_work)
    has_idle = any(step.id not in work_ids for step in steps)
    if has_idle:
        start_turn = 0
    else:
        taken_turns = [
            int(progress.taken_turn.get(step.id, turn)) for step in steps if step.id in work_ids
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

    Закрытие только по вердикту судьи (диалог или «спрашивать
    бессмысленно»). ``attempt_limit`` сохранён в сигнатуре для
    совместимости и в решениях не участвует.

    Args:
        state: состояние звонка (скрипт, профиль, история, turn).
        reply: текст реплики для анализа.
        judge: клиент модели; пусто — боевой.
        progress: прогресс; пусто — из зеркала состояния.
        attempt_limit: устаревший порог; игнорируется.

    Returns:
        Обновлённый прогресс, список закрытий ``(step_id, основание)``
        и признак «клиент просит рассказать про обучение».
    """
    del attempt_limit  # порог больше не закрывает шаги
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
    closures: list[tuple[str, str]] = []
    asks_inform = False
    work_ids = set(updated.in_work)

    # 1. question: fills уже в профиле — закрываем кодом до модели.
    for step_id in script.step_order:
        step = script.step(step_id)
        if is_closed(updated.status.get(step.id)):
            continue
        if (
            isinstance(step, Step)
            and step.kind == "question"
            and step.fills
            and step.id in work_ids
            and any(profile_has(profile, key) for key in step.fills)
        ):
            updated.status[step.id] = "closed"
            closures.append((step.id, "диалог"))

    # inform закрывает код по доставке — в pending чекера не отдаём.
    # На проверку — только взятые в работу и ещё не закрытые.
    available = iter_available(script, status=updated.status, profile=profile)
    pending = []
    rejected: list[tuple[str, str]] = []
    for step in available:
        if step.id not in work_ids:
            rejected.append((step.id, "не в работе"))
            continue
        if isinstance(step, Step) and step.kind == "inform":
            rejected.append((step.id, "inform"))
            continue
        pending.append(step)
    # Закрытые, бывшие в работе — «исчерпан»: в available их уже нет.
    pending_ids = {step.id for step in pending}
    for step_id in updated.in_work:
        if step_id in pending_ids:
            continue
        if any(sid == step_id for sid, _ in rejected):
            continue
        if is_closed(updated.status.get(step_id)):
            rejected.append((step_id, "исчерпан"))
    pending_pairs = [(step.id, int(updated.attempts.get(step.id, 0))) for step in pending]
    available_pairs = [(step.id, int(updated.attempts.get(step.id, 0))) for step in available]
    log.info(
        "[check|pending] %s",
        format_check_pending(
            pending=pending_pairs,
            rejected=rejected,
            available=available_pairs if not pending else None,
        ),
    )
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
            age = _step_age(updated, step.id, turn)
            verdict = await client.judge(
                history_slice=history_text,
                client_reply=reply,
                step=step,
                step_text=render_step_text(step, profile),
                attempts=int(updated.attempts.get(step.id, 0)),
                age=age,
                in_work=True,
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
            if verdict.asking_pointless:
                updated.status[step.id] = "closed"
                closures.append((step.id, "бессмысленно"))
                continue
            break

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
    """Закрывает шаги по вердикту судьи.

    Обёртка над ``check_pass``: текст реплики берётся из хвоста истории.
    Сигнатура сохранена для синхронного узла и существующих тестов.

    Args:
        script: скомпилированный скрипт.
        progress: текущий прогресс (будет изменён копией).
        messages: история звонка.
        profile: профиль для доступности шагов.
        turn: номер хода.
        client: клиент модели; пусто — боевой.
        attempt_limit: устаревший порог; игнорируется.

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
    if isinstance(step, Step) and step.kind == "inform" and pending_step in updated.in_work:
        updated.status[step.id] = "closed"
    return updated
