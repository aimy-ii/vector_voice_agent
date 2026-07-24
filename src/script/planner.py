"""Планировщик скрипта: шапка шагов для генератора.

Куда идти дальше по скрипту, решает код, а не модель. Все функции здесь
чистые: на вход скомпилированный скрипт, статусы шагов, счётчики и профиль,
на выход — идентификаторы и тексты. Ни сети, ни модели, ни состояния.

Статусов два: ``pending`` (ждёт отработки) и ``closed`` (закрыт). Заданность
говорит счётчик: ноль — не спрашивали, больше нуля — вопрос уходил в
генерацию. Пропуск по признаку статусом не помечается: шаг просто не
попадает в шапку при чтении.
"""

from __future__ import annotations

import re
from typing import Literal, Mapping

from script.build import CompiledScript
from script.models import Step

#: Статус шага в состоянии звонка.
#:
#: * ``pending`` — ждёт отработки;
#: * ``closed`` — закрыт чекером.
StepStatus = Literal["pending", "closed"]

#: Статусы, после которых к шагу не возвращаются.
CLOSED: frozenset[str] = frozenset({"closed"})

#: Совместимость со старыми слепками в треде (идущие звонки на v1).
_LEGACY_CLOSED: frozenset[str] = frozenset({"done", "refused", "skipped", "closed"})

#: Виды шагов, которые рассказывают, а не спрашивают.
_INFORM_KINDS: frozenset[str] = frozenset({"inform", "inform_check"})

#: Клиент сам спросил про обучение / условия / состав пакета.
_CLIENT_ASKS_INFORM = re.compile(
    r"обучен|условия|что\s+вход|стоимость|сколько\s+стоит|срок|"
    r"как\s+проход|теория|практик|пакет|под\s+ключ|что\s+включ",
    re.IGNORECASE,
)


def is_closed(status: str | None) -> bool:
    """Закрыт ли шаг.

    Args:
        status: статус шага или None, если о шаге ещё ничего не известно.

    Returns:
        True, если возвращаться к шагу не нужно.
    """
    return status in _LEGACY_CLOSED


def profile_has(profile: Mapping[str, str], key: str) -> bool:
    """Заполнено ли поле профиля непустым значением.

    Args:
        profile: собранный профиль.
        key: ключ поля.

    Returns:
        True, если значение есть и оно непустое.
    """
    value = profile.get(key)
    return bool(value and str(value).strip())


def should_skip(step: Step, profile: Mapping[str, str]) -> bool:
    """Сработало ли условие пропуска шага.

    Такой шаг не попадает в шапку: отдельного статуса «пропущен» нет.

    Args:
        step: описание шага.
        profile: собранный профиль.

    Returns:
        True, если шаг нужно отсеять при чтении скрипта.
    """
    if step.fills and all(profile_has(profile, key) for key in step.fills):
        return True

    rule = step.skip_when
    if rule is None:
        return False
    if rule.filled and all(profile_has(profile, key) for key in rule.filled):
        return True
    for key, values in rule.equals.items():
        current = str(profile.get(key, "")).strip().lower()
        if current and current in {v.strip().lower() for v in values}:
            return True
    return False


def is_available(
    step: Step,
    *,
    status: Mapping[str, str],
    profile: Mapping[str, str],
) -> bool:
    """Доступен ли шаг прямо сейчас.

    Args:
        step: описание шага.
        status: статусы шагов.
        profile: собранный профиль.

    Returns:
        True, если шаг не закрыт и все его условия выполнены.
    """
    if is_closed(status.get(step.id)):
        return False
    if should_skip(step, profile):
        return False
    if not all(is_closed(status.get(dep)) for dep in step.after):
        return False
    return all(profile_has(profile, key) for key in step.requires)


def iter_available(
    script: CompiledScript,
    *,
    status: Mapping[str, str],
    profile: Mapping[str, str],
) -> list[Step]:
    """Доступные шаги в порядке приоритета и объявления.

    Args:
        script: скомпилированный скрипт.
        status: статусы шагов.
        profile: собранный профиль.

    Returns:
        Список доступных шагов с верхушки.
    """
    ranked: list[tuple[int, int, Step]] = []
    for order, step_id in enumerate(script.step_order):
        step = script.step(step_id)
        if not is_available(step, status=status, profile=profile):
            continue
        ranked.append((step.priority, order, step))
    ranked.sort(key=lambda item: item[:2])
    return [item[2] for item in ranked]


def client_asks_inform(text: str) -> bool:
    """Клиент сам спросил про обучение, условия или состав пакета.

    Args:
        text: реплика клиента.

    Returns:
        True, если в тексте есть повод выдать информирующий блок.
    """
    return bool(text and _CLIENT_ASKS_INFORM.search(text))


def answered_inform_check(
    script: CompiledScript,
    *,
    status: Mapping[str, str],
    pending_step: str | None,
) -> bool:
    """Клиент ответил на проверочный вопрос предыдущего информирующего блока.

    Args:
        script: скомпилированный скрипт.
        status: статусы шагов после чекера.
        pending_step: шаг прошлого хода (с проверочным вопросом).

    Returns:
        True, если прошлый ``inform_check`` только что закрыт.
    """
    if not pending_step:
        return False
    step = script.steps.get(pending_step)
    if step is None or step.kind != "inform_check":
        return False
    return is_closed(status.get(pending_step))


def _fresh_questions_left(
    script: CompiledScript,
    *,
    status: Mapping[str, str],
    attempts: Mapping[str, int],
    profile: Mapping[str, str],
) -> bool:
    """Есть ли ещё незаданный вопрос или действие среди доступных шагов."""
    for step in iter_available(script, status=status, profile=profile):
        if int(attempts.get(step.id, 0)) > 0:
            continue
        if step.kind not in _INFORM_KINDS:
            return True
    return False


def script_head(
    script: CompiledScript,
    *,
    status: Mapping[str, str],
    attempts: Mapping[str, int],
    profile: Mapping[str, str],
    inform_reason: bool = False,
) -> list[Step]:
    """Шапка для генератора: все уже заданные и один новый с верхушки.

    Информирующий шаг попадает в шапку как новый только по поводу: клиент
    спросил сам, ответил на проверочный вопрос предыдущего блока, либо
    вопросов среди доступных больше нет. Иначе ждёт, берём вопрос.

    Args:
        script: скомпилированный скрипт.
        status: статусы шагов.
        attempts: счётчики попыток.
        profile: собранный профиль.
        inform_reason: внешний повод (вопрос клиента или ответ на проверку).

    Returns:
        Список шагов: сначала со счётчиком больше нуля, затем один с нулём.
    """
    asked: list[Step] = []
    fresh: Step | None = None
    questions_left = _fresh_questions_left(
        script, status=status, attempts=attempts, profile=profile
    )
    allow_inform = inform_reason or not questions_left

    for step in iter_available(script, status=status, profile=profile):
        count = int(attempts.get(step.id, 0))
        if count > 0:
            asked.append(step)
            continue
        if fresh is not None:
            continue
        if step.kind in _INFORM_KINDS and not allow_inform:
            continue
        fresh = step

    if fresh is not None:
        return [*asked, fresh]
    return asked


def pick_step(
    script: CompiledScript,
    *,
    status: Mapping[str, str],
    profile: Mapping[str, str],
    resume: str | None = None,
    attempts: Mapping[str, int] | None = None,
    inform_reason: bool = False,
) -> Step | None:
    """Выбирает ведущий шаг хода (первый из шапки или resume).

    Args:
        script: скомпилированный скрипт.
        status: статусы шагов.
        profile: собранный профиль.
        resume: шаг, на который надо вернуться после отработки вопроса.
        attempts: счётчики попыток; нужны для шапки.
        inform_reason: повод выдать информирующий блок.

    Returns:
        Описание шага или None, если закрывать больше нечего.
    """
    counts = attempts or {}
    if resume and resume in script.steps:
        step = script.step(resume)
        if is_available(step, status=status, profile=profile):
            return step

    head = script_head(
        script,
        status=status,
        attempts=counts,
        profile=profile,
        inform_reason=inform_reason,
    )
    return head[0] if head else None


def peek_next_step(
    script: CompiledScript,
    *,
    current: Step,
    status: Mapping[str, str],
    profile: Mapping[str, str],
    attempts: Mapping[str, int] | None = None,
    inform_reason: bool = False,
) -> Step | None:
    """Какой шаг откроется, если текущий закроется прямо сейчас.

    Args:
        script: скомпилированный скрипт.
        current: шаг, который клиент закрывает этим ответом.
        status: статусы шагов на этот ход.
        profile: собранный профиль на этот ход.
        attempts: счётчики попыток.
        inform_reason: повод выдать информирующий блок.

    Returns:
        Следующий шаг или None, если после закрытия текущего открывать нечего.
    """
    preview_status = dict(status)
    preview_status[current.id] = "closed"

    preview_profile = dict(profile)
    for key in current.fills:
        if not profile_has(preview_profile, key):
            preview_profile[key] = "_"

    return pick_step(
        script,
        status=preview_status,
        profile=preview_profile,
        attempts=attempts,
        inform_reason=inform_reason,
    )


def blocked_by(
    script: CompiledScript,
    step: Step,
    *,
    status: Mapping[str, str],
    profile: Mapping[str, str],
) -> list[str]:
    """Объясняет, чего шагу не хватает, чтобы открыться.

    Args:
        script: скомпилированный скрипт.
        step: описание шага.
        status: статусы шагов.
        profile: собранный профиль.

    Returns:
        Список причин человеческим текстом.
    """
    reasons: list[str] = []
    for dep in step.after:
        if not is_closed(status.get(dep)):
            reasons.append(f"ждёт шаг {dep}")
    for key in step.requires:
        if not profile_has(profile, key):
            owner = script.filled_by.get(key, "никто")
            reasons.append(f"нужно поле {key} (заполняет {owner})")
    return reasons


def render_step_text(step: Step, profile: Mapping[str, str]) -> str | None:
    """Собирает текст шага с учётом ветвления по значению профиля.

    Args:
        step: описание шага.
        profile: собранный профиль.

    Returns:
        Текст шага или None, если текста у шага нет.
    """
    if step.branches is not None:
        value = str(profile.get(step.branches.field, "")).strip().lower()
        for case, text in step.branches.cases.items():
            if case.strip().lower() == value:
                return text
        return step.branches.default or step.text
    return step.text


def next_attempt(attempts: Mapping[str, int], step_id: str) -> int:
    """Возвращает номер следующей попытки по шагу.

    Args:
        attempts: счётчики возвратов к шагам.
        step_id: идентификатор шага.

    Returns:
        Номер попытки, начиная с единицы.
    """
    return int(attempts.get(step_id, 0)) + 1


def exhausted(step: Step, attempts: Mapping[str, int], *, limit: int) -> bool:
    """Исчерпан ли порог попыток по шагу.

    Порог задаётся окружением, не полем шага: на формулировку счётчик не
    влияет, закрытие делает чекер.

    Args:
        step: описание шага.
        attempts: счётчики попыток.
        limit: порог из настроек.

    Returns:
        True, если чекер должен закрыть шаг без вызова модели.
    """
    return int(attempts.get(step.id, 0)) >= limit


def steps_to_skip(
    script: CompiledScript,
    *,
    status: Mapping[str, str],
    profile: Mapping[str, str],
) -> list[str]:
    """Идентификаторы шагов, отсеянных условием пропуска.

    Статусом они не помечаются; функция нужна тестам и журналу.

    Args:
        script: скомпилированный скрипт.
        status: статусы шагов.
        profile: собранный профиль.

    Returns:
        Идентификаторы отсеянных незакрытых шагов.
    """
    return [
        step_id
        for step_id in script.step_order
        if not is_closed(status.get(step_id)) and should_skip(script.step(step_id), profile)
    ]
