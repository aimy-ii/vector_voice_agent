"""Планировщик скрипта: какой шаг сейчас открыт.

Куда идти дальше по скрипту, решает код, а не модель. Все функции здесь
чистые: на вход скомпилированный скрипт, статусы шагов и собранный профиль,
на выход — идентификаторы и тексты. Ни сети, ни модели, ни состояния.

Порядок не задан списком, а выводится: шаг доступен, когда закрыты шаги,
которых он ждёт, и заполнены поля профиля, которые он требует. Поэтому один
скрипт описывает и Калининград, где имя не спросили вовсе, и Пермь, где его
спросили вторым, и Екатеринбург, где третьим.
"""

from __future__ import annotations

from typing import Literal, Mapping

from script.build import CompiledScript
from script.models import Step

#: Статус шага в состоянии звонка.
#:
#: * `open` — ещё не закрыт;
#: * `done` — закрыт;
#: * `refused` — клиент не ответил и переспрашивать больше нельзя;
#: * `skipped` — пропущен по признаку.
StepStatus = Literal["open", "done", "refused", "skipped"]

#: Статусы, после которых к шагу не возвращаются.
CLOSED: frozenset[str] = frozenset({"done", "refused", "skipped"})


def is_closed(status: str | None) -> bool:
    """Закрыт ли шаг.

    Args:
        status: статус шага или None, если о шаге ещё ничего не известно.

    Returns:
        True, если возвращаться к шагу не нужно.
    """
    return status in CLOSED


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

    Два основания. Первое — общее: если всё, что шаг собирает, уже известно,
    спрашивать нечего. Профиль заполняется и до начала скрипта — из первой же
    реплики бывают известны коробка, возраст и готовность записаться, и
    переспрашивать это ошибка. Второе — объявленное в скрипте условие: шаг
    умеет пропускаться по признаку, а не только закрываться, поэтому готовому
    клиенту презентация не нужна.

    Args:
        step: описание шага.
        profile: собранный профиль.

    Returns:
        True, если шаг нужно пометить пропущенным.
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
        True, если шаг открыт и все его условия выполнены.
    """
    if is_closed(status.get(step.id)):
        return False
    if not all(is_closed(status.get(dep)) for dep in step.after):
        return False
    return all(profile_has(profile, key) for key in step.requires)


def steps_to_skip(
    script: CompiledScript,
    *,
    status: Mapping[str, str],
    profile: Mapping[str, str],
) -> list[str]:
    """Собирает шаги, которые прямо сейчас нужно пометить пропущенными.

    Args:
        script: скомпилированный скрипт.
        status: статусы шагов.
        profile: собранный профиль.

    Returns:
        Идентификаторы шагов к пропуску.
    """
    return [
        step_id
        for step_id in script.step_order
        if not is_closed(status.get(step_id)) and should_skip(script.step(step_id), profile)
    ]


def pick_step(
    script: CompiledScript,
    *,
    status: Mapping[str, str],
    profile: Mapping[str, str],
    resume: str | None = None,
) -> Step | None:
    """Выбирает шаг, которым бот занимается на этом ходу.

    Порядок разрешения:

    1. Если после справки или возражения надо вернуться на место и тот шаг всё
       ещё доступен — возвращаемся туда.
    2. Иначе берётся доступный шаг с наименьшим приоритетом; при равенстве —
       тот, что объявлен раньше.

    Args:
        script: скомпилированный скрипт.
        status: статусы шагов.
        profile: собранный профиль.
        resume: шаг, на который надо вернуться после отработки вопроса.

    Returns:
        Описание шага или None, если закрывать больше нечего.
    """
    if resume and resume in script.steps:
        step = script.step(resume)
        if is_available(step, status=status, profile=profile) and not should_skip(step, profile):
            return step

    best: tuple[int, int, Step] | None = None
    for order, step_id in enumerate(script.step_order):
        step = script.step(step_id)
        if not is_available(step, status=status, profile=profile):
            continue
        if should_skip(step, profile):
            continue
        rank = (step.priority, order, step)
        if best is None or rank[:2] < best[:2]:
            best = rank
    return best[2] if best is not None else None


def peek_next_step(
    script: CompiledScript,
    *,
    current: Step,
    status: Mapping[str, str],
    profile: Mapping[str, str],
) -> Step | None:
    """Какой шаг откроется, если текущий закроется прямо сейчас.

    Считается детерминированно: текущий помечается закрытым, его ``fills`` —
    заполненными фиктивными значениями, затем прогон ``pick_step`` (с учётом
    пропусков). Ничего нового не изобретается — переиспользуются те же правила.

    Args:
        script: скомпилированный скрипт.
        current: шаг, который клиент закрывает этим ответом.
        status: статусы шагов на этот ход.
        profile: собранный профиль на этот ход.

    Returns:
        Следующий шаг или None, если после закрытия текущего открывать нечего.
    """
    preview_status = dict(status)
    preview_status[current.id] = "done"

    preview_profile = dict(profile)
    for key in current.fills:
        if not profile_has(preview_profile, key):
            preview_profile[key] = "_"

    for step_id in steps_to_skip(script, status=preview_status, profile=preview_profile):
        preview_status[step_id] = "skipped"

    return pick_step(script, status=preview_status, profile=preview_profile)


def blocked_by(
    script: CompiledScript,
    step: Step,
    *,
    status: Mapping[str, str],
    profile: Mapping[str, str],
) -> list[str]:
    """Объясняет, чего шагу не хватает, чтобы открыться.

    Нужно для логов и тестов: «шаг филиала ждёт поле города».

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

    Формат теории даёт три разные реплики, коробка передач — разный перечень
    машин: это один шаг с разным содержанием, а не разные шаги.

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


def exhausted(step: Step, attempts: Mapping[str, int]) -> bool:
    """Исчерпаны ли попытки по шагу.

    Предохранитель от зацикливания: клиент не всегда отвечает («потом скажу»,
    игнорирует и спрашивает своё). Без счётчика бот переспросит второй и
    третий раз.

    Args:
        step: описание шага.
        attempts: счётчики возвратов к шагам.

    Returns:
        True, если шаг пора закрывать отказом.
    """
    return int(attempts.get(step.id, 0)) >= step.max_attempts
