"""Сборка скрипта: сырые данные → скомпилированный объект.

`build_script` — чистая функция без сети и без ввода-вывода. Здесь же вся
валидация: ссылки разрешены, поля профиля объявлены, недостижимых шагов нет,
у каждого дословного блока есть текст. Скрипт падает при загрузке, а не
посреди звонка.

Скомпилированный объект неизменяемый и живёт в памяти процесса. **В состояние
он не кладётся**: в состоянии лежат только идентификатор и версия, а тексты
достаются отсюда по идентификатору. Состояние получается маленьким и дешёвым,
скрипт — тяжёлым и общим.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from script.models import Help, Objection, ProfileField, RawScript, ScriptParams, Step


class ScriptError(ValueError):
    """Скрипт не проходит проверку и не может быть собран."""


@dataclass(frozen=True)
class CompiledScript:
    """Готовый к работе скрипт разговора.

    Неизменяемый: словари и списки внутри собраны один раз при сборке и
    больше не трогаются. Один и тот же объект обслуживает все звонки,
    идущие на этой версии.
    """

    id: str
    version: str
    persona: Any
    opening_line: str
    rules: tuple[str, ...]
    params: ScriptParams
    steps: Mapping[str, Step]
    #: Порядок объявления — тай-брейк при равном приоритете.
    step_order: tuple[str, ...]
    profile_fields: Mapping[str, ProfileField]
    helps: Mapping[str, Help]
    objections: Mapping[str, Objection]
    #: Поле профиля → шаг, который его заполняет. Нужно планировщику, чтобы
    #: понять, кто закроет недостающее требование.
    filled_by: Mapping[str, str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        """Ключ кэша: идентификатор и версия."""
        return (self.id, self.version)

    def step(self, step_id: str) -> Step:
        """Возвращает шаг по идентификатору.

        Args:
            step_id: идентификатор шага.

        Returns:
            Описание шага.

        Raises:
            KeyError: шага с таким идентификатором нет.
        """
        return self.steps[step_id]

    def aside(self, aside_id: str) -> Help | Objection | None:
        """Возвращает справку или возражение по идентификатору.

        Args:
            aside_id: идентификатор из справок или возражений.

        Returns:
            Найденная запись или None.
        """
        return self.helps.get(aside_id) or self.objections.get(aside_id)


def _check_unique_ids(raw: RawScript) -> None:
    """Проверяет, что идентификаторы шагов, справок и возражений не дублируются."""
    for title, items in (
        ("шага", [s.id for s in raw.steps]),
        ("справки", [h.id for h in raw.helps]),
        ("возражения", [o.id for o in raw.objections]),
        ("поля профиля", [f.key for f in raw.profile_fields]),
    ):
        seen: set[str] = set()
        for item in items:
            if item in seen:
                raise ScriptError(f"Повтор идентификатора {title}: {item!r}")
            seen.add(item)

    common = {h.id for h in raw.helps} & {o.id for o in raw.objections}
    if common:
        raise ScriptError(f"Идентификатор занят и справкой, и возражением: {sorted(common)}")


def _check_references(raw: RawScript) -> None:
    """Проверяет, что все ссылки на поля профиля и шаги разрешимы."""
    fields = {f.key for f in raw.profile_fields}
    step_ids = {s.id for s in raw.steps}

    for step in raw.steps:
        unknown_fields = set(step.fills) | set(step.requires)
        if step.branches is not None:
            unknown_fields.add(step.branches.field)
        missing = sorted(unknown_fields - fields)
        if missing:
            raise ScriptError(f"Шаг {step.id!r} ссылается на необъявленные поля профиля: {missing}")

        missing_steps = sorted(set(step.after) - step_ids)
        if missing_steps:
            raise ScriptError(f"Шаг {step.id!r} ждёт несуществующие шаги: {missing_steps}")

    for objection in raw.objections:
        missing = sorted(set(objection.sets) - fields)
        if missing:
            raise ScriptError(
                f"Возражение {objection.id!r} пишет в необъявленные поля профиля: {missing}"
            )


def _check_texts(raw: RawScript) -> None:
    """Проверяет, что у дословных блоков и информирования есть что произнести."""
    for step in raw.steps:
        has_text = bool(step.text) or step.branches is not None
        if step.verbatim and not has_text:
            raise ScriptError(f"Дословный шаг {step.id!r} без текста")
        if step.kind in ("inform", "inform_check") and not has_text:
            raise ScriptError(f"Шаг информирования {step.id!r} без текста")
        if step.kind == "inform_check" and not step.check_question:
            raise ScriptError(f"Шаг {step.id!r} требует проверочного вопроса")
        if step.kind == "question" and not (step.goal or step.text):
            raise ScriptError(f"Шаг-вопрос {step.id!r} без цели и без текста")

    if not raw.params.fallback:
        raise ScriptError("В параметрах скрипта нет аварийной реплики (params.fallback)")
    if not raw.params.unknown:
        raise ScriptError("В параметрах скрипта нет реплики на «не знаю» (params.unknown)")


def _check_reachable(raw: RawScript) -> None:
    """Ищет шаги, до которых разговор не дойдёт никогда.

    Недостижим шаг, чьи требования не закрывает ни один шаг, а также шаг,
    попавший в цикл ожидания: A ждёт B, B ждёт A.
    """
    filled_by = {f: s.id for s in raw.steps for f in s.fills}
    prefilled = {f.key for f in raw.profile_fields if f.prefilled}

    for step in raw.steps:
        orphan = [f for f in step.requires if f not in filled_by and f not in prefilled]
        if orphan:
            raise ScriptError(
                f"Шаг {step.id!r} недостижим: поля {sorted(orphan)} никто не заполняет"
            )

    # Топологическая проверка ожиданий между шагами.
    pending = {s.id: set(s.after) for s in raw.steps}
    resolved: set[str] = set()
    while True:
        ready = {sid for sid, deps in pending.items() if deps <= resolved}
        if not ready:
            break
        resolved |= ready
        pending = {sid: deps for sid, deps in pending.items() if sid not in resolved}
    if pending:
        raise ScriptError(f"Шаги замкнуты в цикл ожидания: {sorted(pending)}")


def build_script(raw: RawScript) -> CompiledScript:
    """Собирает скомпилированный скрипт из сырых данных.

    Чистая функция: одинаковый вход даёт одинаковый выход, сети и диска здесь
    нет. Всё, что может быть не так со скриптом, выясняется тут — на загрузке,
    а не в середине звонка.

    Args:
        raw: разобранные данные скрипта.

    Returns:
        Неизменяемый скомпилированный скрипт.

    Raises:
        ScriptError: скрипт не проходит одну из проверок.
    """
    if not raw.steps:
        raise ScriptError("В скрипте нет ни одного шага")
    if not raw.version:
        raise ScriptError("У скрипта не указана версия")

    _check_unique_ids(raw)
    _check_references(raw)
    _check_texts(raw)
    _check_reachable(raw)

    return CompiledScript(
        id=raw.id,
        version=raw.version,
        persona=raw.persona,
        opening_line=raw.opening_line,
        rules=tuple(raw.rules),
        params=raw.params,
        steps={s.id: s for s in raw.steps},
        step_order=tuple(s.id for s in raw.steps),
        profile_fields={f.key: f for f in raw.profile_fields},
        helps={h.id: h for h in raw.helps},
        objections={o.id: o for o in raw.objections},
        filled_by={f: s.id for s in raw.steps for f in s.fills},
    )
