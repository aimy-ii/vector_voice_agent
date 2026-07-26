"""Сборка скрипта: сырые данные → скомпилированный объект.

`build_script` — чистая функция без сети и без ввода-вывода. Здесь же вся
валидация: ссылки разрешены, поля профиля объявлены, недостижимых шагов нет,
у каждого дословного блока есть текст. Скрипт падает при загрузке, а не
посреди звонка.

Скомпилированный объект неизменяемый и живёт в памяти процесса. **В состояние
он не кладётся**: в состоянии лежат только идентификатор и версия, а тексты
достаются отсюда по идентификатору. Состояние получается маленьким и дешёвым,
скрипт — тяжёлым и общим.

Новый формат (скрипт продаж) определяется по полю ``requirements`` у шагов:
у шага шесть полей, зависимостей и технических счётчиков в файле нет.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from core.config import settings
from script.models import (
    Help,
    Objection,
    ProfileField,
    RawSalesScript,
    RawScript,
    SalesStep,
    ScriptParams,
    Step,
    StepKnowledge,
)


class ScriptError(ValueError):
    """Скрипт не проходит проверку и не может быть собран."""


#: Тип шага в скомпилированном скрипте: старый или продажи.
AnyStep = Step | SalesStep


@dataclass(frozen=True)
class CompiledScript:
    """Готовый к работе скрипт разговора.

    Неизменяемый: словари и списки внутри собраны один раз при сборке и
    больше не трогаются. Один и тот же объект обслуживает все звонки,
    идущие на этой версии.

    ``is_sales`` — True для нового формата (шаги с ``requirements``).
    """

    id: str
    version: str
    opening_line: str
    params: ScriptParams
    steps: Mapping[str, AnyStep]
    #: Порядок объявления — тай-брейк при равном приоритете; для продаж —
    #: порядок по возрастанию ``order``.
    step_order: tuple[str, ...]
    profile_fields: Mapping[str, ProfileField]
    helps: Mapping[str, Help]
    objections: Mapping[str, Objection]
    #: Поле профиля → шаг, который его заполняет. Нужно планировщику, чтобы
    #: понять, кто закроет недостающее требование. В формате продаж пусто.
    filled_by: Mapping[str, str] = field(default_factory=dict)
    #: Новый формат: список шагов продаж, без kind/fills/needs.
    is_sales: bool = False

    @property
    def key(self) -> tuple[str, str]:
        """Ключ кэша: идентификатор и версия."""
        return (self.id, self.version)

    def step(self, step_id: str) -> AnyStep:
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


def params_from_settings() -> ScriptParams:
    """Собирает параметры скрипта из настроек агента.

    Для нового формата в файле скрипта params нет: заглушки и фраза при
    сбое живут рядом с персоной. Формулировки цены для продаж приходят
    из базы вместе со стоимостью — сюда кладутся запасные шаблоны.
    """
    return ScriptParams(
        price=settings.agent_price_texts,
        fillers=list(settings.agent_fillers),
        city_fillers=list(settings.agent_city_fillers),
        branch_fillers=list(settings.agent_branch_fillers),
        unknown=settings.agent_unknown,
        fallback=settings.agent_fallback,
    )


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
        has_text = bool(step.text) or step.branches is not None or bool(step.examples)
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


def _build_legacy(raw: RawScript) -> CompiledScript:
    """Собирает скрипт старого формата (v1–v3)."""
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
        opening_line=raw.opening_line,
        params=raw.params,
        steps={s.id: s for s in raw.steps},
        step_order=tuple(s.id for s in raw.steps),
        profile_fields={f.key: f for f in raw.profile_fields},
        helps={h.id: h for h in raw.helps},
        objections={o.id: o for o in raw.objections},
        filled_by={f: s.id for s in raw.steps for f in s.fills},
        is_sales=False,
    )


def _check_sales_steps(raw: RawSalesScript) -> None:
    """Проверяет шаги скрипта продаж: уникальность и непустоту полей."""
    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    for step in raw.steps:
        if step.id in seen_ids:
            raise ScriptError(f"Повтор идентификатора шага: {step.id!r}")
        seen_ids.add(step.id)
        if step.order in seen_orders:
            raise ScriptError(f"Повтор порядка шага (order): {step.order}")
        seen_orders.add(step.order)
        if not step.name.strip():
            raise ScriptError(f"У шага {step.id!r} пустое название (name)")
        if not step.requirements.strip():
            raise ScriptError(f"У шага {step.id!r} пустые требования (requirements)")
        if not step.examples:
            raise ScriptError(f"У шага {step.id!r} пустой список образцов (examples)")
        for example in step.examples:
            if not str(example).strip():
                raise ScriptError(f"У шага {step.id!r} пустой образец в examples")


def _build_sales(raw: RawSalesScript) -> CompiledScript:
    """Собирает скрипт продаж: шаги по ``order``, params из настроек."""
    if not raw.steps:
        raise ScriptError("В скрипте нет ни одного шага")
    if not raw.version:
        raise ScriptError("У скрипта не указана версия")

    _check_sales_steps(raw)

    ranked = sorted(raw.steps, key=lambda s: s.order)
    # Нормализуем knowledge, чтобы оба списка всегда были списками.
    steps: dict[str, SalesStep] = {}
    for step in ranked:
        knowledge = step.knowledge or StepKnowledge()
        steps[step.id] = step.model_copy(
            update={
                "knowledge": StepKnowledge(
                    есть_в_базе=list(knowledge.есть_в_базе),
                    нужно_завести=list(knowledge.нужно_завести),
                )
            }
        )

    return CompiledScript(
        id=raw.id,
        version=raw.version,
        opening_line="",
        params=params_from_settings(),
        steps=steps,
        step_order=tuple(s.id for s in ranked),
        profile_fields={},
        helps={},
        objections={},
        filled_by={},
        is_sales=True,
    )


def build_script(raw: RawScript | RawSalesScript) -> CompiledScript:
    """Собирает скомпилированный скрипт из сырых данных.

    Чистая функция: одинаковый вход даёт одинаковый выход, сети и диска здесь
    нет. Всё, что может быть не так со скриптом, выясняется тут — на загрузке,
    а не в середине звонка.

    Args:
        raw: разобранные данные скрипта (старый или продажи).

    Returns:
        Неизменяемый скомпилированный скрипт.

    Raises:
        ScriptError: скрипт не проходит одну из проверок.
    """
    if isinstance(raw, RawSalesScript):
        return _build_sales(raw)
    return _build_legacy(raw)
