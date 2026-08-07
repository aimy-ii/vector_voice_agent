"""Контекст разговора: один накопительный документ.

Статика кладётся один раз и до конца разговора не меняется — её нельзя резать.
Динамика наполняется контекстером и едет со статусом; генератор статус читает.
В промпт документ подшивается целиком.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from graph.transcript import TranscriptEntry

#: Динамика не нужна по этой реплике.
DYN_NONE = "не требуется"
#: Ответ уже в статике или накопленном — можно генерить.
DYN_READY = "готово"
#: Фон начал разбирать реплику, предмет ещё неизвестен.
DYN_WORKING = "в работе"
#: Идёт поиск; генератор отдаёт реплику ожидания.
DYN_SEARCHING = "в поиске"
#: Поиск завершился без результата.
DYN_MISSING = "не нашлось"

#: Условия оплаты: английский ключ → русская формулировка.
_PAYMENT_LABELS: dict[str, str] = {
    "installment": "есть рассрочка",
    "installment_no_overpay": "рассрочка без переплаты",
    "installment_with_overpay": "рассрочка с переплатой",
    "matcap": "можно материнским капиталом",
    "maternal_capital": "можно материнским капиталом",
    "cash": "наличные",
    "card": "карта",
    "transfer": "перевод",
    "credit": "кредит",
    "tax_deduction": "можно оформить налоговый вычет",
}


class ConversationContext(BaseModel):
    """Единый контекст разговора.

    Attributes:
        static_text: запечённая статика города и филиала.
        dynamic_text: динамическая часть (пока пусто — следующий этап).
        dynamic_status: статус динамики; лайв ставит «в работе», контекстер
            ведёт поиск и итог, генератор читает.
        situation_slug: предмет вопроса одним-двумя словами («медкомиссия»,
            «филиалы», «пересдача»), который контекстер отдаёт вместе со
            статусом «в поиске».
        filler_spoken: заглушка по этому предмету уже произнесена (один заход).
        dynamic_reply: реплика клиента, по которой собрана текущая динамика.
            Нужна, чтобы основной ход не пересчитывал то, что лайв-канал
            уже испёк по этой же реплике.
        last_agent_reply: последняя реплика бота. Пишет основной ход, читает
            агент контекста в лайв-канале — там истории разговора нет, а без
            неё «Да.» неотличимо от вопроса.
        dynamic_turn: номер хода, на котором выставлен статус динамики.
        last_reply_hash: хеш реплики, по которой контекстер уже отработал.
        dynamic_reply_hash: хеш реплики, по которой выставлен текущий статус
            динамики. Ход сравнивает его с текущей репликой: без совпадения
            статус прошлого хода не подхватывается.
        pending_fields: ключи полей профиля, которые лайв-канал сейчас
            разбирает; для формы это состояние «уточняется».
        empty_needs: потребности, за которыми уже ходили и справочник
            вернул пусто. Повторно за ними не ходим до конца звонка:
            данных в справочнике нет, и следующая попытка ничего не изменит.
        city_slug: слаг города после фиксации.
        city_name: читаемое название города.
        branch_slug: слаг выбранного филиала.
        branch_candidates: слаги филиалов, отобранные инструментом ``branches``.
        nearby_text: блок ближайших филиалов по названному человеком месту:
            либо строка о том, что подбор идёт, либо перечень адресов, либо
            строка о том, что место не опознано. Хранится отдельно от
            ``dynamic_text``: динамика копится за звонок, а этот блок должен
            заменяться целиком, когда человек назвал другое место.
        nearby_key: ключ, по которому подбор уже сделан («город: место»).
            Совпал с текущим — пересчитывать нечего.
        city_faq: FAQ меты города (вопрос → ответ) для ``CityFaqTool``.
        conversation_ended: разговор закончен по решению фонового агента
            прощания; переставляется на каждом ходу с репликой человека.
        frozen: статика уже зафиксирована и не пересобирается.
        transcript: полная история звонка. Пишет основной ход: свои реплики
            в момент генерации, чужие — из снимка бота. Читают оба графа,
            у фонового своего снимка нет.
    """

    static_text: str = ""
    dynamic_text: str = ""
    dynamic_status: str = DYN_NONE
    situation_slug: str | None = None
    filler_spoken: bool = False
    dynamic_reply: str = ""
    last_agent_reply: str = ""
    dynamic_turn: int = 0
    last_reply_hash: str = ""
    dynamic_reply_hash: str = ""
    pending_fields: list[str] = Field(default_factory=list)
    empty_needs: list[str] = Field(default_factory=list)
    city_slug: str | None = None
    city_name: str | None = None
    branch_slug: str | None = None
    branch_candidates: list[str] = Field(default_factory=list)
    nearby_text: str = ""
    nearby_key: str = ""
    city_faq: list[dict[str, str]] = Field(default_factory=list)
    conversation_ended: bool = False
    frozen: bool = False
    transcript: list[TranscriptEntry] = Field(default_factory=list)

    def render(self) -> str:
        """Собирает документ для промпта: статика, ближайшие филиалы, динамика.

        Ближайшие идут отдельным блоком между статикой и динамикой: они
        заменяются целиком при смене места, а динамика только копится.

        Returns:
            Текст контекста; пустая строка, если ещё нечего класть.
        """
        parts: list[str] = []
        if self.static_text.strip():
            parts.append(self.static_text.strip())
        if self.nearby_text.strip():
            parts.append(self.nearby_text.strip())
        if self.dynamic_text.strip():
            parts.append(self.dynamic_text.strip())
        return "\n\n".join(parts)


def _line(label: str, value: Any) -> str | None:
    """Строка «метка: значение», если значение непустое."""
    if value is None or value == "" or value == [] or value == {}:
        return None
    if isinstance(value, list):
        text = ", ".join(str(item) for item in value)
    elif isinstance(value, dict):
        text = ", ".join(f"{k}={v}" for k, v in value.items() if v not in (None, "", [], {}))
    else:
        text = str(value)
    if not text.strip():
        return None
    return f"{label}: {text}"


def _theory_format_names(raw: Any) -> list[str]:
    """Достаёт только названия форматов теории, без рекламных абзацев."""
    names: list[str] = []
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                name = value.get("name") or value.get("title") or key
            elif isinstance(value, str) and len(value) < 40 and "\n" not in value:
                name = value
            else:
                name = key
            text = str(name).strip()
            if text:
                names.append(text)
        return names
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping):
                name = item.get("name") or item.get("title") or item.get("format")
                if name:
                    names.append(str(name).strip())
            elif isinstance(item, str):
                # Рекламный абзац — длинный текст с призывами; название короткое.
                text = item.strip()
                if text and len(text) <= 40 and "\n" not in text:
                    names.append(text)
    return [n for n in names if n]


def _document_names(raw: Any) -> list[str]:
    """Достаёт названия документов через запятую."""
    names: list[str] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping):
                name = item.get("name") or item.get("title") or item.get("doc")
                if name:
                    names.append(str(name).strip())
            elif item:
                names.append(str(item).strip())
    elif isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                name = value.get("name") or key
            else:
                name = key if value else None
            if name:
                names.append(str(name).strip())
    elif isinstance(raw, str) and raw.strip():
        names.append(raw.strip())
    return [n for n in names if n]


def _payment_phrases(payment: Mapping[str, Any]) -> list[str]:
    """Переводит условия оплаты в русские формулировки."""
    phrases: list[str] = []
    for key, value in payment.items():
        if value in (None, "", False, [], {}):
            continue
        label = _PAYMENT_LABELS.get(str(key))
        if label is None:
            # Неизвестный ключ — не пускаем английский в промпт.
            if isinstance(value, str) and value.strip():
                phrases.append(value.strip())
            continue
        if value is True:
            phrases.append(label)
        elif isinstance(value, str) and value.strip():
            phrases.append(f"{label}: {value.strip()}")
        else:
            phrases.append(label)
    return phrases


def normalize_city_faq(raw: Any) -> list[dict[str, str]]:
    """Приводит FAQ меты города к списку ``{question, answer}``.

    Args:
        raw: поле ``faq`` из меты города.

    Returns:
        Список пар вопрос/ответ; пустой, если FAQ нет.
    """
    items: list[dict[str, str]] = []
    if not raw:
        return items
    if isinstance(raw, Mapping):
        for question, answer in raw.items():
            q = str(question).strip()
            a = "" if answer is None else str(answer).strip()
            if q:
                items.append({"question": q, "answer": a})
        return items
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            q = (
                entry.get("question")
                or entry.get("q")
                or entry.get("вопрос")
                or entry.get("title")
                or ""
            )
            a = (
                entry.get("answer")
                or entry.get("a")
                or entry.get("ответ")
                or entry.get("text")
                or ""
            )
            q_text = str(q).strip()
            if q_text:
                items.append(
                    {"question": q_text, "answer": str(a).strip() if a is not None else ""}
                )
    return items


def format_city_static(
    *,
    city_slug: str,
    city_name: str,
    city_meta: Mapping[str, Any],
    price_line: str | None = None,
) -> str:
    """Собирает статику города без списка филиалов и сырых полей цены.

    Args:
        city_slug: слаг города.
        city_name: читаемое название.
        city_meta: мета города из справочника.
        price_line: готовая фраза о цене, если уже известна.

    Returns:
        Текстовый блок статики города.
    """
    vehicles = city_meta.get("vehicles") or {}
    categories = city_meta.get("categories") or []
    lines = [
        "Статика разговора (не меняется до конца звонка):",
        f"Город: {city_name} (слаг {city_slug}).",
    ]
    if categories:
        cat_parts = []
        for item in categories:
            code = item.get("code") or "?"
            duration = item.get("duration") or ""
            freq = item.get("start_frequency") or ""
            piece = f"{code}"
            if duration:
                piece += f" — {duration}"
            if freq:
                piece += f", набор: {freq}"
            cat_parts.append(piece)
        lines.append("Категории: " + "; ".join(cat_parts) + ".")
    manual = vehicles.get("manual") or []
    auto = vehicles.get("automatic") or []
    if manual or auto:
        fleet = []
        if manual:
            fleet.append("механика: " + ", ".join(str(x) for x in manual))
        if auto:
            fleet.append("автомат: " + ", ".join(str(x) for x in auto))
        if vehicles.get("fleet_age"):
            fleet.append(f"возраст парка: {vehicles['fleet_age']}")
        lines.append("Автопарк: " + "; ".join(fleet) + ".")
    theory_names = _theory_format_names(city_meta.get("theory_formats"))
    if theory_names:
        lines.append("Форматы теории: " + ", ".join(theory_names) + ".")
    doc_names = _document_names(city_meta.get("documents"))
    if doc_names:
        lines.append("Документы: " + ", ".join(doc_names) + ".")
    messengers = city_meta.get("messengers")
    row = _line("Мессенджеры", messengers)
    if row:
        lines.append(row + ".")
    payment = city_meta.get("payment") or {}
    if isinstance(payment, Mapping) and payment:
        phrases = _payment_phrases(payment)
        if phrases:
            lines.append("Оплата: " + "; ".join(phrases) + ".")
    if city_meta.get("call_hours"):
        lines.append(f"Часы колл-центра: {city_meta['call_hours']}.")
    contacts = city_meta.get("contacts") or city_meta.get("phones") or {}
    if contacts:
        row = _line("Контакты", contacts)
        if row:
            lines.append(row + ".")
    if price_line:
        lines.append(f"Цена (готовая фраза, произносить только так): {price_line}")
    count = city_meta.get("branches_count")
    if count:
        lines.append(
            f"Филиалов в городе: {count} (служебно, вслух не называть само по себе — "
            "только если речь зашла о филиале, районе или адресе)."
        )
    lines.append(
        "Список филиалов и адреса в статику не входят — их подбирает контекстер "
        "по району или ориентиру от клиента."
    )
    return "\n".join(lines)


def format_branch_static(branch: Mapping[str, Any], *, branch_slug: str) -> str:
    """Собирает статику выбранного филиала.

    Args:
        branch: мета филиала.
        branch_slug: слаг филиала.

    Returns:
        Текстовый блок филиала.
    """
    lines = [f"Выбранный филиал (слаг {branch_slug}):"]
    for label, key in (
        ("Адрес", "address"),
        ("Ориентир", "landmark"),
        ("Тип", "place_type"),
        ("Статус", "status"),
        ("Часы", "working_hours"),
    ):
        value = branch.get(key)
        if value:
            lines.append(f"{label}: {value}.")
    return "\n".join(lines)


def merge_static(
    context: ConversationContext,
    *,
    city_slug: str | None = None,
    city_name: str | None = None,
    city_meta: Mapping[str, Any] | None = None,
    price_line: str | None = None,
    branch_slug: str | None = None,
    branch_meta: Mapping[str, Any] | None = None,
) -> ConversationContext:
    """Дописывает статику один раз: город, затем филиал.

    Args:
        context: текущий контекст.
        city_slug: слаг города.
        city_name: читаемое название.
        city_meta: мета города.
        price_line: готовая фраза о цене.
        branch_slug: слаг филиала.
        branch_meta: мета филиала.

    Returns:
        Обновлённый контекст; если статика уже заморожена целиком — без изменений.
    """
    updated = context.model_copy(deep=True)
    if city_slug and city_name and city_meta is not None and not updated.city_slug:
        updated.city_slug = city_slug
        updated.city_name = city_name
        updated.static_text = format_city_static(
            city_slug=city_slug,
            city_name=city_name,
            city_meta=city_meta,
            price_line=price_line,
        )
        updated.city_faq = normalize_city_faq(city_meta.get("faq"))
    if branch_slug and branch_meta is not None and updated.city_slug and not updated.branch_slug:
        updated.branch_slug = branch_slug
        branch_block = format_branch_static(branch_meta, branch_slug=branch_slug)
        updated.static_text = (updated.static_text + "\n\n" + branch_block).strip()
        updated.frozen = True
    return updated


class ContextState(BaseModel):
    """Сериализуемое представление контекста в состоянии треда."""

    static_text: str = ""
    dynamic_text: str = ""
    dynamic_status: str = DYN_NONE
    situation_slug: str | None = None
    filler_spoken: bool = False
    dynamic_reply: str = ""
    last_agent_reply: str = ""
    dynamic_turn: int = 0
    last_reply_hash: str = ""
    dynamic_reply_hash: str = ""
    pending_fields: list[str] = Field(default_factory=list)
    empty_needs: list[str] = Field(default_factory=list)
    city_slug: str | None = None
    city_name: str | None = None
    branch_slug: str | None = None
    branch_candidates: list[str] = Field(default_factory=list)
    nearby_text: str = ""
    nearby_key: str = ""
    city_faq: list[dict[str, str]] = Field(default_factory=list)
    conversation_ended: bool = False
    frozen: bool = False

    def to_context(self) -> ConversationContext:
        """Преобразует в рабочий объект контекста."""
        return ConversationContext.model_validate(self.model_dump())


def context_from_state(data: Mapping[str, Any] | None) -> ConversationContext:
    """Достаёт контекст из состояния треда.

    Args:
        data: словарь ``conversation_context`` или None.

    Returns:
        Контекст разговора.
    """
    if not data:
        return ConversationContext()
    return ConversationContext.model_validate(dict(data))


def _city_known(
    context: ConversationContext,
    profile: Mapping[str, str] | None,
) -> bool:
    """Город уже назван: в контексте или в форме разговора."""
    if (context.city_slug or "").strip() or (context.city_name or "").strip():
        return True
    if profile and str(profile.get("city") or "").strip():
        return True
    return False


def _has_city_static(context: ConversationContext) -> bool:
    """В контексте уже запечена статика города."""
    return bool((context.city_slug or "").strip() and (context.static_text or "").strip())


def _has_price_static(context: ConversationContext) -> bool:
    """В статике уже есть готовая фраза о цене."""
    return "Цена (готовая фраза" in (context.static_text or "")


def _has_branch_static(context: ConversationContext) -> bool:
    """В статике уже есть блок выбранного филиала."""
    return "Выбранный филиал" in (context.static_text or "")


def _has_branches_dynamic(context: ConversationContext) -> bool:
    """В динамике уже лежит список филиалов."""
    return "Филиалы" in (context.dynamic_text or "")


def missing_needs(
    context: ConversationContext,
    needs: Sequence[str],
    profile: Mapping[str, str] | None = None,
) -> list[str]:
    """Оставляет потребности, за которыми действительно надо идти.

    Потребность остаётся, если нужного нет в контексте и есть от чего
    плясать: городских данных не существует, пока не известен город.
    Потребности из ``empty_needs`` отбрасываются — справочник уже ответил
    пусто, повторный поход ничего не даст.

    Args:
        context: текущий контекст разговора.
        needs: потребности справочника (``needs_of``).
        profile: форма разговора; из неё берутся город и филиал.

    Returns:
        Недостающие потребности в порядке поступления.
    """
    city_known = _city_known(context, profile)
    branch_selected = bool((context.branch_slug or "").strip())
    if profile and str(profile.get("branch") or "").strip():
        branch_selected = True

    empty = {str(n).strip() for n in (context.empty_needs or []) if str(n).strip()}

    missing: list[str] = []
    for raw in needs:
        need = str(raw).strip()
        if not need:
            continue
        if need in empty:
            continue
        if need == "city_choices":
            if not city_known:
                missing.append(need)
        elif need == "city_meta":
            if city_known and not _has_city_static(context):
                missing.append(need)
        elif need == "price":
            if city_known and not _has_price_static(context):
                missing.append(need)
        elif need == "branches":
            if city_known and not branch_selected and not _has_branches_dynamic(context):
                missing.append(need)
        elif need == "branch_meta":
            if branch_selected and not _has_branch_static(context):
                missing.append(need)
        else:
            missing.append(need)
    return missing


def record_empty_needs(
    context: ConversationContext,
    needs: Sequence[str],
    *,
    found: bool,
) -> None:
    """Обновляет ``empty_needs`` по результату похода в справочник.

    Пустой ответ — потребности запоминаются: за ними больше не ходим.
    Данные получены — потребности из списка убираются, если были.

    Args:
        context: контекст разговора; список меняется на месте.
        needs: потребности этого похода.
        found: инструмент вернул данные.
    """
    cleaned = [str(n).strip() for n in needs if str(n).strip()]
    if not cleaned:
        return
    if found:
        drop = set(cleaned)
        context.empty_needs = [n for n in context.empty_needs if n not in drop]
        return
    known = set(context.empty_needs)
    for need in cleaned:
        if need not in known:
            context.empty_needs.append(need)
            known.add(need)
