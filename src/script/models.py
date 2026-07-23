"""Сырая модель скрипта разговора.

Скрипт — это ДАННЫЕ. Здесь описано только то, как эти данные выглядят;
проверка связности и превращение в рабочий объект — в `build.py`.

Главное решение модели: **порядок шагов не задан списком, а выводится**.
У шага объявлено, что он заполняет, что требует заполненным до себя и какие
шаги должны быть закрыты раньше. Разброс из пяти реальных звонков (имя то
первым, то третьим, то не спросили вовсе; цена то до подбора группы, то после)
описывается одним скриптом — потому что менеджер идёт не по списку, а по тому,
что уже знает о клиенте.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: Вид шага.
#:
#: * `question` — спрашиваем, закрывается ответом клиента;
#: * `inform` — рассказываем, закрывается фактом произнесения;
#: * `inform_check` — рассказываем и проверяем («как Вам такой подход?»);
#: * `action` — целевое действие, закрывается результатом (слот встречи,
#:   отправленная анкета, удержанное место).
StepKind = Literal["question", "inform", "inform_check", "action"]

#: Что шагу нужно принести из справочника ДО вызова модели.
#:
#: Ходим всегда, когда положено по шагу, а не когда модель сочтёт нужным —
#: это и есть механизм, которым бот перестаёт выдумывать факты.
NeedKind = Literal["city_meta", "branches", "branch_meta", "price"]


class Persona(BaseModel):
    """Кого отыгрывает бот."""

    agent_name: str
    company: str
    role: str
    tone: str


class ProfileField(BaseModel):
    """Поле профиля, которое собирается за разговор.

    Звонящий не всегда учащийся: мама узнаёт про сына семнадцати лет — её имя
    и его имя это разные значения. Поэтому у поля есть роль, а профиль
    хранится по паре «роль + ключ», а не плоско.
    """

    key: str
    title: str
    role: Literal["caller", "student"] = "caller"
    #: Значение известно заранее и переспрашивать его — ошибка.
    prefilled: bool = False


class StepBranches(BaseModel):
    """Ветвление текста шага по значению поля профиля.

    Формат теории даёт три разные реплики, коробка передач — разный перечень
    машин. Это не разные шаги, а один шаг с разным содержанием.
    """

    field: str
    cases: dict[str, str]
    default: str | None = None


class SkipWhen(BaseModel):
    """Условие пропуска шага.

    Шаг должен уметь пропускаться по признаку, а не только закрываться:
    готовому клиенту презентация не нужна.
    """

    #: Пропустить, если все перечисленные поля профиля уже заполнены.
    filled: list[str] = Field(default_factory=list)
    #: Пропустить, если поле профиля равно одному из значений.
    equals: dict[str, list[str]] = Field(default_factory=dict)


class Step(BaseModel):
    """Один шаг скрипта."""

    id: str
    kind: StepKind
    #: Меньше — раньше. При равенстве порядок берётся из объявления.
    priority: int = 100
    #: Какие поля профиля закрывает шаг.
    fills: list[str] = Field(default_factory=list)
    #: Какие поля профиля обязаны быть заполнены до шага.
    requires: list[str] = Field(default_factory=list)
    #: Какие шаги обязаны быть закрыты до этого.
    after: list[str] = Field(default_factory=list)
    skip_when: SkipWhen | None = None
    #: Что принести из справочника перед вызовом модели.
    needs: list[NeedKind] = Field(default_factory=list)

    #: Задача шага человеческим языком — уходит в промпт.
    goal: str = ""
    #: Готовый текст (для `inform` и дословных блоков).
    text: str | None = None
    #: Ветки текста по значению поля профиля.
    branches: StepBranches | None = None
    #: Проверочный вопрос для `inform_check`.
    check_question: str | None = None
    #: Альтернативный вопрос как приём: «механика или автомат?».
    options: list[str] = Field(default_factory=list)
    #: Текст произносится дословно и моделью не переформулируется.
    verbatim: bool = False
    #: Сколько раз можно вернуться к шагу, прежде чем считать его отказом.
    max_attempts: int = 2


class Help(BaseModel):
    """Справка: медкомиссия, сдача в другом городе, запись на вождение.

    Справки не двигают скрипт и возвращают разговор на место. Опознаются по
    признакам срабатывания; когда вместо списка появится поиск, изменится
    только реализация подбора — форма записи останется прежней.
    """

    id: str
    triggers: list[str] = Field(default_factory=list)
    text: str


class Objection(BaseModel):
    """Возражение: «сравниваю со школами», «подумаю», «надо посоветоваться».

    Тоже возвращает на место, но меняет состояние: после «подумаю» включается
    срочность и переход в мессенджер.
    """

    id: str
    triggers: list[str] = Field(default_factory=list)
    text: str
    #: Что выставить в профиле при срабатывании.
    sets: dict[str, str] = Field(default_factory=dict)


class PriceTexts(BaseModel):
    """Три ветки разговора о цене.

    Ветка выбирается по полям ответа справочника, а не по списку городов и не
    по константам в коде. Когда заказчик заведёт настоящие цены и `reliable`
    станет `true`, агент заговорит третьей веткой без единой правки кода.
    """

    #: Суммы нет — цену не называем.
    no_amount: str
    #: Сумма есть, но не подтверждена — называем как примерную «от».
    unreliable: str
    #: Сумма подтверждена — называем как точную, без оговорки о примерности.
    reliable: str


class ScriptParams(BaseModel):
    """Параметры скрипта: тексты, которые не принадлежат конкретному шагу."""

    price: PriceTexts
    #: Фразы-заглушки в эфир, пока идём за данными. Лежат в данных, а не в коде,
    #: и берутся вперемешку.
    fillers: list[str] = Field(default_factory=list)
    #: Что говорить, когда ответа нет ни в скрипте, ни в справочнике.
    #: Выдумывать нельзя, молчать невозможно.
    unknown: str = ""
    #: Реплика, когда модель не ответила в бюджет хода.
    fallback: str = ""


class RawScript(BaseModel):
    """Скрипт разговора как он лежит в источнике."""

    id: str
    version: str
    persona: Persona
    opening_line: str = ""
    rules: list[str] = Field(default_factory=list)
    profile_fields: list[ProfileField] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    helps: list[Help] = Field(default_factory=list)
    objections: list[Objection] = Field(default_factory=list)
    params: ScriptParams
