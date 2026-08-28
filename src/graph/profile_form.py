"""Форма профиля: перечень полей, которые бот заполняет за разговор.

Форма — рабочее состояние агента, а не текст заказчика, поэтому объявлена в коде
и от скрипта не зависит. Значения полей здесь не хранятся: они лежат в профиле
разговора, форма объявляет только сам перечень.

До появления этого модуля перечень выдёргивался регуляркой из текста требований
шагов: опечатка в ключе молча теряла поле, а лишнее латинское слово добавляло
мусорное. Теперь перечень объявлен явно.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProfileFormField(BaseModel):
    """Одно поле формы профиля.

    Attributes:
        key: ключ поля; под ним значение лежит в профиле разговора.
        title: человеческое название для перечня в промпте агента профиля.
        rewritable: поле разрешено уточнять. Если человек назвал другое
            значение, агент профиля возвращает новое и оно затирает старое.
            По умолчанию поля неперезаписываемы: имя и город случайным
            упоминанием затираться не должны.
    """

    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rewritable: bool = False


#: Пометка уточняемого поля в перечне, который видит агент профиля.
REWRITABLE_MARK = "уточняемое"

#: Форма разговора: девятнадцать полей в порядке появления по скрипту.
PROFILE_FORM: tuple[ProfileFormField, ...] = (
    ProfileFormField(key="caller_name", title="Имя звонящего"),
    ProfileFormField(key="city", title="Город обучения"),
    ProfileFormField(key="student_is_caller", title="Учится сам звонящий"),
    ProfileFormField(key="student_name", title="Имя будущего курсанта"),
    ProfileFormField(key="student_age", title="Возраст будущего курсанта"),
    ProfileFormField(
        key="location_hint",
        title=(
            "Адресный ориентир внутри города для подбора филиала: улица, "
            "станция метро или известное здание с названием — «Проспект "
            "Просвещения», «Коломяжский проспект 15». Город сюда не писать, "
            "он хранится отдельно; стороны света и общие слова без названия "
            "тоже не подходят"
        ),
        rewritable=True,
    ),
    ProfileFormField(key="experience", title="Опыт обучения"),
    ProfileFormField(key="transmission", title="Коробка передач"),
    ProfileFormField(key="theory_format", title="Формат теории"),
    ProfileFormField(key="branch", title="Выбранный филиал"),
    ProfileFormField(key="discount_category", title="Льготная категория"),
    ProfileFormField(key="tariff_choice", title="Выбранный тариф"),
    ProfileFormField(key="payment_pref", title="Как удобнее платить"),
    ProfileFormField(key="appointment_time", title="Время встречи"),
    ProfileFormField(key="outcome", title="Итог разговора"),
    ProfileFormField(key="second_category", title="Интерес ко второй категории"),
    ProfileFormField(key="messenger", title="Мессенджер для документов"),
    ProfileFormField(
        key="caller_phone",
        title=(
            "Номер для переписки, если он не тот, с которого звонят: только "
            "сам номер цифрами — «+7 921 555-01-23». Слова про номер "
            "(«тот же», «на этот», «как для связи») сюда не писать"
        ),
    ),
    ProfileFormField(key="urgency", title="Срочность и сравнение с другими школами"),
)


def _check_unique(form: tuple[ProfileFormField, ...]) -> None:
    """Роняет импорт, если в форме задублирован ключ.

    Args:
        form: перечень полей формы.

    Raises:
        ValueError: если хотя бы один ключ встречается дважды.
    """
    keys = [field.key for field in form]
    doubles = sorted({key for key in keys if keys.count(key) > 1})
    if doubles:
        raise ValueError(f"Дубли ключей в форме профиля: {', '.join(doubles)}")


_check_unique(PROFILE_FORM)


def form_keys() -> frozenset[str]:
    """Все ключи формы.

    Returns:
        Множество ключей полей.
    """
    return frozenset(field.key for field in PROFILE_FORM)


def rewritable_keys() -> frozenset[str]:
    """Ключи полей, которые разрешено уточнять.

    Returns:
        Множество ключей с признаком ``rewritable``.
    """
    return frozenset(field.key for field in PROFILE_FORM if field.rewritable)


def field_pairs() -> list[tuple[str, str]]:
    """Перечень для агента профиля: ключ и название.

    Уточняемые поля помечены в названии, чтобы агент понимал, где новое
    значение допустимо.

    Returns:
        Список пар ``(key, title)`` в порядке объявления формы.
    """
    pairs: list[tuple[str, str]] = []
    for field in PROFILE_FORM:
        title = f"{field.title} ({REWRITABLE_MARK})" if field.rewritable else field.title
        pairs.append((field.key, title))
    return pairs
