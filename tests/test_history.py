"""Тесты разбора истории звонка — без звука и без провайдеров."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from graph.history import (
    find_aside,
    is_acknowledgement,
    is_first_turn,
    last_agent_text,
    last_user_text,
    matches_triggers,
    normalize,
    strip_system,
    text_of,
)
from graph.state import replace_messages


def test_системные_сообщения_бота_отбрасываются():
    """Бот кладёт свои инструкции в историю; свой промпт граф собирает сам."""
    messages = [
        SystemMessage(content="Ты менеджер автошколы", id="lk.agent_task.instructions"),
        HumanMessage(content="Здравствуйте"),
        AIMessage(content="Добрый день"),
    ]
    cleaned = strip_system(messages)
    assert len(cleaned) == 2
    assert not any(m.type in ("system", "developer") for m in cleaned)


def test_история_словарями_через_редьюсер():
    """Сервер шлёт JSON; редьюсер приводит, history работает на объектах."""
    converted = replace_messages(
        [],
        [
            {
                "role": "system",
                "content": "инструкции бота",
                "id": "lk.agent_task.instructions",
            },
            {"role": "human", "content": "Здравствуйте"},
            {"role": "ai", "content": "Добрый день"},
        ],
    )
    assert all(isinstance(m, BaseMessage) for m in converted)
    cleaned = strip_system(converted)
    assert len(cleaned) == 2
    assert isinstance(cleaned[0], HumanMessage)
    assert isinstance(cleaned[1], AIMessage)
    assert last_user_text(cleaned) == "Здравствуйте"
    assert last_agent_text(cleaned) == "Добрый день"


def test_последние_реплики_сторон():
    messages = [
        HumanMessage(content="Хочу на механику"),
        AIMessage(content="Отлично"),
        HumanMessage(content="  А сколько стоит?  "),
    ]
    assert last_user_text(messages) == "А сколько стоит?"
    assert last_agent_text(messages) == "Отлично"


def test_пустая_история_не_роняет_разбор():
    assert last_user_text([]) == ""
    assert last_agent_text([]) == ""
    assert is_first_turn([]) is True


def test_первый_ход_реактивный():
    """Во всех входящих первым говорит клиент, и говорит по делу."""
    messages = [HumanMessage(content="Хочу записаться на механику, когда старт?")]
    assert is_first_turn(messages) is True
    messages.append(AIMessage(content="Сориентирую"))
    assert is_first_turn(messages) is False


def test_текст_сообщения_из_списка_кусков():
    assert text_of(HumanMessage(content=["Да", "конечно"])) == "Да конечно"
    assert text_of(HumanMessage(content="  да  ")) == "да"


def test_нормализация_убирает_знаки_и_ё():
    assert normalize("Всё, понятно!") == "все понятно"
    assert normalize("  ДА...  ") == "да"


def test_короткое_подтверждение_опознаётся():
    for text in ("да", "Ага", "хорошо, давайте", "понятно", "Да, конечно"):
        assert is_acknowledgement(text) is True, text


def test_содержательная_реплика_не_подтверждение():
    for text in (
        "да, а сколько стоит обучение",
        "механика",
        "я подумаю ещё",
        "",
        "да нет наверное не знаю пока",
    ):
        assert is_acknowledgement(text) is False, text


def test_признаки_справок_срабатывают(script):
    assert matches_triggers("А медкомиссию когда проходить?", script.helps["medcheck"].triggers)
    assert matches_triggers("Что по цене", script.helps["medcheck"].triggers) is False


def test_справка_и_возражение_ищутся_по_каталогу(script):
    helps = {k: v.triggers for k, v in script.helps.items()}
    objections = {k: v.triggers for k, v in script.objections.items()}

    assert find_aside("а когда практика начинается?", helps) == "practice_start"
    assert find_aside("я ещё подумаю", objections) == "think"
    assert find_aside("механика", helps) is None


def test_смешанная_реплика_даёт_и_ответ_и_вопрос(script):
    """«Механика. А сколько стоит?» — ответ на шаг и посторонний вопрос сразу."""
    text = "Механика. А сопровождение в ГИБДД сколько?"
    helps = {k: v.triggers for k, v in script.helps.items()}
    assert find_aside(text, helps) == "gibdd_support"
    assert is_acknowledgement(text) is False


def test_has_something_to_answer(script):
    """Таблица: когда генератору есть / нечего ответить."""
    from graph.history import has_something_to_answer

    for text in (
        "",
        "   ",
        "да, конечно",
        "механика",
        "в Санкт-Петербурге",
        "хотел бы обучаться на механике",
        "сам собираюсь учиться",
        "да просто хочу на механике",
        "мне всё как раз подходит",
        "никак не решу",
    ):
        assert has_something_to_answer(text, script=script) is False, text
    for text in (
        "а сколько стоит?",
        "когда практика",
        "дорого",
        "подумаю",
        "а медкомиссия нужна?",
        "расскажите про автомат",
        "а-а, хотел бы обучаться на механике. Расскажите про автомат",
        "подскажите, а автомат сложнее?",
        "объясните про рассрочку",
        "а насчёт медкомиссии",
        "повторите",
        "не понял",
        "ещё раз",
    ):
        assert has_something_to_answer(text, script=script) is True, text


def test_is_repeat_request():
    from graph.history import is_repeat_request

    for text in ("повторите", "не понял", "ещё раз", "Перефразируйте пожалуйста"):
        assert is_repeat_request(text) is True, text
    for text in ("да просто хочу на механике", "расскажите про автомат", "механика"):
        assert is_repeat_request(text) is False, text
