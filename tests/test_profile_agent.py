"""Тесты агента профиля: валидация без сети."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from graph.profile_agent import ProfileGuess, ProfileValue, guess_profile


class _FakeAgent:
    """Возвращает заданный разбор или падает."""

    def __init__(
        self,
        result: ProfileGuess | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result or ProfileGuess()
        self.error = error
        self.calls = 0
        self.history_seen: list = []
        self.reply_seen: str = ""

    async def guess(self, reply, known, fields, history=()):
        self.calls += 1
        self.reply_seen = reply
        self.history_seen = list(history)
        if self.error is not None:
            raise self.error
        return self.result


async def test_агент_получает_хвост_диалога():
    """Агент видит хвост истории, а не одну реплику."""
    agent = _FakeAgent(ProfileGuess(values=[ProfileValue(key="transmission", value="механика")]))
    history = [
        AIMessage(content="Механика или автомат?"),
        HumanMessage(content="механику"),
    ]
    result = await guess_profile(
        "механику",
        known={},
        fields=[("transmission", "Коробка")],
        history=history,
        agent=agent,
    )
    assert agent.history_seen == history
    assert [(v.key, v.value) for v in result.values] == [("transmission", "механика")]


async def test_отбрасывает_ключи_вне_перечня_и_пустые():
    agent = _FakeAgent(
        ProfileGuess(
            values=[
                ProfileValue(key="caller_name", value="Андрей"),
                ProfileValue(key="чужой", value="x"),
                ProfileValue(key="city", value=""),
                ProfileValue(key="city", value="   "),
            ]
        )
    )
    result = await guess_profile(
        "Меня зовут Андрей",
        known={},
        fields=[("caller_name", "Имя"), ("city", "Город")],
        history=[HumanMessage(content="Меня зовут Андрей")],
        agent=agent,
    )
    assert [(v.key, v.value) for v in result.values] == [("caller_name", "Андрей")]


async def test_заполненные_поля_не_перезаписываются():
    agent = _FakeAgent(ProfileGuess(values=[ProfileValue(key="caller_name", value="Пётр")]))
    result = await guess_profile(
        "Пётр",
        known={"caller_name": "Андрей"},
        fields=[("caller_name", "Имя")],
        agent=agent,
    )
    assert result.values == []


async def test_имя_прогоняется_через_given_name():
    agent = _FakeAgent(
        ProfileGuess(values=[ProfileValue(key="caller_name", value="Андрей Андреевич")])
    )
    result = await guess_profile(
        "Андрей Андреевич",
        known={},
        fields=[("caller_name", "Имя")],
        agent=agent,
    )
    assert result.values == [ProfileValue(key="caller_name", value="Андрей")]


async def test_сбой_агента_пустой_результат_без_исключения():
    agent = _FakeAgent(error=RuntimeError("модель недоступна"))
    result = await guess_profile(
        "вопрос",
        known={},
        fields=[("city", "Город")],
        agent=agent,
    )
    assert result == ProfileGuess()
