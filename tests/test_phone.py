"""Номер в анкету попадает только цифрами.

Разбор живого звонка: человек сказал «мне удобнее в Телеграме на том же
номере, что и телефон для связи», и агент профиля записал эту фразу в
поле ``caller_phone``. Поле структурное — из него звонят и пишут, — и
текст вместо цифр там хуже пустоты: пустое поле видно, а фраза выглядит
заполненной.
"""

from __future__ import annotations

import pytest

from graph.phone import phone_number
from graph.profile_agent import ProfileGuess, ProfileValue, guess_profile


class FakeAgent:
    """Агент профиля, отдающий заранее заданное."""

    def __init__(self, values: list[tuple[str, str]]) -> None:
        self._values = values

    async def guess(self, *_args, **_kwargs) -> ProfileGuess:
        """Возвращает заготовленные поля."""
        return ProfileGuess(values=[ProfileValue(key=k, value=v) for k, v in self._values])


@pytest.mark.parametrize(
    "spoken",
    [
        "+7 921 555-01-23",
        "8 921 555 01 23",
        "9215550123",
        "запишите 8-921-555-01-23, я на связи",
    ],
)
def test_номер_принимается(spoken: str) -> None:
    """Всё, где набирается номер, проходит как есть."""
    assert phone_number(spoken) == spoken.strip()


@pytest.mark.parametrize(
    "spoken",
    [
        "тот же номер, что и телефон для связи",
        "на этот же",
        "как для связи",
        "",
        "   ",
        "мне 29, свободен после семи",
        "пять через два",
    ],
)
def test_не_номер_отбрасывается(spoken: str) -> None:
    """Слова про номер и любые другие числа номером не считаются."""
    assert phone_number(spoken) == ""


def test_слишком_длинная_цепочка_цифр_не_номер() -> None:
    """Слипшиеся числа длиннее международного номера — не номер."""
    assert phone_number("1234567890123456") == ""


async def test_агент_профиля_отбрасывает_фразу_вместо_номера() -> None:
    """Фраза в поле номера до анкеты не доходит, остальные поля доходят."""
    guess = await guess_profile(
        "мне удобнее в Телеграме на том же номере, что и телефон для связи",
        known={},
        fields=[("caller_phone", "Номер"), ("messenger", "Мессенджер")],
        agent=FakeAgent(
            [
                ("caller_phone", "тот же номер, что и телефон для связи"),
                ("messenger", "Телеграм"),
            ]
        ),
    )

    assert [(v.key, v.value) for v in guess.values] == [("messenger", "Телеграм")]


async def test_агент_профиля_пропускает_настоящий_номер() -> None:
    """Продиктованный номер сохраняется словами клиента."""
    guess = await guess_profile(
        "пишите на +7 921 555-01-23",
        known={},
        fields=[("caller_phone", "Номер")],
        agent=FakeAgent([("caller_phone", "+7 921 555-01-23")]),
    )

    assert [(v.key, v.value) for v in guess.values] == [("caller_phone", "+7 921 555-01-23")]
