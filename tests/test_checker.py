"""Тесты синхронного чекера."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from graph.checker import (
    CheckerVerdict,
    history_slice_for,
    run_checker,
)
from script.store import ScriptProgress


class FakeChecker:
    """Заглушка модели чекера."""

    def __init__(self, verdicts: list[CheckerVerdict | None]) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[dict] = []

    async def judge(self, *, history_slice, client_reply, step, step_text):
        self.calls.append(
            {
                "history_slice": history_slice,
                "client_reply": client_reply,
                "step_id": step.id,
                "step_text": step_text,
            }
        )
        if not self.verdicts:
            return None
        return self.verdicts.pop(0)


async def test_вход_разделён_история_и_реплика_не_склеены(script):
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    messages = [
        AIMessage(content="Как к вам обращаться?"),
        HumanMessage(content="Меня зовут Андрей"),
    ]
    progress = ScriptProgress(
        status={"name": "pending"}, attempts={"name": 1}, taken_turn={"name": 1}
    )
    await run_checker(
        script=script,
        progress=progress,
        messages=messages,
        profile={},
        turn=2,
        client=client,
    )
    assert client.calls
    call = client.calls[0]
    assert "Меня зовут Андрей" == call["client_reply"]
    assert "Меня зовут Андрей" not in call["history_slice"]
    assert "Как к вам обращаться" in call["history_slice"]


async def test_срез_от_взятия_самого_старого(script):
    steps = [script.step("name"), script.step("city")]
    progress = ScriptProgress(
        attempts={"name": 1, "city": 1},
        taken_turn={"name": 1, "city": 3},
    )
    messages = [
        HumanMessage(content="привет"),
        AIMessage(content="имя?"),
        HumanMessage(content="Андрей"),
        AIMessage(content="город?"),
        HumanMessage(content="Пермь"),
    ]
    sliced = history_slice_for(messages, steps=steps, progress=progress, turn=4)
    # Последняя реплика клиента отрезана.
    assert sliced[-1].type != "human" or sliced[-1].content != "Пермь"


async def test_для_счётчика_ноль_история_с_начала(script):
    steps = [script.step("city")]
    progress = ScriptProgress(attempts={"city": 0})
    messages = [
        HumanMessage(content="я из Перми"),
        AIMessage(content="как зовут?"),
        HumanMessage(content="Игорь"),
    ]
    sliced = history_slice_for(messages, steps=steps, progress=progress, turn=2)
    assert sliced[0].content == "я из Перми" or any("Перми" in str(m.content) for m in sliced[:-1])


async def test_реплика_не_годится_цикл_рвётся(script):
    client = FakeChecker([CheckerVerdict(reply_usable=False, step_closed=False)])
    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 1})
    updated = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="э-э")],
        profile={},
        turn=1,
        client=client,
    )
    assert updated.status.get("name") != "closed"
    assert len(client.calls) == 1


async def test_цикл_останавливается_на_первом_незакрывшемся(script):
    client = FakeChecker(
        [
            CheckerVerdict(reply_usable=True, step_closed=True),
            CheckerVerdict(reply_usable=True, step_closed=False),
            CheckerVerdict(reply_usable=True, step_closed=True),
        ]
    )
    progress = ScriptProgress(
        status={"name": "pending", "city": "pending"},
        attempts={"name": 1, "city": 1},
    )
    # who_studies ещё не доступен без закрытых предыдущих — только name и city
    # после закрытия name. Сделаем оба доступными: name closed by first verdict.
    updated = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="Андрей, Пермь")],
        profile={},
        turn=2,
        client=client,
    )
    assert updated.status.get("name") == "closed"
    assert updated.status.get("city") != "closed"
    assert len(client.calls) == 2


async def test_модель_не_ответила_шаги_не_тронуты(script):
    client = FakeChecker([None])
    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 1})
    updated = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="Андрей")],
        profile={},
        turn=1,
        client=client,
    )
    assert updated.status.get("name") == "pending"


async def test_порог_исчерпан_закрывает_без_модели(script):
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 2})
    updated = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="потом скажу")],
        profile={},
        turn=3,
        client=client,
        attempt_limit=2,
    )
    assert updated.status["name"] == "closed"
    assert not any(c["step_id"] == "name" for c in client.calls)
