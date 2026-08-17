"""Тесты разбора реплик бота с ходов без реплики человека."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from graph.checker import CheckerVerdict, checker_system_prompt
from graph.checker_graph import (
    agent_replies_to_check,
    closable_by_agent_reply,
    live_check_node,
)
from graph.context import ConversationContext
from graph.context_agent import ContextDecision
from graph.context_store import MemoryContextStore
from graph.transcript import TranscriptEntry, append_agent, append_client
from script.store import ScriptProgress, progress_to_state

#: Реплика бота на ходе вытаскивания: договорённость о следующем шаге.
PULL_REPLY = "Тогда жду Вас завтра в три часа дня в офисе на Ленина, паспорт и СНИЛС с собой."
#: Реплика бота в ответ на человека — её судья уже разобрал по реплике человека.
ANSWER_REPLY = "Отлично, тогда оформим обучение."
#: Реплика бота на ходе без человека в сценарии продаж: вторая категория
#: предложена, знакомых пригласили. Ответа человека ни то, ни другое не ждёт.
SALES_PULL_REPLY = (
    "И на всякий случай: у нас можно параллельно учиться на вторую категорию. "
    "А если кто-то из знакомых соберётся — условия для них ровно те же."
)


@pytest.fixture(autouse=True)
def _offline_context(monkeypatch) -> MemoryContextStore:
    """Офлайн: кеш контекста в памяти, агенты модель не зовут."""
    from graph import contexter as contexter_module
    from graph import nodes as nodes_module
    from graph.farewell_agent import FarewellDecision
    from graph.profile_agent import ProfileGuess

    mem = MemoryContextStore()
    monkeypatch.setattr(nodes_module, "context_store", mem)
    monkeypatch.setattr(contexter_module, "context_store", mem)

    async def _no_need(*_a, **_k):
        return ContextDecision(need=False)

    async def _no_profile(*_a, **_k):
        return ProfileGuess()

    async def _no_farewell(*_a, **_k):
        return FarewellDecision(conversation_ended=False)

    monkeypatch.setattr("graph.contexter.decide_context", _no_need)
    monkeypatch.setattr("graph.checker_graph.guess_profile", _no_profile)
    monkeypatch.setattr("graph.checker_graph.decide_farewell", _no_farewell)
    return mem


class SpeakerChecker:
    """Судья-заглушка: закрывает заданные шаги, помня, чью реплику ему дали."""

    def __init__(self, close: dict[str, set[str]] | None = None) -> None:
        """Создаёт заглушку по карте «говорящий → шаги, которые он закрывает»."""
        self.close = close or {}
        self.calls: list[dict[str, Any]] = []

    async def judge(
        self,
        *,
        history_slice,
        client_reply,
        step,
        step_text,
        attempts: int = 0,
        age: int = 0,
        in_work: bool = False,
        speaker: str = "client",
    ) -> CheckerVerdict:
        """Возвращает вердикт и записывает вызов."""
        self.calls.append(
            {
                "history_slice": history_slice,
                "client_reply": client_reply,
                "step_id": step.id,
                "speaker": speaker,
            }
        )
        return CheckerVerdict(
            reply_usable=True,
            step_closed=step.id in self.close.get(speaker, set()),
        )

    def calls_of(self, speaker: str) -> list[dict[str, Any]]:
        """Вызовы по одному говорящему."""
        return [call for call in self.calls if call["speaker"] == speaker]


def _transcript() -> list[TranscriptEntry]:
    """История звонка: ответ человеку, а следом ход вытаскивания."""
    entries = append_agent([], turn=1, text="Добрый день! Как к Вам обращаться?")
    entries = append_client(entries, turn=2, text="Хочу записаться на обучение")
    entries = append_agent(entries, turn=2, text=ANSWER_REPLY)
    return append_agent(entries, turn=3, text=PULL_REPLY)


def _progress() -> ScriptProgress:
    """Прогресс: в работе вопрос про имя и действие «Закрытие»."""
    return ScriptProgress(
        status={"name": "pending", "closing": "pending", "tax_deduction": "closed"},
        attempts={"name": 2, "closing": 1},
        taken_turn={"name": 1, "closing": 3},
        in_work=["name", "closing"],
    )


def _sales_transcript() -> list[TranscriptEntry]:
    """История звонка продаж: ответ человеку, а следом ход без человека."""
    entries = append_agent([], turn=1, text="Добрый день! Как к Вам обращаться?")
    entries = append_client(entries, turn=2, text="Андрей, записывайте")
    entries = append_agent(entries, turn=2, text=ANSWER_REPLY)
    return append_agent(entries, turn=3, text=SALES_PULL_REPLY)


def _sales_progress() -> ScriptProgress:
    """Прогресс v4: в работе допродажа, приглашение знакомых и город."""
    return ScriptProgress(
        status={"upsell": "pending", "referral": "pending", "city": "pending"},
        attempts={"upsell": 1, "referral": 1, "city": 2},
        taken_turn={"upsell": 3, "referral": 3, "city": 1},
        in_work=["upsell", "referral", "city"],
    )


def _state(
    script,
    *,
    partial: str,
    progress: ScriptProgress,
    checked_entry: str = "",
    utterance_id: str = "u1",
) -> dict[str, Any]:
    """Состояние служебного прохода: город известен, «Закрытие» доступно."""
    state: dict[str, Any] = {
        "script_id": script.id,
        "script_version": script.version,
        "messages": [],
        "profile": {"city": "Пермь"},
        "turn": 4,
        "turn_kind": "client",
        "partial_reply": partial,
        "partial_utterance_id": utterance_id,
        "partial_is_final": False,
        "last_checked_partial": "",
        "last_checked_utterance_id": "",
        "last_checked_agent_entry": checked_entry,
    }
    state.update(progress_to_state(progress))
    return state


async def _run_live(script, state: dict[str, Any], progress: ScriptProgress, client) -> dict:
    """Гоняет служебный проход офлайн: без Redis, справочника и прогрева."""

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def fake_warmup(*_args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.farewell_min_messages = 5
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        return await live_check_node(state, runtime=None)  # type: ignore[arg-type]


async def test_реплика_бота_на_вытаскивании_закрывает_шаг(script, _offline_context):
    """Договорённость, произнесённая ботом, закрывает шаг «Закрытие»."""
    entries = _transcript()
    await _offline_context.save("local", ConversationContext(transcript=entries))
    progress = _progress()
    client = SpeakerChecker({"agent": {"closing"}})

    out = await _run_live(
        script,
        _state(script, partial="Ага, хорошо, договорились", progress=progress),
        progress,
        client,
    )

    assert out["step_status"]["closing"] == "closed"
    assert out["last_checked_agent_entry"] == entries[-1].entry_id
    agent_calls = client.calls_of("agent")
    assert [call["step_id"] for call in agent_calls] == ["closing"]
    assert agent_calls[0]["client_reply"] == PULL_REPLY
    # Разбираемая реплика уходит отдельным блоком, в срез истории не попадает.
    assert PULL_REPLY not in agent_calls[0]["history_slice"]
    assert ANSWER_REPLY in agent_calls[0]["history_slice"]


async def test_шаг_с_ответом_человека_репликой_бота_не_закрывается(script, _offline_context):
    """Судья не получает вопрос-шаг на реплике бота, и шаг остаётся открытым."""
    await _offline_context.save("local", ConversationContext(transcript=_transcript()))
    progress = _progress()
    # Заглушка закрывает всё, что ей дадут: отсекать должен код, а не вердикт.
    client = SpeakerChecker({"agent": {"name", "closing"}})

    out = await _run_live(
        script,
        _state(script, partial="Ага, хорошо, договорились", progress=progress),
        progress,
        client,
    )

    assert out["step_status"]["name"] == "pending"
    assert "name" not in [call["step_id"] for call in client.calls_of("agent")]
    assert out["step_status"]["closing"] == "closed"


async def test_реплика_человека_разбирается_как_прежде(script, _offline_context):
    """Реплика человека уходит судье целиком и закрывает свои шаги."""
    await _offline_context.save("local", ConversationContext(transcript=_transcript()))
    progress = _progress()
    partial = "Меня зовут Андрей Петрович"
    client = SpeakerChecker({"client": {"name"}})

    out = await _run_live(
        script, _state(script, partial=partial, progress=progress), progress, client
    )

    client_calls = client.calls_of("client")
    assert [call["step_id"] for call in client_calls] == ["name", "closing"]
    assert {call["client_reply"] for call in client_calls} == {partial}
    # Срез клиентского разбора — вся история звонка, включая реплики бота.
    assert PULL_REPLY in client_calls[0]["history_slice"]
    assert out["step_status"]["name"] == "closed"
    assert out["last_checked_partial"] == partial


async def test_реплика_бота_не_разбирается_дважды(script, _offline_context):
    """Отметка о разобранной реплике не пускает её к судье вторым проходом."""
    await _offline_context.save("local", ConversationContext(transcript=_transcript()))
    progress = _progress()
    first_client = SpeakerChecker()
    first = await _run_live(
        script,
        _state(script, partial="Ага, хорошо, договорились", progress=progress),
        progress,
        first_client,
    )
    assert first_client.calls_of("agent")

    second_client = SpeakerChecker()
    second = await _run_live(
        script,
        _state(
            script,
            partial="Ага, хорошо, договорились, буду завтра к трём",
            progress=progress,
            checked_entry=first["last_checked_agent_entry"],
            utterance_id="u2",
        ),
        progress,
        second_client,
    )

    assert second_client.calls_of("agent") == []
    assert second_client.calls_of("client")
    assert "last_checked_agent_entry" not in second


async def test_счётчики_и_взятие_в_работу_не_меняются(script, _offline_context):
    """Разбор реплики бота пишет только статус шага."""
    await _offline_context.save("local", ConversationContext(transcript=_transcript()))
    progress = _progress()
    attempts_before = dict(progress.attempts)
    taken_before = dict(progress.taken_turn)
    in_work_before = list(progress.in_work)

    out = await _run_live(
        script,
        _state(script, partial="Ага, хорошо, договорились", progress=progress),
        progress,
        SpeakerChecker({"agent": {"closing"}}),
    )

    assert out["step_attempts"] == attempts_before
    assert out["step_taken_turn"] == taken_before
    assert out["step_in_work"] == in_work_before
    assert progress.in_work == in_work_before


async def test_допродажа_и_приглашение_закрываются_репликой_бота(script_v4, _offline_context):
    """Бот проговорил своё, человек промолчал — шаги речи закрыты."""
    entries = _sales_transcript()
    await _offline_context.save("local", ConversationContext(transcript=entries))
    progress = _sales_progress()
    # Заглушка закрывает всё, что ей дадут: отсекать должен код, а не вердикт.
    client = SpeakerChecker({"agent": {"upsell", "referral", "city"}})

    out = await _run_live(
        script_v4,
        _state(script_v4, partial="Ага, понятно, спасибо", progress=progress),
        progress,
        client,
    )

    assert out["step_status"]["upsell"] == "closed"
    assert out["step_status"]["referral"] == "closed"
    assert out["last_checked_agent_entry"] == entries[-1].entry_id
    agent_calls = client.calls_of("agent")
    assert [call["step_id"] for call in agent_calls] == ["upsell", "referral"]
    assert agent_calls[0]["client_reply"] == SALES_PULL_REPLY


async def test_шаг_добычи_репликой_бота_не_закрывается(script_v4, _offline_context):
    """«Выявление города» судье с репликой бота не отдают вовсе."""
    await _offline_context.save("local", ConversationContext(transcript=_sales_transcript()))
    progress = _sales_progress()
    client = SpeakerChecker({"agent": {"upsell", "referral", "city"}})

    out = await _run_live(
        script_v4,
        _state(script_v4, partial="Ага, понятно, спасибо", progress=progress),
        progress,
        client,
    )

    assert out["step_status"]["city"] == "pending"
    assert "city" not in [call["step_id"] for call in client.calls_of("agent")]


async def test_разбор_реплики_человека_в_сценарии_продаж_не_изменился(script_v4, _offline_context):
    """Человеку по-прежнему отдают все шаги в работе, включая шаги добычи."""
    await _offline_context.save("local", ConversationContext(transcript=_sales_transcript()))
    progress = _sales_progress()
    partial = "Учиться буду в Перми"
    client = SpeakerChecker({"client": {"city"}})

    out = await _run_live(
        script_v4, _state(script_v4, partial=partial, progress=progress), progress, client
    )

    client_calls = client.calls_of("client")
    assert [call["step_id"] for call in client_calls] == ["city", "upsell", "referral"]
    assert {call["client_reply"] for call in client_calls} == {partial}
    assert out["step_status"]["city"] == "closed"
    assert out["last_checked_partial"] == partial


def test_на_разбор_идут_только_реплики_после_ответа_человеку():
    """Первая реплика хвоста — ответ человеку; на разбор идут следующие."""
    entries = _transcript()
    pending = agent_replies_to_check(entries, checked_entry_id="")
    assert [entry.text for entry in pending] == [PULL_REPLY]

    # Ещё один ход без человека — на разбор идут обе реплики.
    entries = append_agent(entries, turn=4, text="Если удобнее вечером, тоже подстроимся.")
    pending = agent_replies_to_check(entries, checked_entry_id="")
    assert [entry.text for entry in pending] == [PULL_REPLY, entries[-1].text]

    # Разобранное отсекается по отметке.
    pending = agent_replies_to_check(entries, checked_entry_id=entries[-2].entry_id)
    assert [entry.text for entry in pending] == [entries[-1].text]
    assert agent_replies_to_check(entries, checked_entry_id=entries[-1].entry_id) == []


def test_ответ_человеку_на_разбор_не_идёт():
    """Реплика бота сразу после фразы человека разбирается по реплике человека."""
    entries = append_client([], turn=1, text="Здравствуйте")
    entries = append_agent(entries, turn=1, text="Добрый день!")
    assert agent_replies_to_check(entries, checked_entry_id="") == []
    # Новая фраза человека обнуляет хвост.
    entries = append_client(entries, turn=2, text="Записывайте")
    entries = append_agent(entries, turn=2, text="Записываю.")
    assert agent_replies_to_check(entries, checked_entry_id="") == []


def test_шаги_с_ответом_человека_реплике_бота_недоступны(script, script_v4):
    """Вопрос и проверку закрывает только человек; действие — можно."""
    assert not closable_by_agent_reply(script.steps["name"])
    assert not closable_by_agent_reply(script.steps["terms"])
    assert closable_by_agent_reply(script.steps["closing"])
    assert not closable_by_agent_reply(None)
    # Шаг продаж, который добывает ответ, репликой бота тоже не закрыть:
    # «Выявление города» закрывает названный город, а не заданный вопрос.
    assert not closable_by_agent_reply(script_v4.steps["city"])


def test_шаги_речи_бота_закрываются_его_репликой(script_v4):
    """Требование «сказать, предложить, попросить» бот выполняет сам."""
    assert closable_by_agent_reply(script_v4.steps["upsell"])
    assert closable_by_agent_reply(script_v4.steps["referral"])


def test_род_шага_читается_из_требований_сценария(script_v4):
    """Раскладка всего сценария продаж на шаги речи и шаги добычи."""
    by_agent = {
        step_id
        for step_id in script_v4.step_order
        if closable_by_agent_reply(script_v4.steps[step_id])
    }

    assert by_agent == {
        "terms",
        "included",
        "practice",
        "price",
        "tax_deduction",
        "closing",
        "upsell",
        "referral",
    }
    # Оговорки в хвосте требований рода шага не меняют: «уточнить срок при
    # оформлении» у сроков и «если человек сам спросит» у состава курса.
    assert "уточнить" in script_v4.steps["terms"].requirements
    assert "спросит" in script_v4.steps["included"].requirements
    # Условие закрытия названо через человека — шаг ждёт человека, хотя
    # требование начинается с «Предложить».
    assert not closable_by_agent_reply(script_v4.steps["price_lock"])


def test_промпт_судьи_на_реплике_бота_запрещает_чужие_закрытия():
    """Судье сказано, чья реплика, и запрещено закрывать шаг за человека."""
    agent_prompt = checker_system_prompt(in_work=True, speaker="agent")
    assert "реплика агента" in agent_prompt
    assert "принадлежит самому агенту" in agent_prompt
    assert "нужен ответ, согласие, выбор или данные от клиента" in agent_prompt
    assert "asking_pointless=false всегда" in agent_prompt


def test_промпт_судьи_не_срывает_закрытие_вопросом_в_конце():
    """Шаг речи закрывает сама реплика: вопросительная концовка не мешает."""
    agent_prompt = checker_system_prompt(in_work=True, speaker="agent")
    assert "вопрос в её конце закрытию не мешает" in agent_prompt
    assert "Агент спросил, а клиент не ответил" not in agent_prompt
    assert "вопрос в её конце" not in checker_system_prompt(in_work=True)


def test_промпт_судьи_на_реплике_человека_не_изменился():
    """Ветка человека — прежний текст, без единой строки про агента."""
    for in_work in (True, False):
        client_prompt = checker_system_prompt(in_work=in_work)
        assert client_prompt == checker_system_prompt(in_work=in_work, speaker="client")
        assert "реплика клиента" in client_prompt
        assert "реплика агента" not in client_prompt
        assert "принадлежит самому агенту" not in client_prompt
        assert "asking_pointless=false всегда" not in client_prompt
