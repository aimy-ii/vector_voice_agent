"""Тесты речевого фона: связки без сети и без реального ожидания."""

from __future__ import annotations

import asyncio

import pytest

from graph import speech_bg


@pytest.fixture(autouse=True)
def _reset_bg() -> None:
    """Сбрасывает фоновые задачи между тестами."""
    for call_id in list(speech_bg._tasks):
        speech_bg.stop_background_speech(call_id)
    speech_bg._tasks.clear()
    speech_bg._spoken.clear()


@pytest.mark.asyncio
async def test_связки_уходят_через_on_speak_не_больше_лимита(monkeypatch):
    monkeypatch.setattr(speech_bg.settings, "bridge_first_delay", 0.0)
    monkeypatch.setattr(speech_bg.settings, "bridge_next_delay", 0.0)
    monkeypatch.setattr(speech_bg.settings, "bridge_filler_limit", 2)
    monkeypatch.setattr(
        speech_bg.settings,
        "agent_bridge_fillers",
        ["Так, сейчас посмотрю.", "Секунду, уточняю.", "Ага, вижу."],
    )
    monkeypatch.setattr("graph.fillers.random.choice", lambda pool: pool[0])
    heard: list[str] = []

    await speech_bg.start_background_speech(
        "c1",
        templates=speech_bg.settings.agent_bridge_fillers,
        used=[],
        on_speak=heard.append,
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    phrases = speech_bg.stop_background_speech("c1")
    assert len(heard) <= 2
    assert len(phrases) == len(heard)
    assert phrases == heard
    assert all(p.endswith(".") for p in phrases)


@pytest.mark.asyncio
async def test_stop_возвращает_прозвучавшие_и_идемпотентен(monkeypatch):
    monkeypatch.setattr(speech_bg.settings, "bridge_first_delay", 0.0)
    monkeypatch.setattr(speech_bg.settings, "bridge_next_delay", 0.0)
    monkeypatch.setattr(speech_bg.settings, "bridge_filler_limit", 1)
    monkeypatch.setattr(
        speech_bg.settings,
        "agent_bridge_fillers",
        ["Секунду, уточняю."],
    )
    monkeypatch.setattr("graph.fillers.random.choice", lambda pool: pool[0])
    heard: list[str] = []
    await speech_bg.start_background_speech(
        "c2",
        templates=["Секунду, уточняю."],
        used=[],
        on_speak=heard.append,
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    first = speech_bg.stop_background_speech("c2")
    second = speech_bg.stop_background_speech("c2")
    assert first == heard
    assert second == []


@pytest.mark.asyncio
async def test_ошибка_внутри_задачи_наружу_не_летит(monkeypatch):
    monkeypatch.setattr(speech_bg.settings, "bridge_first_delay", 0.0)
    monkeypatch.setattr(speech_bg.settings, "bridge_filler_limit", 1)

    def _boom(_phrase: str) -> None:
        raise RuntimeError("сбой эфира")

    monkeypatch.setattr(
        "graph.speech_bg.bridge_filler",
        lambda templates, used=None: "Ага, вижу.",
    )
    await speech_bg.start_background_speech(
        "c3",
        templates=["Ага, вижу."],
        used=[],
        on_speak=_boom,
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # Задача не должна уронить тест необработанным исключением.
    assert speech_bg.stop_background_speech("c3") == []
