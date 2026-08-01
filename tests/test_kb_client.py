"""Тесты клиента справочника: диагностика HTTP-запросов."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from kb.client import API_PREFIX, LOG_PREFIX, PATH_CITIES, REQUEST_RETRIES, VectorKBClient


@pytest.fixture
def kb_client() -> VectorKBClient:
    """Клиент без живого соединения — HTTP подменяется в тестах."""
    return VectorKBClient(base_url="http://kb.test", timeout=0.5, cache_ttl=0)


def _fake_response(status_code: int, content: bytes) -> httpx.Response:
    """Собирает ответ httpx без сети."""
    return httpx.Response(
        status_code=status_code,
        content=content,
        request=httpx.Request("GET", f"http://kb.test/api{PATH_CITIES}"),
    )


async def test_http_клиент_не_доверяет_окружению(kb_client: VectorKBClient) -> None:
    """Справочник локальный: прокси из переменных окружения не подхватывается."""
    http = await kb_client._init_client()
    try:
        assert http.trust_env is False
    finally:
        await kb_client.close()


async def test_базовый_адрес_и_префикс_прежние(kb_client: VectorKBClient) -> None:
    """Адрес сервиса и префикс API не менялись при отключении прокси из env."""
    assert API_PREFIX == "/api"
    assert kb_client._base_url == "http://kb.test"
    http = await kb_client._init_client()
    try:
        assert str(http.base_url).rstrip("/") == f"http://kb.test{API_PREFIX}"
    finally:
        await kb_client.close()


async def test_запрос_пишет_путь_код_и_размер_тела(
    kb_client: VectorKBClient, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Каждый ответ справочника логируется: путь, код и размер тела."""
    payload = [{"slug": "perm", "name": "Perm"}]
    body = b'[{"slug":"perm","name":"Perm"}]'
    http = AsyncMock()
    http.request = AsyncMock(return_value=_fake_response(200, body))
    monkeypatch.setattr(kb_client, "_init_client", AsyncMock(return_value=http))

    with caplog.at_level(logging.INFO, logger="kb.client"):
        result = await kb_client.list_cities()

    assert result == payload
    info_msgs = [
        r.message
        for r in caplog.records
        if r.name == "kb.client" and r.levelno == logging.INFO and LOG_PREFIX in r.message
    ]
    assert any(
        PATH_CITIES in msg and "200" in msg and f"{len(body)} байт" in msg for msg in info_msgs
    )


async def test_ошибка_пишет_причину(
    kb_client: VectorKBClient, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """При HTTP-ошибке или таймауте в лог уходит отдельная строка с причиной."""
    http = AsyncMock()
    http.request = AsyncMock(side_effect=httpx.ConnectTimeout("timed out"))
    monkeypatch.setattr(kb_client, "_init_client", AsyncMock(return_value=http))
    monkeypatch.setattr("kb.client.asyncio.sleep", AsyncMock())

    with caplog.at_level(logging.INFO, logger="kb.client"):
        result = await kb_client.list_cities()

    assert result == []
    error_msgs = [
        r.message
        for r in caplog.records
        if r.name == "kb.client"
        and r.levelno == logging.INFO
        and "Ошибка запроса" in r.message
        and PATH_CITIES in r.message
    ]
    assert error_msgs
    assert any("timed out" in msg or "ConnectTimeout" in msg for msg in error_msgs)
    assert len(error_msgs) >= REQUEST_RETRIES


async def test_ответ_с_ошибкой_статуса_логирует_код_и_причину(
    kb_client: VectorKBClient, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ответ 5xx: в логе есть путь с кодом и размером, плюс строка с причиной."""
    body = b'{"detail":"boom"}'
    http = AsyncMock()
    http.request = AsyncMock(return_value=_fake_response(503, body))
    monkeypatch.setattr(kb_client, "_init_client", AsyncMock(return_value=http))

    with caplog.at_level(logging.INFO, logger="kb.client"):
        result: Any = await kb_client._request("GET", PATH_CITIES)

    assert result is None
    info_msgs = [r.message for r in caplog.records if r.name == "kb.client"]
    assert any(
        PATH_CITIES in msg and "503" in msg and f"{len(body)} байт" in msg for msg in info_msgs
    )
    assert any("Ошибка запроса" in msg and "503" in msg for msg in info_msgs)
