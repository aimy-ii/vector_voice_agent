"""Тесты клиента справочника: диагностика HTTP-запросов."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from core.config import settings
from kb.client import (
    API_PREFIX,
    CACHE_MAX_CITIES,
    LOG_PREFIX,
    PATH_BRANCHES_NEAREST,
    PATH_CITIES,
    PATH_GEOCODE,
    REQUEST_RETRIES,
    VectorKBClient,
)


@pytest.fixture
def kb_client() -> VectorKBClient:
    """Клиент без живого соединения — HTTP подменяется в тестах."""
    return VectorKBClient(base_url="http://kb.test", timeout=0.5, cache_ttl=0)


@pytest.fixture
def kb_client_cached() -> VectorKBClient:
    """Клиент с живым кэшем — для проверок повторных походов."""
    return VectorKBClient(base_url="http://kb.test", timeout=0.5, cache_ttl=60)


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


async def test_geocode_found_возвращает_координаты(
    kb_client: VectorKBClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """При found=true возвращается пара координат; в params — текст и город."""
    body = b'{"found":true,"lat":59.96,"lon":30.33}'
    http = AsyncMock()
    http.request = AsyncMock(return_value=_fake_response(200, body))
    monkeypatch.setattr(kb_client, "_init_client", AsyncMock(return_value=http))

    result = await kb_client.geocode("Солнечный", city_slug="spb")

    assert result == (59.96, 30.33)
    kwargs = http.request.await_args.kwargs
    assert http.request.await_args.args[:2] == ("GET", PATH_GEOCODE)
    assert kwargs["params"] == {"text": "Солнечный", "city_slug": "spb"}


async def test_geocode_кэширует_нормализованное_место(
    kb_client_cached: VectorKBClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Повтор с другим регистром и «ё» по сети не ходит."""
    body = b'{"found":true,"lat":59.96,"lon":30.33}'
    http = AsyncMock()
    http.request = AsyncMock(return_value=_fake_response(200, body))
    monkeypatch.setattr(kb_client_cached, "_init_client", AsyncMock(return_value=http))

    first = await kb_client_cached.geocode("Зелёный", city_slug="spb")
    second = await kb_client_cached.geocode("  зеленый  ", city_slug="spb")

    assert first == (59.96, 30.33)
    assert second == (59.96, 30.33)
    assert http.request.await_count == 1


async def test_geocode_другой_город_не_из_кэша(
    kb_client_cached: VectorKBClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """То же место в другом городе — отдельный ключ кэша и второй поход."""
    body = b'{"found":true,"lat":59.96,"lon":30.33}'
    http = AsyncMock()
    http.request = AsyncMock(return_value=_fake_response(200, body))
    monkeypatch.setattr(kb_client_cached, "_init_client", AsyncMock(return_value=http))

    await kb_client_cached.geocode("Центральный", city_slug="perm")
    await kb_client_cached.geocode("Центральный", city_slug="krasnoyarsk")

    assert http.request.await_count == 2


async def test_geocode_кэширует_отказ(
    kb_client_cached: VectorKBClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """found=false даёт None и кэшируется: повтор по сети не ходит."""
    body = b'{"found":false}'
    http = AsyncMock()
    http.request = AsyncMock(return_value=_fake_response(200, body))
    monkeypatch.setattr(kb_client_cached, "_init_client", AsyncMock(return_value=http))

    first = await kb_client_cached.geocode("Нигде", city_slug="spb")
    second = await kb_client_cached.geocode("нигде", city_slug="spb")

    assert first is None
    assert second is None
    assert http.request.await_count == 1


async def test_geocode_короткий_таймаут_и_одна_попытка(
    kb_client: VectorKBClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Геокодер уходит с коротким таймаутом и retries=1."""
    captured: dict[str, Any] = {}

    async def fake_request(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> dict[str, Any]:
        captured.update(
            {
                "method": method,
                "path": path,
                "params": params,
                "timeout": timeout,
                "retries": retries,
            }
        )
        return {"found": True, "lat": 1.0, "lon": 2.0}

    monkeypatch.setattr(kb_client, "_request", fake_request)

    result = await kb_client.geocode("Солнечный", city_slug="spb")

    assert result == (1.0, 2.0)
    assert captured["path"] == PATH_GEOCODE
    assert captured["timeout"] == settings.geocode_timeout_seconds
    assert captured["retries"] == 1


async def test_geocode_короткий_текст_без_запроса(
    kb_client: VectorKBClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Текст короче двух символов — None без сетевого похода."""
    http = AsyncMock()
    http.request = AsyncMock()
    monkeypatch.setattr(kb_client, "_init_client", AsyncMock(return_value=http))

    assert await kb_client.geocode("а") is None
    assert await kb_client.geocode("") is None
    assert http.request.await_count == 0


async def test_nearest_branches_параметры_и_пустой_не_кэшируется(
    kb_client_cached: VectorKBClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Передаются lat/lon/limit/radius/city; пустой ответ не кэшируется."""
    nonempty = '[{"slug":"a","address":"ул. А","distance_km":0.4}]'.encode()
    empty = b"[]"
    http = AsyncMock()
    http.request = AsyncMock(
        side_effect=[
            _fake_response(200, nonempty),
            _fake_response(200, empty),
            _fake_response(200, empty),
        ]
    )
    monkeypatch.setattr(kb_client_cached, "_init_client", AsyncMock(return_value=http))

    first = await kb_client_cached.nearest_branches(59.96, 30.33, city_slug="spb")
    assert first == [{"slug": "a", "address": "ул. А", "distance_km": 0.4}]
    kwargs = http.request.await_args.kwargs
    assert http.request.await_args.args[:2] == ("GET", PATH_BRANCHES_NEAREST)
    assert kwargs["params"] == {
        "lat": 59.96,
        "lon": 30.33,
        "limit": settings.nearest_branches_limit,
        "radius_km": settings.nearest_branches_radius_km,
        "city_slug": "spb",
    }

    # Другая точка — пустой ответ дважды: пустое не кэшируется.
    empty_first = await kb_client_cached.nearest_branches(10.0, 20.0, city_slug="spb")
    empty_second = await kb_client_cached.nearest_branches(10.0, 20.0, city_slug="spb")
    assert empty_first == []
    assert empty_second == []
    assert http.request.await_count == 3


def test_cache_max_cities_равен_512() -> None:
    """Лимит общего кэша увеличен до 512."""
    assert CACHE_MAX_CITIES == 512
