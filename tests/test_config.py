"""Тесты настроек: пустые строки из ``.env`` не ломают необязательные поля."""

from __future__ import annotations

from core.config import Settings


def test_пустая_script_version_даёт_none():
    assert Settings(script_version="").script_version is None


def test_пробельная_строка_даёт_none():
    assert Settings(script_version="   ").script_version is None


def test_заполненное_значение_не_портится():
    assert Settings(script_version="1").script_version == "1"
    assert Settings(llm_base_url="https://api.example/v1").llm_base_url == "https://api.example/v1"


def test_дефолт_температуры_живой():
    """В коде дефолт 0.6; значение из окружения его перекрывает."""
    assert Settings.model_fields["llm_temperature"].default == 0.6
    assert Settings(llm_temperature=0.6).llm_temperature == 0.6


def test_пустые_необязательные_строки_дают_none():
    """По одному ассерту на каждое необязательное строковое поле."""
    s = Settings(
        llm_base_url="",
        llm_api_key="",
        llm_model_fast="",
        proxy_host="",
        proxy_port="",
        proxy_user="",
        proxy_pass="",
        proxy_scheme="",
        script_version="",
        script_dir="",
        langsmith_project="",
    )
    assert s.llm_base_url is None
    assert s.llm_api_key is None
    assert s.llm_model_fast is None
    assert s.proxy_host is None
    assert s.proxy_port is None
    assert s.proxy_user is None
    assert s.proxy_pass is None
    assert s.proxy_scheme is None
    assert s.script_version is None
    assert s.script_dir is None
    assert s.langsmith_project is None


def test_прокси_выключен_при_заполненных_полях():
    """``is_proxy=false`` — URL не собирается, даже если PROXY_* заданы."""
    s = Settings(
        is_proxy=False,
        proxy_host="127.0.0.1",
        proxy_port="1080",
        proxy_user="u",
        proxy_pass="p",
    )
    assert s.proxy_url is None


def test_прокси_не_собирается_из_пустых_строк():
    s = Settings(is_proxy=True, proxy_host="", proxy_port="", proxy_user="", proxy_pass="")
    assert s.proxy_host is None
    assert s.proxy_port is None
    assert s.proxy_url is None
