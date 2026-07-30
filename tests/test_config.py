"""Тесты настроек: пустые строки из ``.env`` не ломают необязательные поля."""

from __future__ import annotations

from core.config import Settings


def test_персона_агента_дефолты():
    """Персона одна на агента; значения по умолчанию из кода."""
    assert Settings.model_fields["agent_name"].default == "Дарья"
    assert Settings.model_fields["agent_company"].default == "Вектор"
    assert Settings.model_fields["agent_role"].default == ("менеджер федеральной академии вождения")
    assert Settings.model_fields["agent_tone"].default == (
        "живой, доброжелательный, уважительный, без канцелярита"
    )
    assert Settings.model_fields["agent_gender"].default == "female"


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
        checker_graph_id="",
        checker_multitask_strategy="",
        agent_multitask_strategy="",
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
    assert s.checker_graph_id is None
    assert s.checker_multitask_strategy is None
    assert s.agent_multitask_strategy is None


def test_чекер_настройки_дефолты():
    """Порог прироста и стратегии запусков с запасными значениями."""
    s = Settings(
        checker_graph_id="",
        checker_multitask_strategy="",
        agent_multitask_strategy="",
    )
    assert Settings.model_fields["checker_min_growth_chars"].default == 10
    assert s.checker_assistant_id == "vector_checker"
    assert s.checker_run_strategy == "interrupt"
    assert s.agent_run_strategy == "enqueue"


def test_live_thread_suffix_дефолт():
    """Суффикс лайв-треда по умолчанию — ``-live``."""
    assert Settings.model_fields["live_thread_suffix"].default == "-live"


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


def test_filler_threshold_ms_отсутствует():
    assert "filler_threshold_ms" not in Settings.model_fields


def test_searching_stale_и_waiting_history_дефолты():
    """Порог зависшего поиска и лимит хвоста для реплики ожидания."""
    assert Settings.model_fields["searching_stale_turns"].default == 2
    assert Settings.model_fields["waiting_history_limit"].default == 4


def test_lookup_fillers_удалены():
    assert "lookup_fillers_enabled" not in Settings.model_fields
    assert "agent_fillers" not in Settings.model_fields
    assert "agent_city_fillers" not in Settings.model_fields
    assert "agent_branch_fillers" not in Settings.model_fields
    assert "agent_bridge_fillers" not in Settings.model_fields
    assert "bridge_first_delay" not in Settings.model_fields


def test_llm_max_tokens_short_есть():
    assert Settings.model_fields["llm_max_tokens_short"].default == 200


def test_pending_steps_soft_cap_дефолт_и_алиас():
    assert Settings.model_fields["pending_steps_soft_cap"].default == 4
    assert Settings(pending_steps_soft_cap=3).pending_steps_soft_cap == 3
