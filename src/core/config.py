"""Настройки агентского сервиса.

Всё, что меняется между стендами, живёт здесь и читается из окружения:
провайдер и модель LLM, прокси, адрес справочника, бюджет хода и порог
заглушки в эфир. Правки кода для смены провайдера не требуются — все
провайдеры OpenAI-совместимые.

Секреты в код не попадают: в репозитории лежат только имена переменных
(см. `.env.example`).
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из окружения и `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_title: str = "vector_voice_agent"

    # ─── LLM (любой OpenAI-совместимый провайдер) ───────────────────────────
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    #: Основная модель хода. Совпадает с тем, что стоит в боте.
    llm_model: str = "gpt-4.1-mini"
    #: Быстрая модель для коротких служебных вызовов. Пусто → берётся основная.
    llm_model_fast: str | None = None
    llm_temperature: float = 0.6
    #: Потолок одновременных вызовов модели на процесс.
    llm_max_concurrency: int = 8
    #: Потолок токенов ответа. Реплика в звонке короткая, длинный ответ — это
    #: молчание в трубке, поэтому лимит намеренно жёсткий.
    llm_max_tokens: int = 700

    # ─── Бюджет хода ────────────────────────────────────────────────────────
    #: Сколько секунд ждём соединения с провайдером модели.
    llm_connect_timeout: float = 5.0
    #: Сколько секунд ждём ответ модели. Промах — деградация в заглушку,
    #: а не бесконечное ожидание: в звонке пауза дороже плохого ответа.
    llm_read_timeout: float = 8.0
    #: Сколько всего секунд отводим на ход. По исчерпании узел отдаёт
    #: аварийную реплику из данных скрипта.
    turn_budget_seconds: float = 9.0

    # ─── Прокси (включается флагом, как в акселераторе) ──────────────────────
    is_proxy: bool = False
    proxy_host: str | None = None
    proxy_port: str | None = None
    proxy_user: str | None = None
    proxy_pass: str | None = None
    #: `http` / `socks5h`. Пусто — схема выбирается по номеру порта.
    proxy_scheme: str | None = None

    # ─── Справочник Vector KB ───────────────────────────────────────────────
    #: Общей docker-сети между стеками нет: справочник опубликован на
    #: `127.0.0.1:8317` хоста, поэтому из контейнера графа ходим через
    #: `host.docker.internal` (в compose добавлен `extra_hosts`).
    vector_kb_url: str = "http://host.docker.internal:8317"
    vector_kb_timeout: float = 2.0
    vector_kb_cache_ttl: int = 3600

    # ─── Скрипт разговора ───────────────────────────────────────────────────
    #: Идентификатор скрипта по умолчанию: с ним стартует звонок, если
    #: конкретный не пришёл в context.
    script_id: str = "vector_ru"
    #: Версия скрипта по умолчанию. Пусто — берётся последняя из источника.
    script_version: str | None = None
    #: Каталог с JSON-скриптами. Пусто — каталог `data` рядом с кодом.
    script_dir: str | None = None

    # ─── Заглушка в эфир ────────────────────────────────────────────────────
    #: Порог в миллисекундах: если ожидаемая пауза хода больше — перед
    #: походом за данными выталкиваем фразу-заглушку из данных скрипта.
    #: 0 — механизм выключен. Включается настройкой по замерам на пилоте,
    #: узел при этом не меняется.
    filler_threshold_ms: int = 0

    # ─── Наблюдаемость ──────────────────────────────────────────────────────
    log_level: str = "INFO"
    langsmith_project: str | None = None

    @field_validator(
        "llm_base_url",
        "llm_api_key",
        "llm_model_fast",
        "proxy_host",
        "proxy_port",
        "proxy_user",
        "proxy_pass",
        "proxy_scheme",
        "script_version",
        "script_dir",
        "langsmith_project",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        """Пустая или пробельная строка из ``.env`` → ``None``.

        ``SCRIPT_VERSION=`` даёт ``""``, а не ``None``; без приведения
        источник ищет версию ``''`` и падает, хотя пустое значение значит
        «взять последнюю». То же для остальных необязательных строковых полей.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def fast_model(self) -> str:
        """Модель для коротких вызовов; отдельной нет — берём основную."""
        return self.llm_model_fast or self.llm_model

    @property
    def script_data_dir(self) -> Path:
        """Каталог с JSON-скриптами разговора."""
        if self.script_dir:
            return Path(self.script_dir)
        return Path(__file__).resolve().parent.parent / "script" / "data"

    @property
    def proxy_url(self) -> str | None:
        """Формирует URL прокси для похода в LLM или None, если прокси выключен."""
        if not self.is_proxy or not self.proxy_host or not self.proxy_port:
            return None
        scheme = self.proxy_scheme
        if not scheme:
            scheme = "socks5h" if self.proxy_port in ("1080", "1081", "9050") else "http"
        auth = f"{self.proxy_user}:{self.proxy_pass}@" if self.proxy_user else ""
        return f"{scheme}://{auth}{self.proxy_host}:{self.proxy_port}"


settings = Settings()
