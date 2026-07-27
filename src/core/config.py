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
from pydantic import AliasChoices, Field, field_validator
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
    #: Короткий потолок для реакции перед дословным блоком (одно-два предложения).
    llm_max_tokens_short: int = 200

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

    # ─── Персона агента ─────────────────────────────────────────────────────
    #: Персона одна на агента и от сценария звонка не зависит.
    agent_name: str = "Дарья"
    agent_company: str = "Вектор"
    agent_role: str = "менеджер федеральной академии вождения"
    agent_tone: str = "живой, доброжелательный, уважительный, без канцелярита"
    #: ``female`` или ``male`` — из значения собирается правило о роде в промпте.
    agent_gender: str = "female"
    #: Аварийная реплика, когда модель не ответила в бюджет хода.
    #: Для скрипта продаж (без params в файле) берётся отсюда.
    agent_fallback: str = "Прошу прощения, отвлеклась на секунду. Повторите, пожалуйста?"
    #: Что сказать, когда ответа нет ни в скрипте, ни в справочнике.
    agent_unknown: str = (
        "Честно скажу — этого у меня под рукой нет, выдумывать не буду. "
        "Уточню и вернусь с точным ответом, а пока давайте двигаться дальше."
    )
    #: Общие фразы-заглушки, пока идём за данными.
    agent_fillers: list[str] = Field(
        default_factory=lambda: [
            "Секунду, открываю Ваш город.",
            "Минутку, посмотрю по филиалам.",
            "Так, сейчас уточню по стоимости.",
            "Секундочку, гляну.",
        ]
    )
    #: Реакции, пока открываем город. Плейсхолдер ``{place}`` / ``{city}``.
    agent_city_fillers: list[str] = Field(
        default_factory=lambda: [
            "так, {place}… секунду, открываю по {place}",
            "а, {place}… так, гляну по {place}",
            "{place}, да… минутку, открою карточку",
            "так, по {place}… секундочку",
            "угу, {place}… сейчас открою",
        ]
    )
    #: Реакции, пока подбираем филиал.
    agent_branch_fillers: list[str] = Field(
        default_factory=lambda: [
            "так, {place}… секунду, гляну адреса рядом",
            "а, {place}… минутку, открою адреса",
            "так, по {place}… сейчас сверю",
            "угу, {place}… секунду, посмотрю что ближе",
            "{place}, да… сейчас подберу пару адресов",
        ]
    )
    #: Запасные формулировки цены, если справочник отдал сумму без готовой фразы.
    #: Для скрипта продаж основная формулировка приходит из базы.
    agent_price_no_amount: str = (
        "Точную сумму зафиксируем при оформлении — она зависит от пакета и действующих условий."
    )
    agent_price_unreliable: str = (
        "Стоимость — от {amount} рублей. Итог зафиксируем при оформлении: "
        "зависит от пакета и действующих условий."
    )
    agent_price_reliable: str = "Стоимость — {amount} рублей, это полный курс."

    @property
    def agent_price_texts(self):
        """Тексты трёх веток цены из настроек агента."""
        from script.models import PriceTexts

        return PriceTexts(
            no_amount=self.agent_price_no_amount,
            unreliable=self.agent_price_unreliable,
            reliable=self.agent_price_reliable,
        )

    # ─── Скрипт разговора ───────────────────────────────────────────────────
    #: Идентификатор скрипта по умолчанию: с ним стартует звонок, если
    #: конкретный не пришёл в context.
    script_id: str = "vector_ru"
    #: Версия скрипта по умолчанию. Пусто — берётся последняя из источника.
    script_version: str | None = None
    #: Каталог с JSON-скриптами. Пусто — каталог `data` рядом с кодом.
    script_dir: str | None = None
    #: Порог попыток задать шаг: сколько раз ведущий шаг уходит в генерацию
    #: без ответа клиента, прежде чем чекер закроет его. Счётчик растёт только
    #: у ведущего шага хода, не у висящих в шапке.
    step_attempt_limit: int = Field(
        default=2,
        validation_alias=AliasChoices(
            "step_attempt_limit",
            "STEP_ATTEMPT_LIMIT",
            "step_patience_limit",
            "STEP_PATIENCE_LIMIT",
        ),
    )
    #: Сколько висящих шагов в шапке, при котором новый шаг с верхушки
    #: больше не добавляется — генератор дорабатывает висящее.
    pending_steps_soft_cap: int = Field(
        default=4,
        validation_alias=AliasChoices("pending_steps_soft_cap", "PENDING_STEPS_SOFT_CAP"),
        description=(
            "Сколько висящих шагов в шапке, при котором новый шаг с верхушки "
            "больше не добавляется — генератор дорабатывает висящее."
        ),
    )

    # ─── Служебный чекер в реальном времени ─────────────────────────────────
    #: Порог прироста накопленного текста (символы) между служебными проходами.
    #: Меньше порога и не первый проход — модель не зовём.
    checker_min_growth_chars: int = 10
    #: Имя графа служебного чекера в ``langgraph.json``. Пусто → ``vector_checker``.
    checker_graph_id: str | None = None
    #: ``multitask_strategy`` для запусков служебного графа (клиент SDK).
    checker_multitask_strategy: str | None = None
    #: ``multitask_strategy`` основного хода. Перед стартом клиент отменяет
    #: идущий служебный проход, иначе enqueue поставит основной в очередь.
    agent_multitask_strategy: str | None = None

    # ─── Redis: рабочий прогресс скрипта звонка ─────────────────────────────
    #: Адрес Redis. Совпадает с ``REDIS_URI`` сервера LangGraph.
    redis_url: str = Field(
        default="redis://localhost:6379",
        validation_alias=AliasChoices("redis_url", "REDIS_URI", "redis_uri"),
    )
    #: TTL слепка скрипта: чуть длиннее ожидаемого звонка.
    script_redis_ttl: int = 7200
    #: Суффикс, которым бот помечает тред лайв-канала; отбрасывается при
    #: вычислении идентификатора звонка, чтобы оба канала попадали в один скрипт.
    live_thread_suffix: str = Field(default="-live", alias="LIVE_THREAD_SUFFIX")

    # ─── Заглушка в эфир ────────────────────────────────────────────────────
    #: Фразы в эфир, пока идёт поиск города/филиала. По умолчанию выключены:
    #: формулировки ещё сырые (именительный падеж, не вовремя).
    lookup_fillers_enabled: bool = False

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
        "checker_graph_id",
        "checker_multitask_strategy",
        "agent_multitask_strategy",
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
    def checker_assistant_id(self) -> str:
        """Идентификатор ассистента служебного чекера для SDK."""
        return self.checker_graph_id or "vector_checker"

    @property
    def checker_run_strategy(self) -> str:
        """Стратегия multitask для служебного прохода: interrupt."""
        return self.checker_multitask_strategy or "interrupt"

    @property
    def agent_run_strategy(self) -> str:
        """Стратегия multitask основного хода: enqueue."""
        return self.agent_multitask_strategy or "enqueue"

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
