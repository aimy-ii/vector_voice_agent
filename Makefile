.DEFAULT_GOAL := help
.PHONY: help sync dev lint format test build_image up down restart prod logs ps ok

IMAGE ?= vector_voice_agent

help: ## Показать список целей
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync: ## Синхронизировать зависимости
	uv sync --all-extras

dev: ## Поднять граф локально в памяти (Studio, без postgres и redis)
	uv run langgraph dev --port 8127

lint: ## Проверить линтером
	uv run ruff check src tests

format: ## Отформатировать
	uv run ruff format src tests

test: ## Прогнать офлайн-тесты
	uv run pytest -q

build_image: ## Собрать образ сервера (langgraph build)
	uv run langgraph build -t $(IMAGE)

up: ## Поднять postgres, redis и сервер
	docker compose up -d

down: ## Остановить стек
	docker compose down

restart: build_image ## Пересобрать образ и пересоздать только сервер
	docker compose up -d --force-recreate vector-agent-api

prod: build_image up ## Собрать образ и поднять весь стек одной командой

logs: ## Хвост логов сервера
	docker compose logs -f vector-agent-api

ps: ## Состояние контейнеров
	docker compose ps

ok: ## Проверить живость сервера
	curl -fsS http://localhost:8127/ok && echo
