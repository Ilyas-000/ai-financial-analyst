.PHONY: help install lint format test up down up-obs down-obs logs ps migrate seed ingest run eval eval-fast eval-rag

help:
	@echo "Available targets:"
	@echo "  install     Install dependencies via uv (incl. dev group)"
	@echo "  lint        Run ruff lint"
	@echo "  format      Run ruff format"
	@echo "  test        Run pytest"
	@echo "  up          Start base infra (postgres + qdrant)"
	@echo "  down        Stop base infra"
	@echo "  up-obs      Start observability stack (langfuse + clickhouse + minio + redis)"
	@echo "  down-obs    Stop observability stack"
	@echo "  logs        Tail compose logs"
	@echo "  ps          List running compose services"
	@echo "  migrate     Apply alembic migrations"
	@echo "  seed        Seed database with fixtures"
	@echo "  ingest      Run RAG ingestion pipeline"
	@echo "  run         Run FastAPI app via uvicorn (APP_HOST/APP_PORT from env)"
	@echo "  eval        Run full eval suite"
	@echo "  eval-fast   Run only route_accuracy"
	@echo "  eval-rag    Run only RAGAS metrics"

# --- Dev environment ---
install:
	uv sync --all-groups

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

# --- Infra ---
up:
	docker compose up -d

down:
	docker compose down

up-obs:
	docker compose --profile observability up -d

down-obs:
	docker compose --profile observability down

logs:
	docker compose logs -f

ps:
	docker compose ps

# --- Data ---
migrate:
	uv run alembic upgrade head

seed:
	uv run python -m app.db.seed

ingest:
	uv run python -m app.rag.ingest

# --- Run app ---
APP_HOST ?= 0.0.0.0
APP_PORT ?= 8000
run:
	uv run uvicorn app.main:app --host $(APP_HOST) --port $(APP_PORT)

# --- Evaluation ---
eval:
	uv run python -m eval.run_eval

eval-fast:
	uv run python -m eval.run_eval --only route_accuracy

eval-rag:
	uv run python -m eval.run_eval --only ragas
