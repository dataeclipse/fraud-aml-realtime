.PHONY: install install-ml fmt lint type test run

install:
	uv sync --extra data

install-ml:
	uv sync --extra ml

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run mypy src

test:
	uv run pytest

run:
	uv run uvicorn fraud_aml.serving.app:app --host 0.0.0.0 --port 8000
