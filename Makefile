SHELL := /bin/bash
PACKAGE_SLUG=aiviz
PYTHON_VERSION := $(shell cat .python-version)

.PHONY: help
help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

.PHONY: sync
sync: $(PYTHON_VENV) uv.lock  ## Install/sync dependencies, create .venv
	@command -v uv >/dev/null 2>&1 || { echo >&2 "uv is not installed. Installing via pip..."; pip install uv; }
	uv sync --group dev

.PHONY: test
test:  ## Run pytest with coverage
	uv run pytest --cov=./${PACKAGE_SLUG} --cov-report=term-missing tests

.PHONY: lint
lint:  ## Check types (mypy) and lint/format (ruff)
	@status=0; \
	echo "==> ty"; \
	uv run ty check --no-progress || status=1; \
	echo "==> ruff check"; \
	uv run ruff check . || status=1; \
	echo "==> ruff format --check"; \
	uv run ruff format . --check || status=1; \
	if [ $$status -ne 0 ]; then \
		echo ""; \
		echo "Lint failed. Run 'make lint-fix' to auto-fix formatting/lint issues."; \
	fi; \
	exit $$status

.PHONY: lint-fix
lint-fix:  ## Auto-fix lint and formatting issues with ruff
	uv run ruff check . --fix
	uv run ruff format .

.PHONY: dev
dev:  ## Run local streamlit server
	uv run streamlit run aiviz/main.py

