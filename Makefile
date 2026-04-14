VENV_DIR ?= local_test/.venv
BOOTSTRAP_PYTHON ?= python3
VENV_BIN := $(VENV_DIR)/bin
PYTHON := $(VENV_BIN)/python
PIP := $(PYTHON) -m pip
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy --python-executable $(PYTHON)
PYTEST := $(PYTHON) -m pytest

.PHONY: bootstrap install format format-check lint type test test-unit test-integration check

bootstrap:
	test -x $(PYTHON) || $(BOOTSTRAP_PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

install: bootstrap

format: bootstrap
	$(RUFF) format .

format-check: bootstrap
	$(RUFF) format --check .

lint: bootstrap
	$(RUFF) check src tests

type: bootstrap
	$(MYPY) src tests

test-unit: bootstrap
	$(PYTEST) tests/unit -q

test-integration: bootstrap
	$(PYTEST) tests/integration -q

test: bootstrap
	$(PYTEST) -q

check: format-check lint type test
