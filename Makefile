.PHONY: install lint format type test

install:
	python -m pip install -U pip
	pip install -e ".[dev]"

format:
	ruff format .

lint:
	ruff check .

type:
	mypy src tests

test:
	pytest
