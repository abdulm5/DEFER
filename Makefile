.PHONY: setup test lint repro

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

repro:
	python -m scripts.reproduce --output-root artifacts/demo --tasks-per-domain 300 --repeats 5 --seed 42
