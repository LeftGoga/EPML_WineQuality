.PHONY: all install run train lint format clean reinstall

all: install run

install:
	poetry install --with dev

run train:
	poetry run python -m wine_quality

format:
	poetry run ruff format src
	poetry run ruff check --select I --fix src

lint:
	-poetry run ruff check src
	-poetry run mypy src

clean:
	rm -rf models/ plots/ __pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Очистка завершена"

reinstall:
	poetry env remove --all
	poetry install --with dev
