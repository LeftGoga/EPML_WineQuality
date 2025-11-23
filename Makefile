.PHONY: all install run train lint format bandit clean reinstall

all: install run

install:
	poetry install

run train:
	poetry run python -m wine_quality

format:
	poetry run ruff format src
	poetry run ruff check --select I --fix src
	poetry run bandit -r src/wine_quality


lint:
	poetry run ruff check src/wine_quality
	poetry run mypy src/wine_quality
clean:
ifeq ($(OS),Windows_NT)
	@if exist models (rmdir /s /q models)
	@if exist plots (rmdir /s /q plots)
	@if exist .pytest_cache (rmdir /s /q .pytest_cache)
	@if exist .mypy_cache (rmdir /s /q .mypy_cache)
	@if exist .ruff_cache (rmdir /s /q .ruff_cache)
	del /s /q *.pyc 2>NUL
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo "Очистка завершена"
else
	rm -rf models/ plots/ __pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Очистка завершена"
endif

reinstall:
	poetry env remove --all
	poetry install --with dev
