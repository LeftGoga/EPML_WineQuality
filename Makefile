.PHONY: help install run streamlit docs docs-serve docs-build clean

help:
	@echo "Доступные команды:"
	@echo "  make install      - Установить зависимости"
	@echo "  make run          - Запустить обучение"
	@echo "  make streamlit    - Запустить Streamlit приложение"
	@echo "  make docs         - Собрать документацию"
	@echo "  make docs-serve   - Запустить локальный сервер документации"
	@echo "  make docs-build   - Собрать документацию для публикации"
	@echo "  make clean        - Очистить временные файлы"

install:
	poetry install

run:
	poetry run python run_clearml_experiment.py

streamlit:
	poetry run streamlit run src/wine_quality/app.py

docs:
	poetry run mkdocs build

docs-serve:
	poetry run mkdocs serve

docs-build:
	poetry run mkdocs build --clean

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf site
