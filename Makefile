.PHONY: all install run train lint format bandit clean reinstall

all: install run

install:
	poetry install

prepare:
	poetry run python src/wine_quality/data.py

run train:
	poetry run python src/wine_quality/main.py

train-boosting:
	poetry run python src/wine_quality/main.py --model-type boosting

train-rf:
	poetry run python src/wine_quality/main.py --model-type random_forest

train-mlp:
	poetry run python src/wine_quality/main.py --model-type mlp

train-custom:
	poetry run python src/wine_quality/main.py $(ARGS)

streamlit:
	poetry run streamlit run src/wine_quality/app.py

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

mlflow: 
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000