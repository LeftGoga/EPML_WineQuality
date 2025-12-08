.PHONY: all install run train train-rf train-mlp train-boosting train-no-mlflow train-analyze train-data-small train-data-large train-data-strict train-data-lenient train-dev train-prod train-test train-dev-rf train-prod-boosting lint format bandit clean reinstall

all: install run

install:
	poetry install

prepare:
	poetry run python src/wine_quality/data.py

features:
	poetry run python -c "from src.wine_quality.data import save_features; save_features()"

run:
	poetry run python src/wine_quality/main.py

# Обучение с Hydra - можно передавать параметры через ARGS
# Пример: make train ARGS="model_type=random_forest model.n_estimators=300"
train:
	poetry run python src/wine_quality/main.py $(ARGS)

# Быстрые команды для обучения разных моделей
train-rf:
	poetry run python src/wine_quality/main.py model=random_forest model_type=random_forest $(ARGS)

train-boosting:
	poetry run python src/wine_quality/main.py model=boosting model_type=boosting $(ARGS)

train-mlp:
	poetry run python src/wine_quality/main.py model=mlp model_type=mlp $(ARGS)

# Обучение без MLflow
train-no-mlflow:
	poetry run python src/wine_quality/main.py no_mlflow=true $(ARGS)

# Обучение с анализом экспериментов
train-analyze:
	poetry run python src/wine_quality/main.py analyze_experiments=true $(ARGS)

# Обучение с разными датасетами
train-data-small:
	poetry run python src/wine_quality/main.py data=data_small $(ARGS)

train-data-large:
	poetry run python src/wine_quality/main.py data=data_large $(ARGS)

train-data-strict:
	poetry run python src/wine_quality/main.py data=data_strict $(ARGS)

train-data-lenient:
	poetry run python src/wine_quality/main.py data=data_lenient $(ARGS)

# Обучение в разных окружениях
train-dev:
	poetry run python src/wine_quality/main.py env=dev $(ARGS)

train-prod:
	poetry run python src/wine_quality/main.py env=prod $(ARGS)

train-test:
	poetry run python src/wine_quality/main.py env=test $(ARGS)

# Комбинированные примеры
train-dev-rf:
	poetry run python src/wine_quality/main.py model=random_forest model_type=random_forest env=dev data=data_small $(ARGS)

train-prod-boosting:
	poetry run python src/wine_quality/main.py model=boosting model_type=boosting env=prod data=data_large $(ARGS)

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
	@if exist outputs (rmdir /s /q outputs)
	@if exist .pytest_cache (rmdir /s /q .pytest_cache)
	@if exist .mypy_cache (rmdir /s /q .mypy_cache)
	@if exist .ruff_cache (rmdir /s /q .ruff_cache)
	del /s /q *.pyc 2>NUL
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo "Очистка завершена"
else
	rm -rf models/ plots/ outputs/ __pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Очистка завершена"
endif

reinstall:
	poetry env remove --all
	poetry install --with dev

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000
