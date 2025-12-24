.PHONY: all install run train train-rf train-mlp train-boosting train-no-mlflow train-analyze train-data-small train-data-large train-data-strict train-data-lenient train-dev train-prod train-test train-dev-rf train-prod-boosting lint format bandit clean reinstall prepare features streamlit docs docs-serve docs-build report report-comparison mlflow luigi luigi-rf luigi-boosting luigi-mlp luigi-no-mlflow luigi-fresh luigi-direct luigi-visualizer luigi-clean

all: install run

help:
	@echo "Доступные команды:"
	@echo "  make install           - Установить зависимости"
	@echo "  make prepare           - Подготовить данные"
	@echo "  make features          - Сохранить фичи"
	@echo "  make run               - Запустить обучение"
	@echo "  make train             - Обучение с Hydra (ARGS=\"model_type=rf\")"
	@echo "  make train-rf          - Обучение Random Forest"
	@echo "  make train-boosting    - Обучение Boosting"
	@echo "  make train-mlp         - Обучение MLP"
	@echo "  make train-no-mlflow   - Обучение без MLflow"
	@echo "  make train-analyze     - Обучение с анализом экспериментов"
	@echo "  make train-data-small  - Обучение на малом датасете"
	@echo "  make train-data-large  - Обучение на большом датасете"
	@echo "  make train-data-strict - Обучение на строгом датасете"
	@echo "  make train-data-lenient - Обучение на мягком датасете"
	@echo "  make train-dev         - Обучение в dev окружении"
	@echo "  make train-prod       - Обучение в prod окружении"
	@echo "  make train-test        - Обучение в test окружении"
	@echo "  make train-dev-rf      - Обучение RF в dev окружении"
	@echo "  make train-prod-boosting - Обучение Boosting в prod окружении"
	@echo "  make streamlit         - Запустить Streamlit приложение"
	@echo "  make docs              - Собрать документацию"
	@echo "  make docs-serve        - Запустить локальный сервер документации"
	@echo "  make docs-build        - Собрать документацию для публикации"
	@echo "  make report            - Создать отчет об эксперименте (TASK_ID=xxx)"
	@echo "  make report-comparison  - Создать отчет со сравнением (TASK_ID=xxx COMPARISON=id1 id2)"
	@echo "  make mlflow            - Запустить MLflow сервер"
	@echo "  make luigi             - Запустить Luigi pipeline"
	@echo "  make luigi-rf          - Запустить Luigi pipeline с RF"
	@echo "  make luigi-boosting    - Запустить Luigi pipeline с Boosting"
	@echo "  make luigi-mlp         - Запустить Luigi pipeline с MLP"
	@echo "  make luigi-no-mlflow   - Запустить Luigi pipeline без MLflow"
	@echo "  make luigi-fresh       - Очистить и запустить Luigi pipeline"
	@echo "  make luigi-direct      - Запустить Luigi напрямую"
	@echo "  make luigi-visualizer  - Запустить Luigi визуализатор"
	@echo "  make luigi-clean       - Очистить Luigi выходные файлы"
	@echo "  make lint              - Проверить код линтером"
	@echo "  make format            - Отформатировать код"
	@echo "  make clean             - Очистить временные файлы"
	@echo "  make reinstall         - Переустановить зависимости"

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

docs:
	poetry run mkdocs build

docs-serve:
	poetry run mkdocs serve

docs-build:
	poetry run mkdocs build --clean

report:
	@echo "Создание отчета для задачи: $(TASK_ID)"
	poetry run python -m wine_quality.generate_experiment_report --task-id $(TASK_ID) --output docs/experiment_report.md

report-comparison:
	@echo "Создание отчета со сравнением"
	@if [ -n "$(TASK_ID)" ]; then \
		poetry run python -m wine_quality.generate_experiment_report --task-id $(TASK_ID) --comparison $(COMPARISON) --output docs/experiment_report.md; \
	else \
		poetry run python -m wine_quality.generate_experiment_report --comparison $(COMPARISON) --output docs/experiment_report.md; \
	fi

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000

luigi:
	poetry run python run_luigi_pipeline.py $(ARGS)

luigi-rf:
	poetry run python run_luigi_pipeline.py model_type=random_forest $(ARGS)

luigi-boosting:
	poetry run python run_luigi_pipeline.py model_type=boosting $(ARGS)

luigi-mlp:
	poetry run python run_luigi_pipeline.py model_type=mlp $(ARGS)

luigi-no-mlflow:
	poetry run python run_luigi_pipeline.py no_mlflow=true $(ARGS)

luigi-fresh:
	$(MAKE) luigi-clean
	$(MAKE) luigi

# Прямой запуск Luigi без Hydra (для совместимости)
luigi-direct:
	poetry run python -m luigi --module src.wine_quality.luigi_pipeline WineQualityPipeline --local-scheduler --log-level INFO

luigi-visualizer:
ifeq ($(OS),Windows_NT)
	@if not exist logs\luigi mkdir logs\luigi
	poetry run luigid --logdir ./logs/luigi
else
	@mkdir -p logs/luigi
	poetry run luigid --background --logdir ./logs/luigi
endif

luigi-clean:
ifeq ($(OS),Windows_NT)
	@if exist data\features.csv (del /q data\features.csv)
	@if exist data\winequality-red.csv (del /q data\winequality-red.csv)
	@if exist models\wine_*.pkl (del /q models\wine_*.pkl)
	@if exist models\*_metrics.json (del /q models\*_metrics.json)
	@if exist models\*_report.json (del /q models\*_report.json)
	@if exist outputs\evaluation_report.json (del /q outputs\evaluation_report.json)
	@echo "Luigi выходные файлы удалены"
else
	rm -f data/features.csv data/winequality-red.csv
	rm -f models/wine_*.pkl models/*_metrics.json models/*_report.json
	rm -f outputs/evaluation_report.json
	@echo "Luigi выходные файлы удалены"
endif

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
	@if exist site (rmdir /s /q site)
	del /s /q *.pyc 2>NUL
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo "Очистка завершена"
else
	rm -rf models/ plots/ outputs/ __pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/ site/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Очистка завершена"
endif

reinstall:
	poetry env remove --all
	poetry install --with dev
