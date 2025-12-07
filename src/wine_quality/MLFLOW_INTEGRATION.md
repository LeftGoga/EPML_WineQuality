# Интеграция MLflow

Этот документ описывает интеграцию MLflow в проект wine_quality.

## Структура файлов

- `mlflow_context.py` - Контекстные менеджеры для работы с MLflow
- `mlflow_decorators.py` - Декораторы для автоматического логирования
- `mlflow_utils.py` - Утилиты для поиска, фильтрации и сравнения экспериментов
- `mlflow_example.py` - Примеры использования всех возможностей
- `mlflow_registry.py` - Утилиты для работы с Model Registry (существующий файл)

## Настройка

### Docker Compose

Проект настроен для работы с MLflow через Docker Compose. В `docker-compose.yaml` добавлены:

- **PostgreSQL** - база данных для хранения метаданных экспериментов
- **MLflow Tracking Server** - сервер для трекинга экспериментов

Для запуска:
```bash
docker compose up -d
```

MLflow UI будет доступен по адресу: http://localhost:5000

### Переменные окружения

- `MLFLOW_TRACKING_URI` - URI для подключения к MLflow серверу (по умолчанию: `http://127.0.0.1:5000`)

## Использование

### 1. Контекстные менеджеры

#### MLflowExperimentContext
Автоматически создаёт и настраивает эксперимент:

```python
from mlflow_context import MLflowExperimentContext

with MLflowExperimentContext("my_experiment", tags={"team": "ml"}) as exp:
    mlflow.log_param("param1", "value1")
    mlflow.log_metric("accuracy", 0.95)
```

#### MLflowRunContext
Автоматически создаёт и завершает run:

```python
from mlflow_context import MLflowRunContext

with MLflowRunContext(
    experiment_name="my_experiment",
    run_name="test_run",
    params={"n_estimators": 100},
    tags={"framework": "sklearn"}
) as run:
    mlflow.log_metric("accuracy", 0.95)
```

#### mlflow_artifact_context
Автоматически логирует артефакты:

```python
from mlflow_context import mlflow_artifact_context

with mlflow_artifact_context("plots", "visualizations") as artifact_dir:
    plot_path = artifact_dir / "plot.png"
    plt.savefig(plot_path)
    # Файл автоматически залогируется
```

#### mlflow_model_context
Автоматически логирует модель:

```python
from mlflow_context import mlflow_model_context

with mlflow_model_context(
    model=my_model,
    artifact_path="model",
    registered_model_name="MyModel"
):
    pass
```

### 2. Декораторы

#### @log_params
Автоматически логирует параметры функции:

```python
from mlflow_decorators import log_params

@log_params
def train_model(n_estimators=100, max_depth=5):
    # Параметры автоматически залогируются
    pass
```

#### @log_metrics
Автоматически логирует метрики из возвращаемого значения:

```python
from mlflow_decorators import log_metrics

@log_metrics(["accuracy", "f1_score"])
def evaluate_model():
    return {"accuracy": 0.95, "f1_score": 0.92}
```

#### @log_execution_time
Логирует время выполнения:

```python
from mlflow_decorators import log_execution_time

@log_execution_time
def train_model():
    # Время выполнения залогируется как метрика
    pass
```

#### @mlflow_run
Автоматически создаёт MLflow run:

```python
from mlflow_decorators import mlflow_run

@mlflow_run(experiment_name="my_experiment", run_name="test_run")
def train_model():
    mlflow.log_param("param1", "value1")
    mlflow.log_metric("accuracy", 0.95)
```

#### Комбинирование декораторов

```python
from mlflow_decorators import combine_decorators, log_params, log_execution_time, log_metrics

@combine_decorators(
    log_params,
    log_execution_time,
    log_metrics(["accuracy", "f1_score"])
)
def train_and_evaluate():
    pass
```

### 3. Утилиты для работы с экспериментами

#### Поиск runs

```python
from mlflow_utils import search_runs

runs = search_runs(
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=100
)
```

#### Фильтрация по метрикам

```python
from mlflow_utils import filter_runs_by_metrics

filtered = filter_runs_by_metrics(
    runs,
    {
        "accuracy": (0.8, 1.0),  # 0.8 <= accuracy <= 1.0
        "f1_score": 0.9,  # f1_score >= 0.9
    }
)
```

#### Сравнение runs

```python
from mlflow_utils import compare_runs

df = compare_runs(
    runs,
    metric_names=["accuracy", "f1_score"],
    param_names=["n_estimators", "model_type"]
)
```

#### Получение лучших runs

```python
from mlflow_utils import get_best_runs

best_runs = get_best_runs(runs, "accuracy", ascending=False, top_k=5)
```

#### Сводка по эксперименту

```python
from mlflow_utils import get_experiment_summary

summary = get_experiment_summary("experiment_id")
print(summary)
```

#### Экспорт в CSV

```python
from mlflow_utils import export_runs_to_csv

export_runs_to_csv(runs, "experiments.csv")
```

## Примеры

Полные примеры использования находятся в файле `mlflow_example.py`.

## Интеграция с существующим кодом

Существующий код в `model.py` и `main.py` уже использует базовую интеграцию MLflow. Новые утилиты можно использовать для:

- Анализа результатов экспериментов
- Поиска лучших моделей
- Сравнения различных конфигураций
- Автоматизации логирования через декораторы
