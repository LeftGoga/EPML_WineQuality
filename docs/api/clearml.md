# Модуль ClearML

Модули для интеграции с ClearML.

## Модули

### `clearml_context`

Контекстный менеджер для работы с задачами ClearML.

#### `ClearMLTaskContext`

```python
class ClearMLTaskContext:
    def __init__(
        self,
        project_name: str,
        task_name: str,
        task_type: str = "training"
    )
```

**Пример:**

```python
from wine_quality.clearml_context import ClearMLTaskContext

with ClearMLTaskContext(
    project_name="Wine Quality",
    task_name="My Experiment"
) as ctx:
    task = ctx.task
    # Ваш код
```

### `clearml_utils`

Утилиты для логирования в ClearML.

#### `log_params_to_clearml`

Логирует параметры в ClearML.

```python
def log_params_to_clearml(params: dict[str, Any]) -> None
```

#### `log_metrics_to_clearml`

Логирует метрики в ClearML.

```python
def log_metrics_to_clearml(
    metrics: dict[str, float],
    iteration: int = 0
) -> None
```

#### `log_plot_to_clearml`

Логирует графики в ClearML.

```python
def log_plot_to_clearml(
    plot_path: str,
    title: str = "Plot",
    series: str = "default"
) -> None
```

#### `log_artifact_to_clearml`

Логирует артефакты в ClearML.

```python
def log_artifact_to_clearml(
    artifact_path: str,
    artifact_name: str
) -> None
```

#### `log_model_to_clearml`

Логирует модель в ClearML.

```python
def log_model_to_clearml(
    model: Any,
    model_name: str,
    model_path: str,
    framework: str = "scikit-learn"
) -> None
```

### `clearml_model_registry`

Регистрация и версионирование моделей.

#### `register_model_with_version`

Регистрирует модель с автоматическим версионированием.

```python
def register_model_with_version(
    model: Any,
    model_name: str,
    model_path: str,
    framework: str = "scikit-learn",
    tags: list[str] | None = None,
    labels: dict[str, str] | None = None,
    auto_version: bool = True
) -> dict[str, Any]
```

**Возвращает:**

- `dict`: Информация о версии модели:
  - `version`: Номер версии
  - `model_id`: ID модели в ClearML

**Пример:**

```python
from wine_quality.clearml_model_registry import register_model_with_version

version_info = register_model_with_version(
    model=trained_model,
    model_name="wine_quality_boosting",
    model_path="models/wine_boosting.pkl",
    framework="scikit-learn",
    tags=["boosting", "wine_quality"],
    auto_version=True,
)
```

### `clearml_pipeline`

Пайплайны ClearML.

#### `wine_quality_pipeline`

Основной пайплайн для обучения модели.

```python
@PipelineDecorator.pipeline(
    name="Wine Quality Training Pipeline",
    project="Wine Quality",
    version="1.0",
)
def wine_quality_pipeline(...) -> dict[str, Any]
```

**Пример:**

```python
from wine_quality.clearml_pipeline import wine_quality_pipeline
from clearml import PipelineDecorator

PipelineDecorator.run_locally()

results = wine_quality_pipeline(
    data_path="data/winequality-red.csv",
    model_type="boosting",
)
```

## Полный пример

```python
from wine_quality.clearml_context import ClearMLTaskContext
from wine_quality.clearml_utils import (
    log_params_to_clearml,
    log_metrics_to_clearml,
)
from wine_quality.clearml_model_registry import register_model_with_version

with ClearMLTaskContext("Wine Quality", "My Experiment") as ctx:
    # Логирование параметров
    log_params_to_clearml({
        "learning_rate": 0.1,
        "n_estimators": 100,
    })

    # Обучение модели
    # ... ваш код обучения ...

    # Логирование метрик
    log_metrics_to_clearml({
        "accuracy": 0.95,
        "f1_weighted": 0.94,
    })

    # Регистрация модели
    register_model_with_version(
        model=trained_model,
        model_name="wine_quality_model",
        model_path="models/model.pkl",
    )
```
