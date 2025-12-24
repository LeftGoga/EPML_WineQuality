# Модуль ClearML

Модули для интеграции с ClearML.

## Модули

### `clearml_context`

Контекстный менеджер для работы с задачами ClearML.

::: wine_quality.clearml_context.ClearMLTaskContext
    options:
      show_source: true
      heading_level: 3

::: wine_quality.clearml_context.is_clearml_available
    options:
      show_source: true
      heading_level: 3

### `clearml_utils`

Утилиты для логирования в ClearML.

::: wine_quality.clearml_utils.log_params_to_clearml
    options:
      show_source: true
      heading_level: 3

::: wine_quality.clearml_utils.log_metrics_to_clearml
    options:
      show_source: true
      heading_level: 3

::: wine_quality.clearml_utils.log_plot_to_clearml
    options:
      show_source: true
      heading_level: 3

::: wine_quality.clearml_utils.log_artifact_to_clearml
    options:
      show_source: true
      heading_level: 3

::: wine_quality.clearml_utils.log_model_to_clearml
    options:
      show_source: true
      heading_level: 3

### `clearml_model_registry`

Регистрация и версионирование моделей.

::: wine_quality.clearml_model_registry.register_model_with_version
    options:
      show_source: true
      heading_level: 3

## Примеры использования

### Базовый пример

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
