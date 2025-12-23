# Модуль config

Модуль конфигурации проекта.

## Константы

### Пути

- `BASE_DIR`: Корневая директория проекта
- `PLOTS_DIR`: Директория для графиков
- `MODELS_DIR`: Директория для моделей

### URL данных

- `DATA_URL`: URL датасета Wine Quality

### Параметры данных

- `TEST_SIZE`: Размер тестовой выборки (по умолчанию 0.2)
- `RANDOM_STATE`: Seed для воспроизводимости (по умолчанию 42)

### Параметры моделей

#### Random Forest

- `RF_N_ESTIMATORS`: Количество деревьев (по умолчанию 200)
- `RF_MAX_DEPTH`: Максимальная глубина (по умолчанию None)

#### Gradient Boosting

- `BOOSTING_N_ESTIMATORS`: Количество деревьев (по умолчанию 100)
- `BOOSTING_MAX_DEPTH`: Максимальная глубина (по умолчанию 3)
- `BOOSTING_LEARNING_RATE`: Скорость обучения (по умолчанию 0.1)

#### MLP

- `MLP_HIDDEN_LAYER_SIZES`: Размеры скрытых слоев (по умолчанию (100, 50))
- `MLP_MAX_ITER`: Максимальное количество итераций (по умолчанию 500)

### MLflow

- `MLFLOW_EXPERIMENT_NAME`: Имя эксперимента MLflow

## Использование

```python
from wine_quality.config import (
    BASE_DIR,
    TEST_SIZE,
    RANDOM_STATE,
    BOOSTING_N_ESTIMATORS,
)

print(f"Базовая директория: {BASE_DIR}")
print(f"Размер тестовой выборки: {TEST_SIZE}")
print(f"Количество деревьев: {BOOSTING_N_ESTIMATORS}")
```
