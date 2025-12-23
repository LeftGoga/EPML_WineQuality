# Модуль model

Модуль для обучения и оценки моделей машинного обучения.

## Классы

### `ModelType`

Enum для типов моделей.

```python
class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    BOOSTING = "boosting"
    MLP = "mlp"
```

## Функции

### `run_experiment`

Запускает эксперимент по обучению и оценке модели.

```python
def run_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: ModelType,
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    boosting_n_estimators: int | None = None,
    boosting_max_depth: int | None = None,
    boosting_learning_rate: float | None = None,
    mlp_hidden_layer_sizes: tuple[int, ...] | None = None,
    mlp_max_iter: int | None = None,
    random_state: int | None = None,
    use_mlflow: bool = False,
    use_clearml: bool = False,
    ...
) -> dict[str, Any]
```

**Параметры:**

- `X_train`, `y_train`: Обучающая выборка
- `X_test`, `y_test`: Тестовая выборка
- `model_type`: Тип модели (ModelType)
- `rf_n_estimators`: Количество деревьев для Random Forest
- `rf_max_depth`: Максимальная глубина для Random Forest
- `boosting_n_estimators`: Количество деревьев для Boosting
- `boosting_max_depth`: Максимальная глубина для Boosting
- `boosting_learning_rate`: Скорость обучения для Boosting
- `mlp_hidden_layer_sizes`: Размеры скрытых слоев для MLP
- `mlp_max_iter`: Максимальное количество итераций для MLP
- `random_state`: Seed для воспроизводимости
- `use_mlflow`: Использовать MLflow для логирования
- `use_clearml`: Использовать ClearML для логирования

**Возвращает:**

- `dict`: Словарь с результатами:
  - `model`: Обученная модель
  - `metrics`: Метрики (accuracy, f1_weighted, и т.д.)
  - `predictions`: Предсказания
  - `confusion_matrix`: Матрица ошибок

**Пример:**

```python
from wine_quality.model import ModelType, run_experiment

result = run_experiment(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_type=ModelType.BOOSTING,
    boosting_n_estimators=100,
    boosting_max_depth=3,
    use_clearml=True,
)

print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
```

## Поддерживаемые модели

### Random Forest

```python
result = run_experiment(
    ...,
    model_type=ModelType.RANDOM_FOREST,
    rf_n_estimators=200,
    rf_max_depth=10,
)
```

### Gradient Boosting

```python
result = run_experiment(
    ...,
    model_type=ModelType.BOOSTING,
    boosting_n_estimators=100,
    boosting_max_depth=3,
    boosting_learning_rate=0.1,
)
```

### MLP (Neural Network)

```python
result = run_experiment(
    ...,
    model_type=ModelType.MLP,
    mlp_hidden_layer_sizes=(100, 50),
    mlp_max_iter=500,
)
```
