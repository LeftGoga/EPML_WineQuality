# API Документация

Обзор API проекта Wine Quality ML.

## Модули

Проект состоит из следующих основных модулей:

- **[data](data.md)** - Загрузка и подготовка данных
- **[features](features.md)** - Инженерия признаков
- **[model](model.md)** - Обучение и оценка моделей
- **[config](config.md)** - Конфигурация проекта
- **[ClearML](clearml.md)** - Интеграция с ClearML

## Быстрый обзор

### Загрузка данных

```python
from wine_quality.data import load_data, create_target

df = load_data()
df = create_target(df, threshold=7)
```

### Обучение модели

```python
from wine_quality.model import ModelType, run_experiment

result = run_experiment(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_type=ModelType.BOOSTING,
)
```

### Работа с ClearML

```python
from wine_quality.clearml_context import ClearMLTaskContext

with ClearMLTaskContext("Wine Quality", "My Experiment"):
    # Ваш код
    pass
```

## Структура

Все модули находятся в пакете `wine_quality`:

```
src/wine_quality/
├── data.py              # Работа с данными
├── features.py          # Инженерия признаков
├── model.py             # Модели машинного обучения
├── config.py            # Конфигурация
├── clearml_*.py         # ClearML интеграция
└── ...
```

## Следующие шаги

Изучите документацию отдельных модулей для детальной информации.
