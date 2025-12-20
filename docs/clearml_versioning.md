# Автоматическое версионирование моделей в ClearML

## Описание

Реализована система автоматического версионирования моделей в ClearML с инкрементом версий и метаданными.

## Основные функции

### 1. `log_model_to_clearml()` - Логирование модели с автоматическим версионированием

Автоматически создает новую версию модели при каждом вызове, инкрементируя номер версии.

```python
from wine_quality.clearml_utils import log_model_to_clearml

# Логирование модели с автоматическим версионированием
version = log_model_to_clearml(
    model=trained_model,
    model_name="wine_boosting",
    model_path="models/wine_boosting.pkl",
    framework="scikit-learn",
    tags=["production", "wine-quality"],
    labels={
        "accuracy": "0.85",
        "f1_weighted": "0.84",
        "model_type": "boosting",
    },
    auto_version=True,  # Включено по умолчанию
)

print(f"Модель зарегистрирована с версией: {version}")
```

### 2. `get_model_version_info()` - Получение информации о версиях

Получает информацию о текущих версиях модели.

```python
from wine_quality.clearml_utils import get_model_version_info

version_info = get_model_version_info("wine_boosting")
print(f"Всего версий: {version_info['version_count']}")
print(f"Последняя версия: {version_info['latest_version']}")
print(f"Следующая версия: {version_info['next_version']}")
```

### 3. `compare_clearml_models()` - Сравнение версий моделей

Сравнивает все версии модели по метрикам.

```python
from wine_quality.clearml_utils import compare_clearml_models

comparison = compare_clearml_models(
    model_name="wine_boosting",
    metric_keys=["accuracy", "f1_weighted"],
)

print(f"Всего версий: {comparison['total_versions']}")
print(f"Лучшая версия: {comparison['best_version']}")

for version_data in comparison['versions']:
    print(f"Версия {version_data['version']}:")
    print(f"  Accuracy: {version_data['metrics'].get('accuracy', 'N/A')}")
    print(f"  F1-score: {version_data['metrics'].get('f1_weighted', 'N/A')}")
    print(f"  Создана: {version_data['created_at']}")
```

## Формат версий

Версии моделей имеют формат `major.minor` (например, `1.0`, `1.1`, `1.2`):
- При каждом логировании модели минорная версия автоматически инкрементируется
- Первая версия всегда `1.0`
- Если предыдущая версия была `1.5`, следующая будет `1.6`

## Метаданные версий

Каждая версия модели содержит следующие метаданные в labels:
- `version`: номер версии (например, "1.0", "1.1")
- `version_count`: общее количество версий
- `created_at`: дата и время создания версии (ISO формат)
- `previous_version`: номер предыдущей версии (если есть)
- Метрики модели (accuracy, f1_weighted и т.д.)
- Параметры модели (model_type и т.д.)

## Пример использования в пайплайне

Модели автоматически версионируются при использовании `run_experiment()`:

```python
from wine_quality.model import run_experiment

result = run_experiment(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_type="boosting",
    use_clearml=True,
    register_model_name="wine_boosting",
    auto_version=True,  # Включено по умолчанию
)

# Версия модели доступна через result, если нужно
```

## Интеграция с существующим кодом

Автоматическое версионирование уже интегрировано в:
- `src/wine_quality/model.py` - функция `run_experiment()`
- `src/wine_quality/clearml_pipeline.py` - компоненты пайплайна

Все существующие вызовы `log_model_to_clearml()` автоматически используют версионирование.
