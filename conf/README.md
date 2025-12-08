# Структура конфигураций Hydra

Этот документ описывает структуру конфигураций проекта и способы их использования.

## Структура директорий

```
conf/
├── config.yaml              # Главный конфиг
├── model/                   # Конфигурации моделей
│   ├── boosting.yaml
│   ├── mlp.yaml
│   └── random_forest.yaml
├── data/                    # Конфигурации данных
│   ├── data.yaml            # По умолчанию
│   ├── data_small.yaml      # Маленький тестовый датасет
│   ├── data_large.yaml      # Большой датасет
│   ├── data_strict.yaml     # Строгий порог качества (>=8)
│   └── data_lenient.yaml    # Мягкий порог качества (>=6)
├── mlflow/                  # Конфигурации MLflow
│   └── mlflow.yaml
├── paths/                   # Конфигурации путей
│   └── paths.yaml
└── env/                     # Конфигурации окружений
    ├── dev.yaml             # Разработка
    ├── prod.yaml            # Продакшн
    └── test.yaml            # Тестирование
```

## Примеры использования

### Базовое использование

```bash
# Обучение с конфигом по умолчанию
make train

# Или напрямую
poetry run python src/wine_quality/main.py
```

### Переключение между моделями

```bash
# Random Forest
make train-rf
# или
poetry run python src/wine_quality/main.py model=random_forest model_type=random_forest

# Gradient Boosting
make train-boosting
# или
poetry run python src/wine_quality/main.py model=boosting model_type=boosting

# MLP
make train-mlp
# или
poetry run python src/wine_quality/main.py model=mlp model_type=mlp
```

### Переключение между датасетами

```bash
# Маленький датасет (быстрое тестирование)
poetry run python src/wine_quality/main.py data=data_small

# Большой датасет (больше тестовых данных)
poetry run python src/wine_quality/main.py data=data_large

# Строгий порог качества (>=8)
poetry run python src/wine_quality/main.py data=data_strict

# Мягкий порог качества (>=6)
poetry run python src/wine_quality/main.py data=data_lenient
```

### Переключение между окружениями

```bash
# Окружение разработки (без MLflow)
poetry run python src/wine_quality/main.py env=dev

# Продакшн окружение (с полным логированием)
poetry run python src/wine_quality/main.py env=prod

# Тестовое окружение (минимальное логирование)
poetry run python src/wine_quality/main.py env=test
```

### Комбинирование конфигураций

**Для Linux/Mac (Bash):**
```bash
# Random Forest в dev окружении с маленьким датасетом
poetry run python src/wine_quality/main.py \
    model=random_forest \
    model_type=random_forest \
    data=data_small \
    env=dev

# Boosting в prod с большим датасетом и строгим порогом
poetry run python src/wine_quality/main.py \
    model=boosting \
    model_type=boosting \
    data=data_strict \
    env=prod
```

**Для Windows PowerShell:**
```powershell
# Random Forest в dev окружении с маленьким датасетом
poetry run python src/wine_quality/main.py model=random_forest model_type=random_forest data=data_small env=dev

# Или с обратными кавычками
poetry run python src/wine_quality/main.py `
    model=random_forest `
    model_type=random_forest `
    data=data_small `
    env=dev

# Boosting в prod с большим датасетом и строгим порогом
poetry run python src/wine_quality/main.py model=boosting model_type=boosting data=data_strict env=prod
```

**Использование через Makefile:**
```bash
# Используйте готовые команды или ARGS
make train-dev-rf
make train-prod-boosting

# Или с дополнительными параметрами
make train ARGS="model=random_forest data=data_small env=dev model.n_estimators=200"
```

### Переопределение параметров через командную строку

**Для Linux/Mac (Bash):**
```bash
# Изменить параметры модели
poetry run python src/wine_quality/main.py \
    model.n_estimators=200 \
    model.max_depth=5 \
    model.learning_rate=0.05

# Изменить параметры данных
poetry run python src/wine_quality/main.py \
    data.test_size=0.25 \
    data.quality_threshold=8

# Комбинированное переопределение
poetry run python src/wine_quality/main.py \
    model=boosting \
    model.n_estimators=300 \
    data=data_large \
    env=prod \
    random_state=123
```

**Для Windows PowerShell:**
```powershell
# Изменить параметры модели (в одну строку)
poetry run python src/wine_quality/main.py model.n_estimators=200 model.max_depth=5 model.learning_rate=0.05

# Или с использованием обратных кавычек для продолжения строки
poetry run python src/wine_quality/main.py `
    model.n_estimators=200 `
    model.max_depth=5 `
    model.learning_rate=0.05

# Изменить параметры данных
poetry run python src/wine_quality/main.py data.test_size=0.25 data.quality_threshold=8

# Комбинированное переопределение
poetry run python src/wine_quality/main.py model=boosting model.n_estimators=300 data=data_large env=prod random_state=123
```

**Использование через Makefile (работает на всех платформах):**
```bash
# Переопределение через ARGS
make train ARGS="model.n_estimators=200 model.max_depth=5 model.learning_rate=0.05"

# Комбинированное переопределение
make train ARGS="model=boosting model.n_estimators=300 data=data_large env=prod"
```

## Валидация конфигураций

Все конфигурации автоматически валидируются с помощью Pydantic при запуске. Схемы валидации находятся в `src/wine_quality/config_schema.py`.

### Проверки валидации:

- **Типы данных**: автоматическая проверка типов
- **Диапазоны значений**:
  - `n_estimators`: 1-10000
  - `max_depth`: 1-100
  - `learning_rate`: 0.0-1.0
  - `test_size`: 0.0-1.0
  - `quality_threshold`: 1-10
- **Полнота конфигурации**: проверка наличия всех необходимых параметров для выбранного типа модели
- **Корректность значений**: проверка логической корректности (например, размеры скрытых слоев должны быть положительными)

## Иерархия конфигураций

Конфигурации загружаются в следующем порядке:

1. Базовые значения из `config.yaml`
2. Значения из выбранных групп (model, data, mlflow, paths, env)
3. Переопределения через командную строку

Переопределения через командную строку имеют наивысший приоритет.
