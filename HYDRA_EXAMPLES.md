# Примеры запуска кода с Hydra

Этот документ содержит примеры использования Hydra для запуска проекта wine_quality_epml.

## Базовые примеры

### 1. Простой запуск с конфигурацией по умолчанию

```bash
poetry run python src/wine_quality/main.py
```

Или через make:
```bash
make train
```

### 2. Просмотр конфигурации без запуска

```bash
poetry run python src/wine_quality/main.py --cfg job
```

Показывает полную конфигурацию, которая будет использована.

### 3. Просмотр структуры конфигурации

```bash
poetry run python src/wine_quality/main.py --cfg job --resolve
```

Показывает конфигурацию с разрешенными значениями.

## Переопределение параметров через командную строку

### 4. Изменение типа модели

```bash
poetry run python src/wine_quality/main.py model_type=random_forest
```

```bash
poetry run python src/wine_quality/main.py model_type=boosting
```

```bash
poetry run python src/wine_quality/main.py model_type=mlp
```

### 5. Изменение параметров модели

```bash
# Random Forest с большим количеством деревьев
# ВАЖНО: сначала выбрать модель, затем параметры
poetry run python src/wine_quality/main.py model_type=random_forest model=random_forest model.n_estimators=500 model.max_depth=10

# Boosting с измененным learning rate
poetry run python src/wine_quality/main.py model_type=boosting model=boosting model.learning_rate=0.2 model.n_estimators=300

# MLP с другой архитектурой
# ВАЖНО: сначала нужно выбрать модель MLP, затем устанавливать параметры
poetry run python src/wine_quality/main.py model_type=mlp model=mlp model.hidden_layer_sizes=[200,100,50] model.max_iter=500

# Или использовать предопределенную конфигурацию и переопределить параметры
poetry run python src/wine_quality/main.py model_type=mlp model=mlp_large model.max_iter=500
```

**Важное примечание:** При переопределении параметров модели через командную строку, сначала нужно выбрать соответствующую конфигурацию модели (`model=mlp`, `model=random_forest`, и т.д.), иначе Hydra не позволит устанавливать параметры, которые отсутствуют в текущей конфигурации. Например, `hidden_layer_sizes` доступен только для MLP, поэтому при установке этого параметра нужно сначала выбрать `model=mlp`.

### 6. Изменение параметров данных

```bash
# Изменение размера тестовой выборки
poetry run python src/wine_quality/main.py data.test_size=0.3

# Изменение порога качества вина
poetry run python src/wine_quality/main.py data.quality_threshold=6

# Комбинация параметров
poetry run python src/wine_quality/main.py data.test_size=0.25 data.quality_threshold=6
```

## Использование предопределенных конфигураций

### 7. Выбор конфигурации модели

```bash
# Быстрая конфигурация Random Forest
poetry run python src/wine_quality/main.py model=random_forest_fast

# Глубокая конфигурация Random Forest
poetry run python src/wine_quality/main.py model=random_forest_deep

# Быстрая конфигурация Boosting
poetry run python src/wine_quality/main.py model=boosting_fast model_type=boosting

# Глубокая конфигурация Boosting
poetry run python src/wine_quality/main.py model=boosting_deep model_type=boosting

# Маленькая MLP
poetry run python src/wine_quality/main.py model=mlp_small model_type=mlp

# Большая MLP
poetry run python src/wine_quality/main.py model=mlp_large model_type=mlp
```

### 8. Выбор конфигурации данных

```bash
# Маленький датасет
poetry run python src/wine_quality/main.py data=data_small

# Большой датасет
poetry run python src/wine_quality/main.py data=data_large

# Строгий порог качества
poetry run python src/wine_quality/main.py data=data_strict

# Мягкий порог качества
poetry run python src/wine_quality/main.py data=data_lenient
```

### 9. Выбор окружения

```bash
# Development окружение (без MLflow)
poetry run python src/wine_quality/main.py env=dev

# Production окружение
poetry run python src/wine_quality/main.py env=prod

# Test окружение
poetry run python src/wine_quality/main.py env=test
```

## Комбинированные примеры

### 10. Комбинация модели и данных

```bash
poetry run python src/wine_quality/main.py model=random_forest_deep data=data_large
```

```bash
poetry run python src/wine_quality/main.py model=boosting_fast data=data_small model_type=boosting
```

### 11. Комбинация модели, данных и окружения

```bash
poetry run python src/wine_quality/main.py model=random_forest env=dev data=data_small model_type=random_forest
```

```bash
poetry run python src/wine_quality/main.py model=boosting_deep env=prod data=data_large model_type=boosting
```

### 12. Переопределение нескольких параметров

```bash
poetry run python src/wine_quality/main.py \
  model_type=random_forest \
  model.n_estimators=300 \
  model.max_depth=5 \
  data.test_size=0.25 \
  data.quality_threshold=6 \
  random_state=123
```

## Отключение функций

### 13. Запуск без MLflow

```bash
poetry run python src/wine_quality/main.py no_mlflow=true
```

### 14. Запуск без сохранения модели локально

```bash
poetry run python src/wine_quality/main.py no_save=true
```

### 15. Запуск с анализом экспериментов

```bash
poetry run python src/wine_quality/main.py analyze_experiments=true
```

## Мульти-запуски (Sweeps)

### 16. Запуск с несколькими значениями параметра

```bash
poetry run python src/wine_quality/main.py -m \
  model_type=random_forest,boosting,mlp
```

### 17. Сетка параметров

```bash
poetry run python src/wine_quality/main.py -m \
  model_type=random_forest \
  model.n_estimators=100,200,300 \
  model.max_depth=3,5,7
```

### 18. Комбинация параметров

```bash
poetry run python src/wine_quality/main.py -m \
  model_type=random_forest,boosting \
  model.n_estimators=200,300 \
  data.test_size=0.2,0.3
```

## Продвинутые примеры

### 19. Сохранение конфигурации в файл

```bash
poetry run python src/wine_quality/main.py \
  --config-path=./custom_configs \
  --config-name=my_config \
  model_type=random_forest
```

### 20. Использование кастомного конфигурационного файла

```bash
poetry run python src/wine_quality/main.py \
  --config-path=./conf \
  --config-name=config \
  model=random_forest_deep \
  data=data_large
```

### 21. Экспорт конфигурации

```bash
poetry run python src/wine_quality/main.py \
  --cfg job \
  > my_config.yaml
```

### 22. Запуск с логированием в MLflow и анализом

```bash
poetry run python src/wine_quality/main.py \
  model_type=boosting \
  model.n_estimators=500 \
  analyze_experiments=true \
  export_comparison=./outputs/comparison.csv
```

## Примеры через Makefile

После добавления команд в Makefile, можно использовать:

```bash
# Показать конфигурацию
make hydra-show-config

# Показать структуру
make hydra-show-structure

# Запуск с Random Forest
make hydra-rf

# Запуск с быстрой конфигурацией
make hydra-rf-fast

# Запуск с глубокой конфигурацией
make hydra-rf-deep

# Запуск с Boosting
make hydra-boosting

# Запуск с MLP
make hydra-mlp

# Запуск с разными данными
make hydra-data-small
make hydra-data-large

# Запуск в разных окружениях
make hydra-env-dev
make hydra-env-prod

# Комбинированные примеры
make hydra-override-model
make hydra-override-data
make hydra-override-combined
```

## Полезные флаги Hydra

- `--cfg job` - показать конфигурацию
- `--cfg job --resolve` - показать конфигурацию с разрешенными значениями
- `-m` или `--multirun` - запуск множественных экспериментов
- `--config-path` - путь к директории с конфигурациями
- `--config-name` - имя конфигурационного файла
- `--hydra-help` - показать справку по Hydra

## Примеры для отладки

### 23. Просмотр только конфигурации модели

```bash
poetry run python src/wine_quality/main.py --cfg model
```

### 24. Просмотр только конфигурации данных

```bash
poetry run python src/wine_quality/main.py --cfg data
```

### 25. Сухой прогон (dry run)

```bash
poetry run python src/wine_quality/main.py --cfg job --dry-run
```

## Примечания

- Все пути относительно корня проекта
- Конфигурации находятся в `conf/`
- Результаты сохраняются в `outputs/` (если не отключено)
- MLflow логирование включено по умолчанию (можно отключить через `no_mlflow=true`)
