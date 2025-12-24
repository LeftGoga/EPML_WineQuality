# Быстрый старт

Это руководство поможет вам быстро начать работу с проектом.

## Предварительные требования

- Python 3.12 или выше
- Poetry 2.1.4 или выше
- Git

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/LeftGoga/wine_quality_epml.git
cd wine_quality_epml
```

### 2. Установка зависимостей

```bash
# Установка Poetry (если не установлен)
curl -sSL https://install.python-poetry.org | python3 -

# Установка зависимостей проекта
poetry install

# Активация виртуального окружения
poetry shell
```

Или используйте Make:

```bash
make install
```

### 3. Настройка окружения

Создайте файл `.env` в корне проекта:

```bash
# ClearML конфигурация
CLEARML_API_ACCESS_KEY=your_access_key
CLEARML_API_SECRET_KEY=your_secret_key
CLEARML_WEB_HOST=http://localhost:8080
CLEARML_API_HOST=http://localhost:8008
```

## Запуск

### Запуск обучения модели

```bash
# Базовый запуск
poetry run python run_clearml_experiment.py

# С параметрами
poetry run python run_clearml_experiment.py \
    --model-type boosting \
    --boosting-n-estimators 200 \
    --boosting-max-depth 5
```

Или используйте Make:

```bash
make run
```

### Запуск пайплайна

```bash
poetry run python run_clearml_pipeline.py
```

### Запуск Streamlit приложения

```bash
poetry run streamlit run src/wine_quality/app.py
```

Или:

```bash
make streamlit
```

## Проверка работы

После запуска обучения вы должны увидеть:

1. Логи процесса обучения
2. Сохраненную модель в `models/`
3. Графики в `plots/`
4. Эксперимент в ClearML UI (если настроен)

## Следующие шаги

- 📖 Изучите [Примеры использования](examples.md)
- 🔧 Ознакомьтесь с [Руководством по развертыванию](deployment.md)
- 📚 Прочитайте [API документацию](api/index.md)
