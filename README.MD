# EPML - Wine Quality ML Project

Проект для анализа и моделирования качества вина (Wine Quality dataset).

## Структура

- `src/wine_quality/` - основной код проекта
  - `data.py` - загрузка данных
  - `config.py` - конфигурация экспериментов
  - `features.py` - генерация фичей
  - `model.py` - обучение и оценка моделей
  - `clearml_*.py` - интеграция с ClearML
- `docs/` - документация (MkDocs)
- `conf/` - конфигурационные файлы
- `scripts/` - вспомогательные скрипты
- `hw_reports/` - отчеты об экспериментах

## Быстрый старт

1. **Установить зависимости (Poetry):**
   ```bash
   make install
   ```

2. **Запустить обучение:**
   ```bash
   make run
   ```

3. **Запустить Streamlit приложение:**
   ```bash
   make streamlit
   ```

4. **Запустить в Docker:**
   ```bash
   docker compose up --build
   # Открыть http://localhost:8501
   ```

5. **Просмотреть документацию:**
   ```bash
   make docs-serve
   # Открыть http://127.0.0.1:8000
   ```

## Документация

Полная документация доступна:
- **Локально:** `make docs-serve` (http://127.0.0.1:8000)
- **Онлайн:** [GitHub Pages](https://leftgoga.github.io/wine_quality_epml/)

Документация включает:
- Руководство по установке и развертыванию
- Примеры использования
- API документация
- Отчеты об экспериментах

## Генерация отчетов об экспериментах

```bash
# Генерация отчета по эксперименту
poetry run python scripts/generate_experiment_report.py \
    --task-id <task_id> \
    --output hw_reports/experiment_report.md

# Сравнение экспериментов
poetry run python scripts/generate_experiment_report.py \
    --comparison-only <task_id_1> <task_id_2> \
    --output hw_reports/comparison_report.md
```

## Версии пакетов

- Poetry: 2.1.4
- Python: >=3.12
- mypy: 1.18.2
- ruff: 0.14.5
- docker: 27.2.0
- pre-commit: 4.4.0

Основные зависимости:
- pandas >= 2.3.3
- scikit-learn >= 1.7.2
- matplotlib >= 3.10.7
- seaborn >= 0.13.2
- clearml >= 1.15.0
- streamlit ^1.51.0

## Автор

**Egor Polyakov** — leftgoga@yandex.ru
