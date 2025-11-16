# {{cookiecutter.project_name}}

Описание:
Проект для анализа и моделирования качества вина (Wine Quality dataset).

## Структура
- `data/` — raw и processed
- `notebooks/` — Jupyter ноутбуки
- `src/{{cookiecutter.package_name}}/` — исходники
- `tests/` — тесты

## Быстрый старт
1. Сгенерировать проект:
cookiecutter /path/to/cookiecutter_template


2. Установить зависимости (Poetry):
cd {{cookiecutter.project_slug}}
poetry install
poetry shell

3. Запустить ноутбук:
make notebook


Автор: {{cookiecutter.author_name}} — {{cookiecutter.email}}