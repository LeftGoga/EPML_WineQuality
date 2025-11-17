# EPML_WINE_QUALITY

Описание:
Проект для анализа и моделирования качества вина (Wine Quality dataset).

## Структура
- `data/` — raw и processed
- `notebooks/` — Jupyter ноутбуки
- `src/wine_quality/` — исходники
- `tests/` — тесты

## Быстрый старт
1. Сгенерировать проект:
cookiecutter https://github.com/LeftGoga/EPML_DS_COOCKIECUTTER.git

2. Установить зависимости
make install
3. Запустить обучение:
make run

4. Общая структура
data.py - загрузка и предобработка данныз
features.py - выделение фич и дополнительная предобработка
mode.py - обучение модели
utils.py - дополнительные компонетнты



Автор: Egor Polyakov — leftgoga@gmail.com
