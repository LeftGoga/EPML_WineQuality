# EPML

Описание:
Проект для анализа и моделирования качества вина (Wine Quality dataset).

## Структура
-src/data.py - загрузка данных
-src/config.py - конфигурация экспериментов
-src/features.py - генерацией фичей
-src/models.py - обучение и оценка омделей
-src/utils.py - работа с графиками

## Быстрый старт
1. Сгенерировать проект:
cookiecutter https://github.com/LeftGoga/EPML_DS_COOCKIECUTTER.git


2. Установить зависимости (Poetry):
make install

3. Запустить Обучение:
make run

4. Запустить стримлит
make streamlit

5. Запустить в докере
docker compose up --build


Автор: Egor Polyakov — leftgoga@gmail.com
