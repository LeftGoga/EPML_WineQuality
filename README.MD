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
docker compose up --build -> open localhost:8501

6. Версии пакетов и программ
Poetry 2.1.4
mypy 1.18.2
ruff 0.14.5
docker 27.2.0
pre-commit 4.4.0
```
python = ">=3.12"
pre-commit = ">=4.4.0,<5.0.0"
pandas = ">=2.3.3,<3.0.0"
scikit-learn = ">=1.7.2,<2.0.0"
matplotlib = ">=3.10.7,<4.0.0"
seaborn = ">=0.13.2,<0.14.0"
joblib = ">=1.5.2,<2.0.0"
streamlit = "^1.51.0"
bandit = "^1.9.1"
```

ветвление с помощью github-flow:

    --master - защищенная ветка, в которой находится проверенный рабочий код

    --features/hw* - ветки фичей, который в данном случае будут представлять собой домашнии работы

    -- Мердж в мастер через пуллреквес


Автор: Egor Polyakov — leftgoga@gmail.com
