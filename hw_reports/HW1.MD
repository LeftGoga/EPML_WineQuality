# Отчет по проделанной работе

## 1. Создание шаблона проекта

### a)
Создан собственный шаблон на основе облегчённого DS-шаблона:
https://github.com/LeftGoga/EPML_DS_COOCKIECUTTER.git

### b)
Создан README с описанием основных команд и структуры проекта.

### Структура проекта
```
│   cookiecutter.json
│
└───{{ cookiecutter.project_slug }}
    │   .envexample
    │   .pre-commit-config.yaml
    │   dockerfile
    │   Makefile
    │   poetry.lock
    │   pyproject.toml
    │   README.MD
    │
    ├───.github
    │       .gitkeep
    │
    ├───data
    │       .gitkeep
    │
    ├───models
    │       .gitkeep
    │
    ├───notebooks
    │       .gitkeep
    │
    ├───plots
    │       .gitkeep
    │
    ├───src
    │   └───{{cookiecutter.package_name}}
    │           data.py
    │           features.py
    │           model.py
    │           utils.py
    │           __init__.py
    │           __main__.py
    │
    └───tests
            .gitkeep

```


---

## 2. Настройка линтеров и форматеров

### a) Добавлены настройки в pyproject.toml

#### mypy
```
python_version = "3.13"
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
strict_equality = true
extra_checks = true
ignore_missing_imports = true
```


#### ruff
```
line-length = 100
target-version = "py313"

select = [
"E", "W", "F",
"I",
"B",
"C4",
"UP",
"ARG",
"PIE",
"PL",
"RUF",
]
ignore = [
"E501",
"B008",
"PLR0913",
"PLR2004",
"RUF001",
"RUF002",
"RUF003"
]

quote-style = "double"
indent-style = "space"

combine-as-imports = true
known-first-party = ["wine_quality"]
```



---

### b) Созданы pre-commit хуки (Ruff, mypy, Bandit, автофиксы)
```
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.6
    hooks:
      - id: bandit
        files: ^src/wine_quality/


```


### c) Добавлен Ruff как форматер(заменяет black и isortб при этом быстрее работает)
### d) Создан Makefile
```
.PHONY: all install run train lint format clean reinstall

all: install run

install:
poetry install --with dev

run train:
poetry run python -m wine_quality

format:
poetry run ruff format src
poetry run ruff check --select I --fix src

lint:
-poetry run ruff check src
-poetry run mypy src

clean:
ifeq ($(OS),Windows_NT)
@if exist models (rmdir /s /q models)
@if exist plots (rmdir /s /q plots)
@if exist .pytest_cache (rmdir /s /q .pytest_cache)
@if exist .mypy_cache (rmdir /s /q .mypy_cache)
@if exist .ruff_cache (rmdir /s /q .ruff_cache)
del /s /q .pyc 2>NUL
for /d /r . %%d in (pycache) do @if exist "%%d" rmdir /s /q "%%d"
@echo "Очистка завершена"
else
rm -rf models/ plots/ pycache/ .pytest_cache/ .mypy_cache/ .ruff_cache/
find . -type f -name ".pyc" -delete
find . -type d -name "pycache" -exec rm -rf {} +
@echo "Очистка завершена"
endif

reinstall:
poetry env remove --all
poetry install --with dev

```


### e) Созданы config.py и .env для конфигураций

---

## 3. Настройка окружения и Docker

Настроен Poetry с зависимостями.
Создан requirements.txt через pip freeze.

### Dockerfile
```
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends
build-essential
gcc
&& rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3
RUN pip install "poetry==$POETRY_VERSION"

ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry install --only main --no-ansi

COPY . .

CMD ["python", "-m", "wine_quality"]
```



---

## 4. Работа с Git

- Создан репозиторий с ветками для ДЗ
- Настроен .gitignore (базовый python template)

---

## 5. Подготовлен отчет
Настоящий документ является финальным отчетом.
