# Установка

Подробная инструкция по установке и настройке проекта.

## Системные требования

- **Python**: 3.12 или выше
- **Poetry**: 2.1.4 или выше
- **ОС**: Windows, Linux, macOS
- **Память**: минимум 4 GB RAM
- **Диск**: минимум 1 GB свободного места

## Установка Python

### Windows

1. Скачайте Python с [python.org](https://www.python.org/downloads/)
2. Установите, убедитесь что выбрали "Add Python to PATH"
3. Проверьте установку:

```bash
python --version
```

### Linux

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

### macOS

```bash
brew install python@3.12
```

## Установка Poetry

### Windows/Linux/macOS

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Добавьте Poetry в PATH (для Windows используйте PowerShell):

```bash
# Windows PowerShell
$env:Path += ";$env:APPDATA\Python\Scripts"

# Linux/macOS
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Проверьте установку:

```bash
poetry --version
```

## Установка проекта

### 1. Клонирование репозитория

```bash
git clone https://github.com/LeftGoga/wine_quality_epml.git
cd wine_quality_epml
```

### 2. Установка зависимостей

```bash
poetry install
```

Это установит все зависимости, указанные в `pyproject.toml`:

- pandas >= 2.3.3
- scikit-learn >= 1.7.2
- matplotlib >= 3.10.7
- seaborn >= 0.13.2
- clearml >= 1.15.0
- и другие...

### 3. Активация виртуального окружения

```bash
poetry shell
```

Или запускайте команды через `poetry run`:

```bash
poetry run python script.py
```

## Настройка ClearML (опционально)

### Локальный сервер

1. Запустите ClearML Server через Docker:

```bash
cd clearml-server
docker-compose up -d
```

2. Откройте веб-интерфейс: http://localhost:8080

3. Создайте учетные данные в UI

4. Настройте клиент:

```bash
clearml-init
```

Введите данные из веб-интерфейса.

### Облачный сервер

1. Зарегистрируйтесь на [app.clear.ml](https://app.clear.ml)
2. Получите API ключи
3. Настройте клиент:

```bash
clearml-init
```

## Проверка установки

```bash
# Проверка Python
python --version

# Проверка Poetry
poetry --version

# Проверка зависимостей
poetry check

# Запуск тестов (если есть)
poetry run pytest
```

## Устранение проблем

### Проблема: Poetry не найден

**Решение**: Убедитесь, что Poetry добавлен в PATH

### Проблема: Ошибки при установке зависимостей

**Решение**:
- Обновите Poetry: `poetry self update`
- Очистите кэш: `poetry cache clear pypi --all`
- Переустановите: `poetry install --no-cache`

### Проблема: Ошибки с ClearML

**Решение**:
- Проверьте, что сервер запущен
- Проверьте файл `.env` с правильными ключами
- Убедитесь, что порты 8080 и 8008 свободны

## Следующие шаги

После успешной установки:

- 📖 Перейдите к [Быстрому старту](quickstart.md)
- 💡 Изучите [Примеры использования](examples.md)
- 🚀 Ознакомьтесь с [Руководством по развертыванию](deployment.md)
