# Руководство по развертыванию

Это руководство описывает различные способы развертывания проекта.

## Развертывание в Docker

### Предварительные требования

- Docker 27.2.0 или выше
- Docker Compose

### Быстрый старт

1. Соберите образ:

```bash
docker compose build
```

2. Запустите контейнеры:

```bash
docker compose up -d
```

3. Откройте Streamlit приложение:

```
http://localhost:8501
```

### Конфигурация Docker

Основной Dockerfile находится в корне проекта:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Установка зависимостей
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Копирование кода
COPY . .

# Запуск приложения
CMD ["streamlit", "run", "src/wine_quality/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Развертывание ClearML Server

### Локальное развертывание

1. Перейдите в директорию `clearml-server`:

```bash
cd clearml-server
```

2. Настройте переменные окружения в `docker-compose.yml`:

```yaml
environment:
  - ADMIN_PASSWORD=your_password
```

3. Запустите сервер:

```bash
docker-compose up -d
```

4. Проверьте статус:

```bash
docker-compose ps
```

5. Откройте веб-интерфейс:

```
http://localhost:8080
```

### Производственное развертывание

Для продакшена рекомендуется:

1. Использовать внешнюю базу данных MongoDB
2. Настроить SSL/TLS
3. Использовать обратный прокси (nginx)
4. Настроить резервное копирование

## Развертывание в облаке

### AWS

1. Создайте EC2 инстанс
2. Установите Docker и Docker Compose
3. Клонируйте репозиторий
4. Настройте security groups для портов 8501, 8080, 8008
5. Запустите контейнеры

### Google Cloud Platform

1. Создайте Compute Engine VM
2. Установите Docker
3. Разверните приложение аналогично AWS

### Azure

1. Создайте Azure Container Instances
2. Загрузите образ в Azure Container Registry
3. Разверните контейнер

## CI/CD с GitHub Actions

Проект включает автоматическую настройку CI/CD через GitHub Actions.

### Настройка

1. Создайте секреты в GitHub:
   - `CLEARML_API_ACCESS_KEY`
   - `CLEARML_API_SECRET_KEY`
   - `CLEARML_WEB_HOST`
   - `CLEARML_API_HOST`

2. При каждом push в `main` автоматически:
   - Запускаются тесты
   - Собирается документация
   - Публикуется на GitHub Pages

## Мониторинг и логирование

### Логирование

Логи сохраняются в:
- `logs/` - локальные логи
- ClearML UI - логи экспериментов

### Мониторинг

Используйте ClearML для мониторинга:
- Метрики моделей
- Статус экспериментов
- Использование ресурсов

## Резервное копирование

### Данные для резервного копирования

1. **Модели**: `models/`
2. **Данные**: `data/`
3. **Конфигурации**: `conf/`
4. **ClearML данные**: тома Docker `clearml_data`, `mongo_data`

### Автоматическое резервное копирование

Настройте cron job или scheduled task:

```bash
# Пример скрипта резервного копирования
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf backup_$DATE.tar.gz models/ data/ conf/
```

## Безопасность

### Рекомендации

1. **Пароли**: Используйте сильные пароли для всех сервисов
2. **SSL/TLS**: Настройте HTTPS для продакшена
3. **Firewall**: Ограничьте доступ к портам
4. **Secrets**: Храните секреты в переменных окружения, не в коде
5. **Обновления**: Регулярно обновляйте зависимости

### Проверка безопасности

```bash
# Проверка с помощью bandit
poetry run bandit -r src/
```

## Масштабирование

### Горизонтальное масштабирование

Для увеличения нагрузки:

1. Запустите несколько инстансов Streamlit
2. Используйте load balancer
3. Настройте ClearML агентов для распределенной обработки

### Вертикальное масштабирование

Увеличьте ресурсы сервера:
- CPU
- RAM
- Диск

## Откат (Rollback)

### Откат версии модели

1. Используйте версионирование в ClearML
2. Загрузите предыдущую версию модели
3. Обновите конфигурацию

### Откат кода

```bash
git checkout <previous-commit>
docker compose build
docker compose up -d
```

## Следующие шаги

- 📖 Изучите [Примеры использования](examples.md)
- 📚 Прочитайте [API документацию](api/index.md)
- 🔧 Настройте мониторинг и алерты
