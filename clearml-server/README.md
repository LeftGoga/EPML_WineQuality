# ClearML Server - Docker Compose

Этот каталог содержит конфигурацию Docker Compose для запуска локального сервера ClearML.

## Быстрый старт

1. **Запуск сервера:**
   ```bash
   docker-compose up -d
   ```

2. **Проверка статуса:**
   ```bash
   docker-compose ps
   ```

3. **Просмотр логов:**
   ```bash
   docker-compose logs -f
   ```

4. **Остановка сервера:**
   ```bash
   docker-compose down
   ```

## Доступ к сервисам

После запуска сервер будет доступен по следующим адресам:

- **Веб-интерфейс**: http://localhost:8080
- **API**: http://localhost:8008
- **Файловый сервер**: http://localhost:8081

## Первый запуск

При первом запуске откройте http://localhost:8080 и создайте администратора.

## Настройка

### Изменение паролей

1. Скопируйте `.env.example` в `.env`:
   ```bash
   cp .env.example .env
   ```

2. Отредактируйте `.env` и измените пароли

3. Обновите `docker-compose.yml` с новыми значениями

### Изменение портов

Если порты заняты, измените их в `docker-compose.yml`:

```yaml
ports:
  - "9080:8080"  # Измените внешний порт
```

## Управление данными

### Резервное копирование

```bash
# MongoDB
docker-compose exec mongo mongodump --out /backup

# Elasticsearch
docker-compose exec elasticsearch curl -X POST "localhost:9200/_snapshot/backup_repo/_all?wait_for_completion=true"
```

### Очистка данных

⚠️ **ВНИМАНИЕ**: Это удалит все данные!

```bash
docker-compose down -v
```

## Устранение проблем

### Проблема: Порт уже занят

Измените порты в `docker-compose.yml` или остановите конфликтующий сервис.

### Проблема: Недостаточно памяти

Увеличьте лимиты памяти для Elasticsearch в `docker-compose.yml`:

```yaml
environment:
  - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
```

### Проблема: Сервер не запускается

1. Проверьте логи:
   ```bash
   docker-compose logs clearml-server
   ```

2. Проверьте статус всех сервисов:
   ```bash
   docker-compose ps
   ```

3. Пересоздайте контейнеры:
   ```bash
   docker-compose up -d --force-recreate
   ```

## Структура сервисов

- **mongo**: База данных MongoDB для хранения метаданных
- **redis**: Кэш и очередь задач
- **elasticsearch**: Поиск и индексация
- **clearml-server**: Основной сервер ClearML

## Дополнительная информация

- [Официальная документация ClearML Server](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server/)
- [GitHub репозиторий](https://github.com/allegroai/clearml-server)
