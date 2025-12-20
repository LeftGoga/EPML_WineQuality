#!/bin/bash

# Скрипт для быстрой установки ClearML Server через Docker Compose

set -e

echo "=========================================="
echo "Установка ClearML Server"
echo "=========================================="

# Проверка Docker
if ! command -v docker &> /dev/null; then
    echo "Ошибка: Docker не установлен. Установите Docker и повторите попытку."
    exit 1
fi

# Проверка Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Ошибка: Docker Compose не установлен. Установите Docker Compose и повторите попытку."
    exit 1
fi

echo "✓ Docker и Docker Compose установлены"

# Создание директории для сервера
SERVER_DIR="clearml-server"
if [ ! -d "$SERVER_DIR" ]; then
    mkdir -p "$SERVER_DIR"
    echo "✓ Создана директория $SERVER_DIR"
fi

cd "$SERVER_DIR"

# Копирование docker-compose файла
if [ ! -f "docker-compose.yml" ]; then
    echo "Скачивание docker-compose.yml..."
    curl -L -o docker-compose.yml https://raw.githubusercontent.com/allegroai/clearml-server/master/docker/docker-compose.yml || {
        echo "Не удалось скачать docker-compose.yml. Используем локальный файл..."
        if [ -f "../docker-compose.clearml.yml" ]; then
            cp ../docker-compose.clearml.yml docker-compose.yml
        else
            echo "Ошибка: не найден docker-compose файл"
            exit 1
        fi
    }
    echo "✓ docker-compose.yml создан"
fi

# Создание .env файла, если его нет
if [ ! -f ".env" ]; then
    echo "Создание .env файла..."
    cat > .env <<EOF
# ClearML Server Configuration
CLEARML_HOST_IP=0.0.0.0
CLEARML_PORT=8080

# MongoDB
MONGO_ROOT_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
MONGO_DATA_PATH=./mongo_data

# Redis
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
REDIS_DATA_PATH=./redis_data

# Elasticsearch
ELASTICSEARCH_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
ELASTICSEARCH_DATA_PATH=./elasticsearch_data

# File Server
FILESERVER_PORT=8081
FILESERVER_DATA_PATH=./fileserver_data
EOF
    echo "✓ .env файл создан с случайными паролями"
    echo ""
    echo "ВАЖНО: Сохраните пароли из файла .env для дальнейшего использования!"
    echo ""
fi

# Запуск сервера
echo "Запуск ClearML Server..."
docker-compose up -d

echo ""
echo "=========================================="
echo "ClearML Server запущен!"
echo "=========================================="
echo ""
echo "Веб-интерфейс: http://localhost:8080"
echo "API: http://localhost:8008"
echo "Файловый сервер: http://localhost:8081"
echo ""
echo "Проверка статуса: docker-compose ps"
echo "Просмотр логов: docker-compose logs -f"
echo "Остановка: docker-compose down"
echo ""
echo "При первом запуске откройте http://localhost:8080"
echo "и создайте администратора."
echo ""
