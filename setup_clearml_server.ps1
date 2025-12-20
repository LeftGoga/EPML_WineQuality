# PowerShell скрипт для установки ClearML Server на Windows

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Установка ClearML Server" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Проверка Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Ошибка: Docker не установлен. Установите Docker Desktop и повторите попытку." -ForegroundColor Red
    exit 1
}

# Проверка Docker Compose
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "Ошибка: Docker Compose не установлен. Установите Docker Desktop и повторите попытку." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Docker и Docker Compose установлены" -ForegroundColor Green

# Создание директории для сервера
$ServerDir = "clearml-server"
if (-not (Test-Path $ServerDir)) {
    New-Item -ItemType Directory -Path $ServerDir | Out-Null
    Write-Host "✓ Создана директория $ServerDir" -ForegroundColor Green
}

Set-Location $ServerDir

# Копирование docker-compose файла
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "Скачивание docker-compose.yml..." -ForegroundColor Yellow

    # Проверяем наличие локального файла
    if (Test-Path "..\docker-compose.clearml.yml") {
        Copy-Item "..\docker-compose.clearml.yml" -Destination "docker-compose.yml"
        Write-Host "✓ Использован локальный docker-compose файл" -ForegroundColor Green
    } else {
        # Пытаемся скачать
        try {
            Invoke-WebRequest -Uri "https://raw.githubusercontent.com/allegroai/clearml-server/master/docker/docker-compose.yml" -OutFile "docker-compose.yml"
            Write-Host "✓ docker-compose.yml скачан" -ForegroundColor Green
        } catch {
            Write-Host "Не удалось скачать docker-compose.yml. Создайте его вручную." -ForegroundColor Red
            exit 1
        }
    }
}

# Создание .env файла, если его нет
if (-not (Test-Path ".env")) {
    Write-Host "Создание .env файла..." -ForegroundColor Yellow

    # Генерация случайных паролей
    function Generate-Password {
        $chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        $password = ""
        for ($i = 0; $i -lt 25; $i++) {
            $password += $chars[(Get-Random -Maximum $chars.Length)]
        }
        return $password
    }

    $mongoPassword = Generate-Password
    $redisPassword = Generate-Password
    $esPassword = Generate-Password

    @"
# ClearML Server Configuration
CLEARML_HOST_IP=0.0.0.0
CLEARML_PORT=8080

# MongoDB
MONGO_ROOT_PASSWORD=$mongoPassword
MONGO_DATA_PATH=./mongo_data

# Redis
REDIS_PASSWORD=$redisPassword
REDIS_DATA_PATH=./redis_data

# Elasticsearch
ELASTICSEARCH_PASSWORD=$esPassword
ELASTICSEARCH_DATA_PATH=./elasticsearch_data

# File Server
FILESERVER_PORT=8081
FILESERVER_DATA_PATH=./fileserver_data
"@ | Out-File -FilePath ".env" -Encoding utf8

    Write-Host "✓ .env файл создан с случайными паролями" -ForegroundColor Green
    Write-Host ""
    Write-Host "ВАЖНО: Сохраните пароли из файла .env для дальнейшего использования!" -ForegroundColor Yellow
    Write-Host ""
}

# Запуск сервера
Write-Host "Запуск ClearML Server..." -ForegroundColor Yellow
docker-compose up -d

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "ClearML Server запущен!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Веб-интерфейс: http://localhost:8080" -ForegroundColor Cyan
Write-Host "API: http://localhost:8008" -ForegroundColor Cyan
Write-Host "Файловый сервер: http://localhost:8081" -ForegroundColor Cyan
Write-Host ""
Write-Host "Проверка статуса: docker-compose ps" -ForegroundColor Yellow
Write-Host "Просмотр логов: docker-compose logs -f" -ForegroundColor Yellow
Write-Host "Остановка: docker-compose down" -ForegroundColor Yellow
Write-Host ""
Write-Host "При первом запуске откройте http://localhost:8080" -ForegroundColor Green
Write-Host "и создайте администратора." -ForegroundColor Green
Write-Host ""
