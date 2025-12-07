# Отчет по проделанной работе: Интеграция MLflow

## 1. Настройка выбранного инструмента (4 балла)

### a) Установка и настройка MLflow

Выбранный инструмент: **MLflow** - платформа для управления жизненным циклом машинного обучения.

MLflow установлен через Docker Compose, что обеспечивает изолированное окружение и простоту развертывания.

#### Dockerfile для MLflow сервера

Создан отдельный Dockerfile (`Dockerfile.mlflow`) для MLflow сервера:

```dockerfile
FROM ghcr.io/mlflow/mlflow:v3.7.0

RUN pip install --no-cache-dir psycopg2-binary flask-wtf && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install mlflow[auth]
```

### b) Настройка базы данных и облачного хранилища

Настроена PostgreSQL база данных для хранения метаданных экспериментов и Docker volume для артефактов.

#### Конфигурация в docker-compose.yaml:

```yaml
services:
  postgres:
    image: postgres:15-alpine
    container_name: mlflow_postgres
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    ports:
      - "5432:5432"
    volumes:
      - mlflow_postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow"]
      interval: 10s
      timeout: 5s
      retries: 5

  mlflow:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
    container_name: mlflow_server
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
      --app-name basic-auth
    ports:
      - "5000:5000"
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
      - ./basic_auth.ini:/mlflow/basic_auth.ini
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
      - MLFLOW_FLASK_SERVER_SECRET_KEY=your_super_secret_key
      - MLFLOW_AUTH_CONFIG_PATH=/mlflow/basic_auth.ini

volumes:
  mlflow_postgres_data:
  mlflow_artifacts:
```

**Особенности настройки:**
- PostgreSQL используется как backend store для метаданных экспериментов
- Docker volume `mlflow_artifacts` используется для хранения артефактов (модели, графики, метрики)
- Настроены healthchecks для обеспечения готовности сервисов перед запуском зависимых контейнеров

### c) Создание проекта и экспериментов

Создана система автоматического создания экспериментов через контекстный менеджер `MLflowExperimentContext`:

```python
class MLflowExperimentContext:
    def __init__(
        self,
        experiment_name: str,
        create_if_not_exists: bool = True,
        tags: dict[str, str] | None = None,
    ):
        # Автоматически создает эксперимент, если его нет
        # Восстанавливает удаленные эксперименты
        # Устанавливает теги для экспериментов
```

**Использование:**

```python
from mlflow_context import MLflowExperimentContext

with MLflowExperimentContext("wine_quality_experiments", tags={"team": "ml"}):
    # Эксперимент автоматически создан или выбран
    mlflow.log_param("param1", "value1")
```

**Основной эксперимент проекта:** `wine_quality_experiments` (определен в `config.py`)

### d) Настройка аутентификации и доступа

Настроена базовая HTTP аутентификация через конфигурационный файл `basic_auth.ini`:

```ini
[mlflow]
database_uri = postgresql://mlflow:mlflow@postgres:5432/mlflow

default_permission = READ

admin_username = admin
admin_password = password1234567890
default_admin_username = admin
default_admin_password = password1234567890

authorization_function = mlflow.server.auth:authenticate_request_basic_auth
```

**Интеграция аутентификации в код:**

Создан контекстный менеджер `MLflowTrackingContext` для управления подключением с аутентификацией:

```python
class MLflowTrackingContext:
    def __init__(
        self,
        tracking_uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        # Настраивает tracking URI и credentials
```

**Использование:**

```python
from mlflow_context import MLflowTrackingContext

with MLflowTrackingContext(
    tracking_uri="http://mlflow:5000",
    username="admin",
    password="password1234567890"
):
    # Работа с MLflow с аутентификацией
    mlflow.log_metric("accuracy", 0.95)
```

**Переменные окружения для аутентификации:**
- `MLFLOW_TRACKING_URI` - URI MLflow сервера
- `MLFLOW_TRACKING_USERNAME` - имя пользователя
- `MLFLOW_TRACKING_PASSWORD` - пароль

---

## 2. Проведение экспериментов (4 балла)

### a) Проведение 15+ экспериментов с разными алгоритмами

Реализована функция `run_experiment()` в `model.py`

![alt text](../pics/experiments.png)

![alt text](image.png)

![alt text](image-1.png)

![alt text](../pics/run.png)
### b) Настройка логирования метрик, параметров и артефактов

#### Логирование параметров

Параметры логируются автоматически через декоратор `@log_params` и контекстный менеджер `MLflowRunContext`:

```python
@log_params
def train_model(
    model_type: str | ModelType,
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    # ... другие параметры
):
    # Все параметры автоматически логируются в MLflow
    pass
```

**Логируемые параметры:**
- Тип модели (`model_type`)
- Гиперпараметры модели (n_estimators, max_depth, learning_rate и т.д.)
- Random state
- Версия Python и платформа (автоматически через `MLflowRunContext`)

#### Логирование метрик

Метрики логируются через декоратор `@log_metrics`:

```python
@log_metrics(["accuracy", "f1"])
def evaluate_model(model, X_test, y_test):
    # Метрики автоматически логируются в MLflow
    return {"accuracy": 0.95, "f1": 0.92}
```

**Логируемые метрики:**
- `accuracy` - точность классификации
- `f1` - F1-score (weighted)
- `f1_weighted` - дополнительная метрика F1
- `execution_time_seconds` - время выполнения (автоматически через `@log_execution_time`)

#### Логирование артефактов

Артефакты логируются через контекстный менеджер `mlflow_artifact_context` и напрямую в функции `run_experiment()`:

**Типы логируемых артефактов:**
1. **Модели** - через `mlflow.sklearn.log_model()`
2. **Графики важности признаков** - `feature_importances.png`
3. **Матрицы корреляций** - `correlation_heatmap.png`
4. **Другие визуализации** - через `mlflow.log_artifact()`

**Пример логирования артефактов:**

```python
# Логирование модели
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=signature
)

# Логирование графиков
mlflow.log_artifact("plots/feature_importances.png", artifact_path="plots")
mlflow.log_artifact("plots/correlation_heatmap.png", artifact_path="plots")
```

### c) Создание системы сравнения экспериментов

Создана функция `compare_runs()` в модуле `mlflow_utils.py` для сравнения экспериментов:

```python
def compare_runs(
    runs: list[Run],
    metric_names: list[str] | None = None,
    param_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Сравнивает runs и возвращает DataFrame с метриками и параметрами.
    """
```

**Использование:**

```python
from mlflow_utils import search_runs, compare_runs

# Поиск всех runs в эксперименте
runs = search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"]
)

# Сравнение runs
comparison_df = compare_runs(
    runs,
    metric_names=["accuracy", "f1"],
    param_names=["model_type", "rf_n_estimators", "rf_max_depth"]
)

print(comparison_df)
```

**Результат:** DataFrame с колонками:
- `run_id`, `experiment_id`, `run_name`, `status`, `start_time`, `end_time`
- `metric_accuracy`, `metric_f1` и другие метрики
- `param_model_type`, `param_rf_n_estimators` и другие параметры
- Теги экспериментов

**Дополнительные утилиты для сравнения:**

```python
# Получение лучших runs по метрике
from mlflow_utils import get_best_runs

best_runs = get_best_runs(runs, "accuracy", ascending=False, top_k=5)

# Получение сводки по эксперименту
from mlflow_utils import get_experiment_summary

summary = get_experiment_summary(experiment_id)
# Возвращает: количество runs, средние/мин/макс метрики, стандартное отклонение
```

### d) Настройка фильтрации и поиска экспериментов

Реализованы функции для фильтрации и поиска в модуле `mlflow_utils.py`:

#### Поиск runs

```python
def search_runs(
    experiment_ids: list[str] | None = None,
    filter_string: str | None = None,
    run_view_type: int = 1,
    max_results: int = 1000,
    order_by: list[str] | None = None,
) -> list[Run]:
```

**Примеры использования:**

```python
from mlflow_utils import search_runs

# Поиск runs с accuracy > 0.9
runs = search_runs(
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"]
)

# Поиск runs по параметрам
runs = search_runs(
    filter_string="params.model_type = 'random_forest' AND params.rf_n_estimators = '200'"
)

# Поиск в конкретном эксперименте
runs = search_runs(
    experiment_ids=["123"],
    filter_string="metrics.f1 > 0.85"
)
```

#### Фильтрация по метрикам

```python
def filter_runs_by_metrics(
    runs: list[Run],
    metric_filters: dict[str, tuple[float, float] | float],
) -> list[Run]:
```

**Пример:**

```python
from mlflow_utils import filter_runs_by_metrics

filtered = filter_runs_by_metrics(
    runs,
    {
        "accuracy": (0.8, 1.0),  # 0.8 <= accuracy <= 1.0
        "f1": 0.9,  # f1 >= 0.9
    }
)
```

#### Фильтрация по параметрам

```python
def filter_runs_by_params(
    runs: list[Run],
    param_filters: dict[str, str | list[str]],
) -> list[Run]:
```

**Пример:**

```python
from mlflow_utils import filter_runs_by_params

filtered = filter_runs_by_params(
    runs,
    {
        "model_type": "random_forest",
        "rf_n_estimators": ["100", "200"],  # n_estimators = 100 или 200
    }
)
```

#### Поиск экспериментов

```python
def search_experiments(
    filter_string: str | None = None,
    max_results: int = 1000,
) -> list[Experiment]:
```

**Пример:**

```python
from mlflow_utils import search_experiments

experiments = search_experiments(filter_string="name LIKE '%wine%'")
```

---

## 3. Интеграция с кодом (2 балла)

### a) Интеграция MLflow в Python код

MLflow интегрирован в основные модули проекта:

#### Интеграция в `model.py`

Функция `run_experiment()` полностью интегрирована с MLflow:

```python
def run_experiment(
    X_train, y_train, X_test, y_test,
    model_type: str | ModelType = ModelType.BOOSTING,
    use_mlflow: bool = True,
    experiment_name: str | None = None,
    register_model_name: str | None = None,
    log_artifacts: bool = True,
    # ... другие параметры
):
    # Автоматическое логирование через декораторы
    model = train_model(...)  # @log_params, @log_execution_time
    metrics = evaluate_model(...)  # @log_metrics

    if use_mlflow:
        with MLflowExperimentContext(experiment_name):
            with MLflowRunContext(...):
                # Логирование модели, метрик, артефактов
                mlflow.sklearn.log_model(...)
                mlflow.log_metrics(...)
                mlflow.log_artifacts(...)
```

#### Интеграция в `main.py`

Командная строка поддерживает работу с MLflow:

```python
# Поддержка аргументов командной строки
parser.add_argument("--experiment-name", help="Имя эксперимента в MLflow")
parser.add_argument("--register-model", help="Имя модели для регистрации")
parser.add_argument("--no-mlflow", action="store_true", help="Отключить MLflow")
parser.add_argument("--analyze-experiments", action="store_true", help="Анализ экспериментов")
```

### b) Создание декораторов для автоматического логирования

Создан модуль `mlflow_decorators.py` с набором декораторов:

#### 1. `@log_params` - логирование параметров функции

```python
@log_params
def train_model(n_estimators=100, max_depth=5):
    # Все параметры автоматически логируются
    pass
```

#### 2. `@log_metrics` - логирование метрик из возвращаемого значения

```python
@log_metrics(["accuracy", "f1_score"])
def evaluate_model():
    return {"accuracy": 0.95, "f1_score": 0.92}
```

#### 3. `@log_execution_time` - логирование времени выполнения

```python
@log_execution_time
def train_model():
    pass
```

#### 4. `@combine_decorators` - комбинирование декораторов

```python
@combine_decorators(
    log_params,
    log_execution_time,
    log_metrics(["accuracy", "f1_score"])
)
def train_and_evaluate():
    pass
```

### c) Настройка контекстных менеджеров

Создан модуль `mlflow_context.py` с контекстными менеджерами:

#### 1. `MLflowExperimentContext` - управление экспериментами

```python
with MLflowExperimentContext("my_experiment", tags={"team": "ml"}):
    # Эксперимент автоматически создан или выбран
    mlflow.log_param("param1", "value1")
```

**Функциональность:**
- Автоматическое создание эксперимента, если его нет
- Восстановление удаленных экспериментов
- Установка тегов для экспериментов

#### 2. `MLflowRunContext` - управление runs

```python
with MLflowRunContext(
    experiment_name="my_experiment",
    run_name="test_run",
    params={"n_estimators": 100},
    tags={"framework": "sklearn"}
) as run:
    mlflow.log_metric("accuracy", 0.95)
    # Run автоматически завершается при выходе из контекста
```

**Функциональность:**
- Автоматическое создание и завершение run
- Логирование параметров при входе
- Логирование системной информации (Python версия, платформа)
- Установка тегов

#### 3. `mlflow_artifact_context` - логирование артефактов

```python
with mlflow_artifact_context("plots", "visualizations") as artifact_dir:
    plot_path = artifact_dir / "plot.png"
    plt.savefig(plot_path)
    # Файл автоматически залогируется при выходе из контекста
```

#### 4. `mlflow_model_context` - логирование моделей

```python
with mlflow_model_context(
    model=my_model,
    artifact_path="model",
    registered_model_name="MyModel"
):
    # Модель автоматически залогируется при выходе из контекста
    pass
```

**Поддержка различных типов моделей:**
- scikit-learn модели
- PyTorch модели
- TensorFlow/Keras модели
- Произвольные Python модели (через pyfunc)

#### 5. `MLflowTrackingContext` - управление подключением

```python
with MLflowTrackingContext(
    tracking_uri="http://mlflow:5000",
    username="admin",
    password="password"
):
    # Настроено подключение к MLflow с аутентификацией
    mlflow.log_metric("f1_weighted", 0.95)
    # При выходе из контекста настройки восстанавливаются
```

### d) Создание утилит для работы с экспериментами

Создан модуль `mlflow_utils.py` с набором утилит:

#### Основные функции:

1. **`search_experiments()`** - поиск экспериментов по фильтру
2. **`search_runs()`** - поиск runs с фильтрацией и сортировкой
3. **`filter_runs_by_metrics()`** - фильтрация runs по метрикам
4. **`filter_runs_by_params()`** - фильтрация runs по параметрам
5. **`compare_runs()`** - сравнение runs и создание DataFrame
6. **`get_best_runs()`** - получение лучших runs по метрике
7. **`get_experiment_summary()`** - сводка по эксперименту (статистика метрик)
8. **`export_runs_to_csv()`** - экспорт runs в CSV
9. **`delete_runs()`** - удаление runs
10. **`restore_runs()`** - восстановление удаленных runs

**Пример использования утилит:**

```python
from mlflow_utils import (
    search_runs,
    compare_runs,
    get_best_runs,
    get_experiment_summary,
    export_runs_to_csv
)

runs = search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"]
)

df = compare_runs(runs, metric_names=["accuracy", "f1"])

# Лучшие runs
best = get_best_runs(runs, "accuracy", top_k=5)

# Сводка
summary = get_experiment_summary(experiment_id)


```
![alt text](../pics/search.png)
---

## 4. Отчет о проделанной работе (2 балла)

Данный отчет является финальной версией
