# Отчет по проделанной работе: Интеграция Luigi и Hydra

## Выбранные инструменты

### Для оркестрации пайплайна
**Luigi**

### Для управления конфигурациями
**Hydra**

---

## 1. Настройка выбранного инструмента оркестрации (4 балла)

### a) Установка и настройка Luigi

Luigi установлен через Poetry и добавлен в зависимости проекта:

![alt text](../pics/luigi_install.png)

Luigi интегрирован в проект через модуль `src/wine_quality/luigi_pipeline.py`

### b) Создание workflow для ML пайплайна

Создан полный workflow для ML пайплайна, состоящий из следующих задач:

#### 1. `PrepareData` - Подготовка исходных данных
```python
class PrepareData(luigi.Task):
    """Задача для подготовки исходных данных."""

    output_path = luigi.Parameter(default=str(BASE_DIR / "data" / "winequality-red.csv"))

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.output_path)

    def run(self) -> None:
        df = load_data()
        df.to_csv(self.output_path, index=False)
```

#### 2. `GenerateFeatures` - Генерация признаков
```python
class GenerateFeatures(luigi.Task):
    """Задача для генерации признаков из исходных данных."""

    def requires(self) -> PrepareData:
        return PrepareData(output_path=self.input_path)

    def run(self) -> None:
        df = load_data()
        df = create_target(df, threshold=self.quality_threshold)
        df = engineer_features(df)
        df.to_csv(self.output_path, index=False)
```

#### 3. `TrainModel` - Обучение модели
```python
class TrainModel(luigi.Task):
    """Задача для обучения модели с поддержкой конфигураций Hydra."""

    def requires(self) -> GenerateFeatures:
        return GenerateFeatures(output_path=self.features_path)

    def run(self) -> None:
        ...
```

#### 4. `EvaluateModel` - Оценка модели
```python
class EvaluateModel(luigi.Task):
    """Задача для оценки обученной модели."""

    def requires(self) -> TrainModel:
        return TrainModel(...)

    def run(self) -> None:
        ...
```

#### 5. `WineQualityPipeline` - Главная задача пайплайна
```python
class WineQualityPipeline(luigi.WrapperTask):
    """Главная задача, объединяющая все этапы пайплайна."""

    def requires(self) -> list[luigi.Task]:
        return [
            EvaluateModel(...)
        ]
```

### c) Настройка зависимостей между этапами

Зависимости между задачами настроены через метод `requires()`:

```
PrepareData
    ↓
GenerateFeatures (requires: PrepareData)
    ↓
TrainModel (requires: GenerateFeatures)
    ↓
EvaluateModel (requires: TrainModel)
    ↓
WineQualityPipeline (requires: EvaluateModel)
```

Luigi автоматически определяет порядок выполнения задач на основе зависимостей и выполняет только те задачи, выходные файлы которых отсутствуют или устарели.

### d) Реализация кэширования и параллельного выполнения

#### Кэширование

Кэширование реализовано через механизм `output()` в Luigi. Каждая задача определяет свои выходные файлы:

```python
def output(self) -> luigi.LocalTarget:
    return luigi.LocalTarget(self.output_path)
```

Luigi автоматически проверяет существование выходных файлов и пропускает выполнение задачи, если файлы уже существуют и актуальны.

![alt text](../pics/luigi_cache.png)
#### Параллельное выполнение

Параллельное выполнение настроено через параметры `luigi.build()`:

```python
def _prepare_build_params(cfg: DictConfig) -> dict[str, Any]:
    """Подготавливает параметры для luigi.build()."""
    pipeline_cfg = cfg.get("pipeline", {})
    workers = pipeline_cfg.get("workers", 1)
    parallel = pipeline_cfg.get("parallel", False)

    build_params = {
        "local_scheduler": scheduler_type == "local",
    }

    if parallel and workers > 1:
        build_params["workers"] = workers
        logger.info(f"Запуск пайплайна с параллельным выполнением: {workers} workers")

    return build_params
```

Конфигурация параллельного выполнения задается в `conf/pipeline/pipeline_luigi.yaml`:

```yaml
pipeline:
  runner: luigi
  parallel: true
  workers: 4
  scheduler: local
  log_level: INFO
```

---
![alt text](../pics/luigi_run.png)


## 2. Настройка выбранного инструмента конфигураций (3 балла)

### a) Настройка Hydra для управления конфигурациями

Hydra установлен через Poetry:

```toml
hydra-core = "^1.3.2"
omegaconf = "^2.3.0"
```

Hydra интегрирован в проект через декоратор `@hydra.main()` в модуле `src/wine_quality/orchestrator.py`:

```python
@hydra.main(
    config_path=str(BASE_DIR / "conf"),
    config_name="config",
    version_base=None,
)
def run_pipeline_with_hydra(cfg: DictConfig) -> None:
    """Запускает Luigi пайплайн с конфигурацией из Hydra."""
    # ...
```

### b) Создание конфигураций для разных алгоритмов

Создана иерархическая структура конфигураций в директории `conf/`:

#### Конфигурации моделей (`conf/model/`)

**Random Forest:**
- `random_forest.yaml` - базовая конфигурация
- `random_forest_fast.yaml` - быстрая конфигурация (меньше деревьев)
- `random_forest_deep.yaml` - глубокая конфигурация (больше деревьев, больше глубина)

**Boosting:**
- `boosting.yaml` - базовая конфигурация
- `boosting_fast.yaml` - быстрая конфигурация
- `boosting_deep.yaml` - глубокая конфигурация

**MLP:**
- `mlp.yaml` - базовая конфигурация
- `mlp_small.yaml` - маленькая сеть
- `mlp_large.yaml` - большая сеть

#### Конфигурации данных (`conf/data/`)

- `data.yaml` - базовая конфигурация
- `data_small.yaml` - небольшой датасет
- `data_large.yaml` - большой датасет
- `data_strict.yaml` - строгий порог качества
- `data_lenient.yaml` - мягкий порог качества

#### Конфигурации окружения (`conf/env/`)

- `dev.yaml` - конфигурация для разработки
- `test.yaml` - конфигурация для тестирования
- `prod.yaml` - конфигурация для продакшена

#### Конфигурации пайплайна (`conf/pipeline/`)

- `pipeline_luigi.yaml` - параллельное выполнение
- `pipeline_sequential.yaml` - последовательное выполнение

### c) Настройка валидации конфигураций

Валидация конфигураций реализована через Pydantic схемы в модуле `src/wine_quality/config_schema.py`:

```python
class ModelConfig(BaseModel):
    n_estimators: int | None = Field(
        default=None, ge=1, le=10000, description="Количество деревьев"
    )
    max_depth: int | None = Field(
        default=None, ge=1, le=100, description="Максимальная глубина дерева"
    )
    learning_rate: float | None = Field(
        default=None, gt=0.0, le=1.0, description="Скорость обучения"
    )
    # ...

class DataConfig(BaseModel):
    data_url: str = Field(description="URL для загрузки данных")
    test_size: float = Field(ge=0.0, lt=1.0, description="Доля тестовой выборки")
    quality_threshold: int = Field(
        ge=1, le=10, description="Порог качества для бинарной классификации"
    )

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("test_size должен быть между 0 и 1")
        return v

class AppConfig(BaseModel):
    model_type: Literal["random_forest", "boosting", "mlp"] = Field(description="Тип модели")
    random_state: int = Field(ge=0, description="Случайное состояние для воспроизводимости")
    model: ModelConfig
    data: DataConfig
    mlflow: MLflowConfig
    paths: PathsConfig
```

Валидация выполняется в функции `run_pipeline_with_hydra()`:

```python
cfg_dict = OmegaConf.to_container(cfg, resolve=True)
if "env" in cfg_dict and cfg_dict["env"] is None:
    del cfg_dict["env"]
validated_cfg = AppConfig(**cfg_dict)
logger.info("✓ Конфигурация успешно валидирована")
```

### d) Создание системы композиции конфигураций

Композиция конфигураций реализована через механизм `defaults` в Hydra:

```yaml
# conf/config_luigi.yaml
# @package _global_

defaults:
  - _self_
  - model: random_forest
  - data: data
  - mlflow: mlflow
  - paths: paths
  - pipeline: pipeline_luigi
  - env: null

model_type: random_forest
random_state: 42
```

Это позволяет легко переключаться между различными конфигурациями:

```bash
# Запуск с базовой конфигурацией Random Forest
python run_luigi_pipeline.py

# Запуск с быстрой конфигурацией Random Forest
python run_luigi_pipeline.py model=random_forest_fast

# Запуск с глубокой конфигурацией Boosting
python run_luigi_pipeline.py model=boosting_deep data=data_strict

# Запуск с последовательным выполнением
python run_luigi_pipeline.py pipeline=pipeline_sequential
```

---

## 3. Интеграция и тестирование (2 балла)

### a) Интеграция выбранных инструментов

Интеграция Luigi и Hydra реализована в модуле `src/wine_quality/orchestrator.py`:

```python
@hydra.main(
    config_path=str(BASE_DIR / "conf"),
    config_name="config",
    version_base=None,
)
def run_pipeline_with_hydra(cfg: DictConfig) -> None:
    """Запускает Luigi пайплайн с конфигурацией из Hydra."""
    # Валидация конфигурации через Pydantic
    validated_cfg = AppConfig(**cfg_dict)

    # Преобразование конфигурации Hydra в параметры Luigi
    luigi_params = _prepare_luigi_params(validated_cfg, cfg)

    # Создание и запуск Luigi задачи
    task = WineQualityPipeline(**luigi_params)
    build_params = _prepare_build_params(cfg)
    success = luigi.build([task], **build_params)
```

Конфигурация Hydra передается в Luigi задачи через параметр `hydra_config` в формате JSON, что позволяет задачам Luigi использовать те же конфигурации.

### b) Создание системы мониторинга выполнения

Реализован класс `PipelineMonitor` для мониторинга выполнения пайплайна:

```python
class PipelineMonitor:
    """Класс для мониторинга выполнения пайплайна."""

    def __init__(self, log_dir: Path | str | None = None):
        if log_dir is None:
            log_dir = BASE_DIR / "outputs" / "pipeline_logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_task_start(self, task_name: str, params: dict[str, Any]) -> None:
        """Логирует начало выполнения задачи."""
        log_entry = {
            "task": task_name,
            "status": "started",
            "params": params,
            "timestamp": str(Path(__file__).stat().st_mtime),
        }
        self._write_log(log_entry)

    def log_task_complete(self, task_name: str, metrics: dict[str, Any] | None = None) -> None:
        """Логирует завершение задачи."""
        # ...

    def log_task_failure(self, task_name: str, error: str) -> None:
        """Логирует ошибку выполнения задачи."""
        # ...
```

Логи сохраняются в JSON формате в директории `outputs/pipeline_logs/` с временными метками.

### c) Настройка уведомлений о результатах

Реализован класс `NotificationSystem` для уведомлений о результатах выполнения:

```python
class NotificationSystem:
    """Система уведомлений о результатах выполнения."""

    def notify_success(self, message: str, metrics: dict[str, Any] | None = None) -> None:
        """Отправляет уведомление об успешном выполнении."""
        logger.info(f"✓ УСПЕХ: {message}")
        if metrics:
            logger.info(f"  Метрики: {metrics}")

    def notify_failure(self, message: str, error: str | None = None) -> None:
        """Отправляет уведомление об ошибке."""
        logger.error(f"✗ ОШИБКА: {message}")
        if error:
            logger.error(f"  Детали: {error}")

    def notify_completion(self, summary: dict[str, Any]) -> None:
        """Отправляет итоговое уведомление."""
        logger.info("=" * 80)
        logger.info("ИТОГИ ВЫПОЛНЕНИЯ ПАЙПЛАЙНА")
        logger.info("=" * 80)
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
```

Уведомления можно отключить через параметр `no_notifications` в конфигурации.

### d) Тестирование воспроизводимости

Воспроизводимость обеспечивается через:

1. **Фиксация random_state** в конфигурации:
```yaml
random_state: 42
```

2. **Версионирование зависимостей** через Poetry (`poetry.lock`)

3. **Логирование всех параметров** в MLflow и локальные файлы метрик

4. **Кэширование результатов** через Luigi - при повторном запуске с теми же параметрами задачи не перевыполняются

Пример выполнения пайплайна:

```bash
# Первый запуск
python run_luigi_pipeline.py model=random_forest

# Повторный запуск с теми же параметрами - задачи не перевыполняются
python run_luigi_pipeline.py model=random_forest
```

Результаты выполнения сохраняются в:
- `models/wine_<model_type>.pkl` - обученная модель
- `models/wine_<model_type>_metrics.json` - метрики модели
- `models/wine_<model_type>_report.json` - отчет о модели
- `outputs/pipeline_logs/` - логи выполнения пайплайна

---
![alt text](../pics/hydra_run1.png)
![alt text](../pics/hydra_run2.png)



## Подготовлен отчет
Настоящий документ является финальным отчетом.
