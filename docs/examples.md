# Примеры использования

Практические примеры работы с проектом.

## Базовый пример: Загрузка и подготовка данных

```python
from wine_quality.data import load_data, create_target, engineer_features
from wine_quality.features import get_features_and_target

# Загрузка данных
df = load_data()

# Создание целевой переменной (качество >= 7)
df = create_target(df, threshold=7)

# Инженерия признаков
df = engineer_features(df)

# Получение признаков и целевой переменной
X, y = get_features_and_target(df)

print(f"Размер данных: {X.shape}")
print(f"Количество признаков: {len(X.columns)}")
```

## Обучение модели

### Random Forest

```python
from wine_quality.model import ModelType, run_experiment
from sklearn.model_selection import train_test_split

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучение модели
result = run_experiment(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_type=ModelType.RANDOM_FOREST,
    rf_n_estimators=200,
    rf_max_depth=10,
    use_clearml=True,
)

print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
print(f"F1-score: {result['metrics']['f1_weighted']:.4f}")
```

### Gradient Boosting

```python
result = run_experiment(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_type=ModelType.BOOSTING,
    boosting_n_estimators=100,
    boosting_max_depth=3,
    boosting_learning_rate=0.1,
    use_clearml=True,
)
```

### MLP (Neural Network)

```python
result = run_experiment(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_type=ModelType.MLP,
    mlp_hidden_layer_sizes=(100, 50),
    mlp_max_iter=500,
    use_clearml=True,
)
```

## Работа с ClearML

### Создание эксперимента

```python
from wine_quality.clearml_context import ClearMLTaskContext

with ClearMLTaskContext(
    project_name="Wine Quality",
    task_name="My Experiment",
    task_type="training"
) as ctx:
    # Ваш код эксперимента
    task = ctx.task

    # Логирование параметров
    task.connect({"learning_rate": 0.1, "n_estimators": 100})

    # Логирование метрик
    task.logger.report_scalar(
        title="Metrics",
        series="Accuracy",
        value=0.95,
        iteration=0
    )
```

### Регистрация модели

```python
from wine_quality.clearml_model_registry import register_model_with_version

version_info = register_model_with_version(
    model=trained_model,
    model_name="wine_quality_boosting",
    model_path="models/wine_boosting.pkl",
    framework="scikit-learn",
    tags=["boosting", "wine_quality"],
    labels={"accuracy": "0.95"},
    auto_version=True,
)

print(f"Модель зарегистрирована: версия {version_info['version']}")
```

## Запуск пайплайна

### Локальный запуск

```python
from wine_quality.clearml_pipeline import wine_quality_pipeline
from clearml import PipelineDecorator

# Запуск локально
PipelineDecorator.run_locally()

results = wine_quality_pipeline(
    data_path="data/winequality-red.csv",
    model_type="boosting",
    quality_threshold=7,
    test_size=0.2,
)
```

### Запуск через скрипт

```bash
python run_clearml_pipeline.py \
    --model-type boosting \
    --quality-threshold 7 \
    --test-size 0.2
```

## Работа с конфигурацией

### Использование Hydra

```python
from hydra import compose, initialize
from omegaconf import DictConfig

with initialize(config_path="conf", version_base=None):
    cfg = compose(config_name="config")

    print(cfg.model.boosting.n_estimators)
    print(cfg.data.test_size)
```

### Загрузка конфигурации модели

```python
from wine_quality.config_schema import ModelConfig, load_model_config

# Загрузка конфигурации
config = load_model_config("conf/model/boosting.yaml")

# Использование
model_config = ModelConfig(**config)
print(f"n_estimators: {model_config.n_estimators}")
```

## Визуализация результатов

### Создание графиков

```python
import matplotlib.pyplot as plt
import seaborn as sns
from wine_quality.utils import plot_feature_importances

# График важности признаков
plot_feature_importances(
    model=result['model'],
    feature_names=X.columns,
    save_path="plots/feature_importances.png"
)

# Корреляционная матрица
corr_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.savefig("plots/correlation_heatmap.png")
```

## Использование Streamlit приложения

### Запуск

```bash
streamlit run src/wine_quality/app.py
```

### Программный запуск

```python
import subprocess

subprocess.run([
    "streamlit", "run",
    "src/wine_quality/app.py",
    "--server.port=8501"
])
```

## Пакетный запуск экспериментов

```python
from wine_quality.clearml_experiment import run_clearml_experiment

# Список конфигураций
configs = [
    {"model_type": "boosting", "boosting_n_estimators": 100},
    {"model_type": "boosting", "boosting_n_estimators": 200},
    {"model_type": "random_forest", "rf_n_estimators": 100},
    {"model_type": "random_forest", "rf_n_estimators": 200},
]

# Запуск экспериментов
results = []
for config in configs:
    result = run_clearml_experiment(**config)
    results.append(result)

# Сравнение результатов
for i, result in enumerate(results):
    print(f"Эксперимент {i+1}:")
    print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"  F1-score: {result['metrics']['f1_weighted']:.4f}")
```

## Работа с сохраненными моделями

### Загрузка модели

```python
import joblib

# Загрузка модели
model = joblib.load("models/wine_boosting.pkl")

# Предсказание
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Оценка модели

```python
from sklearn.metrics import accuracy_score, classification_report

# Оценка точности
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Детальный отчет
report = classification_report(y_test, predictions)
print(report)
```

## Следующие шаги

- 📚 Изучите [API документацию](api/index.md) для детальной информации
- 🔧 Ознакомьтесь с [Руководством по развертыванию](deployment.md)
- 📊 Посмотрите [Отчеты об экспериментах](experiments/index.md)
