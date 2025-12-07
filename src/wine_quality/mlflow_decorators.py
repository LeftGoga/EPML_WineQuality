"""
Декораторы для автоматического логирования в MLflow.

Этот модуль предоставляет декораторы для автоматизации логирования
параметров, метрик, времени выполнения и артефактов в MLflow.
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import mlflow

# Импорты для логирования моделей (опциональные, импортируются по требованию)
try:
    import mlflow.sklearn
except ImportError:
    pass

try:
    import mlflow.pytorch
except ImportError:
    pass

try:
    import mlflow.tensorflow
except ImportError:
    pass

F = TypeVar("F", bound=Callable[..., Any])


def log_params(func: F) -> F:
    """
    Декоратор для автоматического логирования параметров функции в MLflow.

    Логирует все аргументы функции как параметры MLflow.

    Пример использования:
        @log_params
        def train_model(n_estimators=100, max_depth=5):
            # Параметры автоматически залогируются
            pass
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Получаем сигнатуру функции
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Логируем параметры только если есть активный MLflow run
        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                params = {}
                for param_name, param_value in bound_args.arguments.items():
                    # Пропускаем self для методов
                    if param_name == "self":
                        continue
                    # Конвертируем значения в строки для MLflow
                    if isinstance(param_value, (list, tuple, dict)):
                        params[param_name] = str(param_value)
                    else:
                        params[param_name] = str(param_value)

                if params:
                    mlflow.log_params(params)
        except Exception:  # nosec B110
            # Игнорируем ошибки, если MLflow не настроен или нет активного run
            pass

        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def log_metrics(metric_names: list[str] | None = None) -> Callable[[F], F]:
    """
    Декоратор для автоматического логирования метрик из возвращаемого значения.

    Если metric_names указан, логирует только указанные метрики.
    Если metric_names не указан, пытается логировать все метрики из словаря.

    Пример использования:
        @log_metrics(["accuracy", "f1_score"])
        def evaluate_model():
            return {"accuracy": 0.95, "f1_score": 0.92}

        @log_metrics()
        def evaluate_model():
            return {"accuracy": 0.95, "f1_score": 0.92}  # Все метрики залогируются
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Если результат - словарь, логируем метрики только если есть активный MLflow run
            try:
                active_run = mlflow.active_run()
                if active_run is not None and isinstance(result, dict):
                    metrics_to_log = {}
                    if metric_names:
                        # Логируем только указанные метрики
                        for metric_name in metric_names:
                            if metric_name in result:
                                value = result[metric_name]
                                if isinstance(value, (int, float)):
                                    metrics_to_log[metric_name] = float(value)
                    else:
                        # Логируем все числовые значения из словаря
                        for key, value in result.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                metrics_to_log[key] = float(value)

                    if metrics_to_log:
                        mlflow.log_metrics(metrics_to_log)
            except Exception:  # nosec B110
                # Игнорируем ошибки, если MLflow не настроен или нет активного run
                pass

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def log_execution_time(func: F) -> F:
    """
    Декоратор для логирования времени выполнения функции.

    Пример использования:
        @log_execution_time
        def train_model():
            # Время выполнения будет залогировано как метрика "execution_time_seconds"
            pass
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            try:
                active_run = mlflow.active_run()
                if active_run is not None:
                    mlflow.log_metric("execution_time_seconds", execution_time)
            except Exception:  # nosec B110
                pass
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            try:
                active_run = mlflow.active_run()
                if active_run is not None:
                    mlflow.log_metric("execution_time_seconds", execution_time)
                    mlflow.log_param("error", str(e))
            except Exception:  # nosec B110
                pass
            raise

    return wrapper  # type: ignore[return-value]


def log_artifacts(artifact_path: str | None = None) -> Callable[[F], F]:
    """
    Декоратор для автоматического логирования артефактов из возвращаемого значения.

    Если функция возвращает путь к файлу или директории, он автоматически логируется.

    Пример использования:
        @log_artifacts("plots")
        def create_plot():
            plt.savefig("plot.png")
            return "plot.png"  # Файл будет залогирован
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Если результат - строка (путь к файлу)
            if isinstance(result, str):
                path = Path(result)
                if path.exists():
                    if path.is_file():
                        mlflow.log_artifact(str(path), artifact_path=artifact_path)
                    elif path.is_dir():
                        mlflow.log_artifacts(str(path), artifact_path=artifact_path)

            # Если результат - Path объект
            elif isinstance(result, Path):
                if result.exists():
                    if result.is_file():
                        mlflow.log_artifact(str(result), artifact_path=artifact_path)
                    elif result.is_dir():
                        mlflow.log_artifacts(str(result), artifact_path=artifact_path)

            # Если результат - список путей
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, (str, Path)):
                        path = Path(item) if isinstance(item, str) else item
                        if path.exists() and path.is_file():
                            mlflow.log_artifact(str(path), artifact_path=artifact_path)

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def mlflow_run(
    experiment_name: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Callable[[F], F]:
    """
    Декоратор для автоматического создания MLflow run.

    Обёртывает выполнение функции в MLflow run.

    Пример использования:
        @mlflow_run(experiment_name="my_experiment", run_name="test_run")
        def train_model():
            mlflow.log_param("param1", "value1")
            mlflow.log_metric("accuracy", 0.95)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Устанавливаем эксперимент, если указан
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            # Создаём run
            with mlflow.start_run(run_name=run_name, tags=tags):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def log_model_artifact(
    artifact_path: str = "model",
    registered_model_name: str | None = None,
) -> Callable[[F], F]:
    """
    Декоратор для автоматического логирования модели из возвращаемого значения.

    Пример использования:
        @log_model_artifact(artifact_path="model", registered_model_name="MyModel")
        def train_model():
            model = RandomForestClassifier()
            model.fit(X, y)
            return model  # Модель будет залогирована
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Пытаемся определить тип модели и залогировать
            if result is not None:
                model_type = type(result).__module__

                try:
                    if "sklearn" in model_type or hasattr(result, "fit"):
                        mlflow.sklearn.log_model(
                            sk_model=result,
                            artifact_path=artifact_path,
                            registered_model_name=registered_model_name,
                        )
                    elif "pytorch" in model_type or "torch" in model_type:
                        mlflow.pytorch.log_model(
                            pytorch_model=result,
                            artifact_path=artifact_path,
                            registered_model_name=registered_model_name,
                        )
                    elif "tensorflow" in model_type or "keras" in model_type:
                        mlflow.tensorflow.log_model(
                            model=result,
                            artifact_path=artifact_path,
                            registered_model_name=registered_model_name,
                        )
                except Exception as e:
                    print(f"Предупреждение: не удалось залогировать модель: {e}")

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def combine_decorators(
    *decorators: Callable[[F], F],
) -> Callable[[F], F]:
    """
    Утилита для комбинирования нескольких декораторов.

    Пример использования:
        @combine_decorators(
            log_params,
            log_execution_time,
            log_metrics(["accuracy", "f1_score"])
        )
        def train_and_evaluate():
            pass
    """

    def decorator(func: F) -> F:
        for dec in reversed(decorators):
            func = dec(func)
        return func

    return decorator
