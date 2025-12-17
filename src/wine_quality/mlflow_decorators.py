from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import mlflow

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
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                params = {}
                for param_name, param_value in bound_args.arguments.items():
                    if param_name == "self":
                        continue

                    if isinstance(param_value, (list, tuple, dict)):
                        params[param_name] = str(param_value)
                    else:
                        params[param_name] = str(param_value)

                if params:
                    mlflow.log_params(params)
        except Exception:  # nosec B110
            pass

        return func(*args, **kwargs)

    return cast(F, wrapper)


def log_metrics(metric_names: list[str] | None = None) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            try:
                active_run = mlflow.active_run()
                if active_run is not None and isinstance(result, dict):
                    metrics_to_log = {}
                    if metric_names:
                        for metric_name in metric_names:
                            if metric_name in result:
                                value = result[metric_name]
                                if isinstance(value, (int, float)):
                                    metrics_to_log[metric_name] = float(value)
                    else:
                        for key, value in result.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                metrics_to_log[key] = float(value)

                    if metrics_to_log:
                        mlflow.log_metrics(metrics_to_log)
            except Exception:  # nosec B110
                pass

            return result

        return cast(F, wrapper)

    return decorator


def log_artifacts(artifact_path: str | None = None) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            if isinstance(result, str):
                path = Path(result)
                if path.exists():
                    if path.is_file():
                        mlflow.log_artifact(str(path), artifact_path=artifact_path)
                    elif path.is_dir():
                        mlflow.log_artifacts(str(path), artifact_path=artifact_path)

            elif isinstance(result, Path):
                if result.exists():
                    if result.is_file():
                        mlflow.log_artifact(str(result), artifact_path=artifact_path)
                    elif result.is_dir():
                        mlflow.log_artifacts(str(result), artifact_path=artifact_path)

            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, (str, Path)):
                        path = Path(item) if isinstance(item, str) else item
                        if path.exists() and path.is_file():
                            mlflow.log_artifact(str(path), artifact_path=artifact_path)

            return result

        return cast(F, wrapper)

    return decorator


def mlflow_run(
    experiment_name: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=run_name, tags=tags):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
