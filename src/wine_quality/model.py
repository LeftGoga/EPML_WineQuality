# model.py
"""
Модуль для тренировки/оценки/логирования RandomForestClassifier с аккуратной интеграцией MLflow.
Ключевые идеи:
- Используем `registered_model_name` при логировании, чтобы избежать двойной регистрации.
- Предлагаем совместимый вызов log_model (поддержка deprecated artifact_path -> name).
- Добавляем сигнатуру модели (если возможно) и input_example, чтобы удалить предупреждение.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import joblib
import mlflow
import pandas as pd
from config import MAX_DEPTH, MODEL_PATH, N_ESTIMATORS, RANDOM_STATE
from mlflow.tracking import MlflowClient
from mlflow_registry import (
    log_metadata,
    register_model_from_run,
    set_model_version_tags,
    transition_model_stage,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Optional MLflow integration flag
try:
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature

    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False
    if TYPE_CHECKING:
        from mlflow.models.signature import infer_signature
    else:
        infer_signature = None


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    random_state: int | None = None,
) -> RandomForestClassifier:
    if n_estimators is None:
        n_estimators = N_ESTIMATORS
    if max_depth is None:
        max_depth = MAX_DEPTH
    if random_state is None:
        random_state = RANDOM_STATE

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (weighted): {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def save_model(model: Any, path: str | None = None) -> None:
    if path is None:
        path = str(MODEL_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Модель сохранена: {path}")


def load_model(path: str | None = None) -> Any:
    if path is None:
        path = str(MODEL_PATH)
    return joblib.load(path)


def _log_model_compat(
    sk_model: Any,
    artifact_path: str,
    registered_model_name: str | None = None,
    X_sample: pd.DataFrame | None = None,
) -> str | None:
    """Совместимый вызов логирования модели, учитывающий предупреждение о artifact_path.
    Попытается infer_signature и передать signature/input_example.
    """
    kwargs = {}
    # Попытка infer_signature
    if infer_signature is not None and X_sample is not None:
        try:
            signature = infer_signature(X_sample, sk_model.predict(X_sample))
            kwargs["signature"] = signature
        except Exception as exc:
            # Игнорируем ошибки при создании сигнатуры - это не критично
            print(f"Warning: could not infer signature: {exc}")

    # Некоторые версии mlflow ожидают name вместо artifact_path. Попробуем оба безопасно.
    # Сначала пробуем вызвать с name, если есть поддержка
    used = None
    try:
        # mlflow.sklearn.log_model может принимать 'name' вместо 'artifact_path' в новых версиях
        if registered_model_name is not None:
            mlflow.sklearn.log_model(
                sk_model=sk_model,
                name=artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        else:
            mlflow.sklearn.log_model(sk_model=sk_model, name=artifact_path, **kwargs)
        used = "name"
    except TypeError:
        # fallback на artifact_path
        if registered_model_name is not None:
            mlflow.sklearn.log_model(
                sk_model=sk_model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        else:
            mlflow.sklearn.log_model(sk_model=sk_model, artifact_path=artifact_path, **kwargs)
        used = "artifact_path"
    return used


def run_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int | None = None,
    use_mlflow: bool = True,
    experiment_name: str | None = None,
    model_artifact_path: str = "model",
    save_local: bool = True,
    register_model_name: str | None = None,
) -> dict[str, Any]:
    """
    Запускает тренировку, оценку и логгирование.

    Ключевой момент: не смешиваем два способа регистрации.
    Если передан `register_model_name`, используем `mlflow.sklearn.log_model(..., registered_model_name=...)`.
    Не вызываем ручную register_model_from_run, чтобы не получить дубликаты `models:/...`.
    """
    if use_mlflow and not MLFLOW_AVAILABLE:
        raise RuntimeError(
            "MLflow не установлен или доступен. Установите mlflow: pip install mlflow"
        )

    model = train_model(
        X_train, y_train, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )

    metrics = evaluate_model(model, X_test, y_test)

    mlflow_run_id = None

    if use_mlflow and MLFLOW_AVAILABLE:
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            mlflow_run_id = run.info.run_id

            params = {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "random_state": model.random_state,
            }
            mlflow.log_params(params)

            mlflow.log_metric("accuracy", float(metrics["accuracy"]))
            mlflow.log_metric("f1_weighted", float(metrics["f1"]))

            # Логируем модель. Если указан register_model_name — используем registered_model_name
            _log_model_compat(
                sk_model=model,
                artifact_path=model_artifact_path,
                registered_model_name=register_model_name,
                X_sample=X_train,
            )

            model_uri = f"runs:/{run.info.run_id}/{model_artifact_path}"
            mlflow.log_param("model_uri", model_uri)

            # Логируем metadata
            metadata = {
                "feature_columns": X_train.columns.tolist()
                if hasattr(X_train, "columns")
                else None,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
            log_metadata(metadata)

            # Если register_model_name указан, mlflow.sklearn.log_model уже создал версию в Registry
            if register_model_name:
                client = MlflowClient()
                # найти версии, связанные с этим run и artifact (без создания новых)
                versions = [
                    v
                    for v in client.get_latest_versions(
                        register_model_name, stages=["None", "Staging", "Production", "Archived"]
                    )
                    if v.run_id == run.info.run_id
                ]
                if versions:
                    # обычно это одна версия — берём первую
                    version = str(versions[0].version)
                    tags = {
                        "accuracy": float(metrics["accuracy"]),
                        "f1_weighted": float(metrics["f1"]),
                        "n_estimators": model.n_estimators,
                    }
                    set_model_version_tags(register_model_name, version, tags)
                    # Переводим в Staging
                    transition_model_stage(register_model_name, version, "Staging")
                else:
                    # На удивление версия не найдена — можно создать вручную (редкий кейс)
                    try:
                        created_version = register_model_from_run(
                            run.info.run_id, model_artifact_path, register_model_name
                        )
                        set_model_version_tags(
                            register_model_name,
                            created_version,
                            {"accuracy": float(metrics["accuracy"])},
                        )
                        transition_model_stage(register_model_name, created_version, "Staging")
                    except Exception as exc:
                        print("Warning: could not find or create model version in registry:", exc)

    if save_local:
        save_model(model)

    return {"model": model, "metrics": metrics, "mlflow_run_id": mlflow_run_id}
