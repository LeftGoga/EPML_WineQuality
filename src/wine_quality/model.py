from __future__ import annotations

import os
from enum import Enum
from typing import Any

import joblib
import mlflow
import pandas as pd
from config import (
    BOOSTING_LEARNING_RATE,
    BOOSTING_MAX_DEPTH,
    BOOSTING_N_ESTIMATORS,
    MLP_HIDDEN_LAYER_SIZES,
    MLP_MAX_ITER,
    MODEL_PATH,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
)
from mlflow.tracking import MlflowClient
from mlflow_registry import log_metadata, set_model_version_tags, transition_model_stage
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier

try:
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature

    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False
    infer_signature = None


class ModelType(str, Enum):
    """Типы поддерживаемых моделей."""

    RANDOM_FOREST = "random_forest"
    BOOSTING = "boosting"
    MLP = "mlp"


def train_model(
    model_type: str | ModelType,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    boosting_n_estimators: int | None = None,
    boosting_max_depth: int | None = None,
    boosting_learning_rate: float | None = None,
    mlp_hidden_layer_sizes: tuple[int, ...] | None = None,
    mlp_max_iter: int | None = None,
    random_state: int | None = None,
) -> RandomForestClassifier | GradientBoostingClassifier | MLPClassifier:
    """Обучает модель указанного типа."""
    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower())

    random_state = random_state or RANDOM_STATE

    if model_type == ModelType.RANDOM_FOREST:
        model = RandomForestClassifier(
            n_estimators=rf_n_estimators or RF_N_ESTIMATORS,
            max_depth=rf_max_depth or RF_MAX_DEPTH,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == ModelType.BOOSTING:
        model = GradientBoostingClassifier(
            n_estimators=boosting_n_estimators or BOOSTING_N_ESTIMATORS,
            max_depth=boosting_max_depth or BOOSTING_MAX_DEPTH,
            learning_rate=boosting_learning_rate or BOOSTING_LEARNING_RATE,
            random_state=random_state,
        )
    elif model_type == ModelType.MLP:
        model = MLPClassifier(
            hidden_layer_sizes=mlp_hidden_layer_sizes or MLP_HIDDEN_LAYER_SIZES,
            max_iter=mlp_max_iter or MLP_MAX_ITER,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier | GradientBoostingClassifier | MLPClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
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


def run_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str | ModelType = ModelType.BOOSTING,
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    boosting_n_estimators: int | None = None,
    boosting_max_depth: int | None = None,
    boosting_learning_rate: float | None = None,
    mlp_hidden_layer_sizes: tuple[int, ...] | None = None,
    mlp_max_iter: int | None = None,
    random_state: int | None = None,
    use_mlflow: bool = True,
    experiment_name: str | None = None,
    model_artifact_path: str = "model",
    save_local: bool = True,
    register_model_name: str | None = None,
) -> dict[str, Any]:
    """Запускает тренировку, оценку и логгирование."""
    if use_mlflow and not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow не установлен. Установите mlflow: pip install mlflow")

    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower())

    model = train_model(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        boosting_n_estimators=boosting_n_estimators,
        boosting_max_depth=boosting_max_depth,
        boosting_learning_rate=boosting_learning_rate,
        mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
        mlp_max_iter=mlp_max_iter,
        random_state=random_state,
    )

    metrics = evaluate_model(model, X_test, y_test)
    mlflow_run_id = None

    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name or "wine_quality")

        with mlflow.start_run() as run:
            mlflow_run_id = run.info.run_id

            # Логируем параметры
            params: dict[str, Any] = {
                "model_type": model_type.value,
                "random_state": random_state or RANDOM_STATE,
            }
            if model_type == ModelType.RANDOM_FOREST:
                params.update(
                    {"rf_n_estimators": model.n_estimators, "rf_max_depth": model.max_depth}
                )
            elif model_type == ModelType.BOOSTING:
                params.update(
                    {
                        "boosting_n_estimators": model.n_estimators,
                        "boosting_max_depth": model.max_depth,
                        "boosting_learning_rate": model.learning_rate,
                    }
                )
            elif model_type == ModelType.MLP:
                params.update(
                    {
                        "mlp_hidden_layer_sizes": str(model.hidden_layer_sizes),
                        "mlp_max_iter": model.max_iter,
                    }
                )

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", float(metrics["accuracy"]))
            mlflow.log_metric("f1_weighted", float(metrics["f1"]))

            # Логируем модель
            kwargs = {}
            if infer_signature and X_train is not None:
                try:
                    kwargs["signature"] = infer_signature(X_train, model.predict(X_train))
                except Exception as exc:
                    # Сигнатура не критична, продолжаем без неё
                    print(f"Warning: could not infer signature: {exc}")

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_artifact_path,
                registered_model_name=register_model_name,
                **kwargs,
            )

            log_metadata(
                {
                    "feature_columns": list(X_train.columns)
                    if hasattr(X_train, "columns")
                    else None,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
            )

            # Устанавливаем теги и переводим в Staging
            if register_model_name:
                client = MlflowClient()
                versions = [
                    v
                    for v in client.get_latest_versions(
                        register_model_name, stages=["None", "Staging", "Production", "Archived"]
                    )
                    if v.run_id == run.info.run_id
                ]
                if versions:
                    version = str(versions[0].version)
                    tags = {
                        "accuracy": float(metrics["accuracy"]),
                        "f1_weighted": float(metrics["f1"]),
                        "model_type": model_type.value,
                    }
                    if hasattr(model, "n_estimators"):
                        tags["n_estimators"] = model.n_estimators
                    set_model_version_tags(register_model_name, version, tags)
                    transition_model_stage(register_model_name, version, "Staging")

    if save_local:
        save_model(model)

    return {"model": model, "metrics": metrics, "mlflow_run_id": mlflow_run_id}
