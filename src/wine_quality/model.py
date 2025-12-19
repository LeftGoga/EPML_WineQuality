from __future__ import annotations

import logging
import os
import shutil
import tempfile
import traceback
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier

try:
    from .config import (
        BOOSTING_LEARNING_RATE,
        BOOSTING_MAX_DEPTH,
        BOOSTING_N_ESTIMATORS,
        MLFLOW_EXPERIMENT_NAME,
        MLP_HIDDEN_LAYER_SIZES,
        MLP_MAX_ITER,
        MODEL_PATH,
        RANDOM_STATE,
        RF_MAX_DEPTH,
        RF_N_ESTIMATORS,
    )
    from .mlflow_context import (
        MLflowExperimentContext,
        MLflowRunContext,
    )
    from .mlflow_decorators import log_metrics, log_params
    from .mlflow_registry import log_metadata, set_model_version_tags, transition_model_stage
except ImportError:
    from config import (
        BOOSTING_LEARNING_RATE,
        BOOSTING_MAX_DEPTH,
        BOOSTING_N_ESTIMATORS,
        MLFLOW_EXPERIMENT_NAME,
        MLP_HIDDEN_LAYER_SIZES,
        MLP_MAX_ITER,
        MODEL_PATH,
        RANDOM_STATE,
        RF_MAX_DEPTH,
        RF_N_ESTIMATORS,
    )
    from mlflow_context import (
        MLflowExperimentContext,
        MLflowRunContext,
    )
    from mlflow_decorators import log_metrics, log_params
    from mlflow_registry import log_metadata, set_model_version_tags, transition_model_stage

try:
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature

    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False
    infer_signature = None

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    BOOSTING = "boosting"
    MLP = "mlp"


@log_params
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


@log_metrics(["accuracy", "f1_weighted"])
def evaluate_model(
    model: RandomForestClassifier | GradientBoostingClassifier | MLPClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1-score (weighted): {f1:.4f}")
    logger.debug("Classification Report:\n%s", classification_report(y_test, y_pred))
    logger.debug("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    metrics = {
        "accuracy": acc,
        "f1_weighted": f1,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def save_model(model: Any, path: str | None = None) -> None:
    if path is None:
        path = str(MODEL_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.debug(f"Модель сохранена: {path}")


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
    log_artifacts: bool = True,
    df_for_plots: pd.DataFrame | None = None,
) -> dict[str, Any]:
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
        try:
            client = MlflowClient()
            client.search_experiments(max_results=1)
        except Exception as e:
            logger.warning(f"Проблема с подключением к MLflow: {e}")

        exp_name = experiment_name or MLFLOW_EXPERIMENT_NAME
        with MLflowExperimentContext(exp_name):
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

            tags: dict[str, str] = {
                "framework": "sklearn",
                "task": "classification",
                "model_type": model_type.value,
            }

            with MLflowRunContext(
                experiment_name=exp_name,
                params=params,
                tags=tags,
            ) as run:
                mlflow_run_id = run.info.run_id

                mlflow.log_metric("accuracy", float(metrics["accuracy"]))
                mlflow.log_metric("f1_weighted", float(metrics["f1_weighted"]))

                signature = None
                if infer_signature and X_train is not None:
                    try:
                        signature = infer_signature(X_train, model.predict(X_train))
                    except Exception:  # nosec B110
                        pass

                active_run = mlflow.active_run()
                if active_run:
                    try:
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path=model_artifact_path,
                            signature=signature,
                        )

                        if register_model_name:
                            run_uri = f"runs:/{active_run.info.run_id}/{model_artifact_path}"
                            client = MlflowClient()
                            try:
                                client.get_registered_model(register_model_name)
                            except Exception:
                                client.create_registered_model(register_model_name)

                            mlflow.register_model(model_uri=run_uri, name=register_model_name)
                    except Exception as e:
                        logger.error(f"Ошибка при логировании модели: {e}")
                        logger.debug(traceback.format_exc())

                log_metadata(
                    {
                        "feature_columns": list(X_train.columns)
                        if hasattr(X_train, "columns")
                        else None,
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                    }
                )

                if log_artifacts:
                    active_run = mlflow.active_run()
                    if active_run:
                        temp_plots_dir = tempfile.mkdtemp()
                        plots_dir = Path(temp_plots_dir)
                        try:
                            if hasattr(model, "feature_importances_") and X_train is not None:
                                feature_plot_path = plots_dir / "feature_importances.png"
                                matplotlib.use("Agg")

                                importances = model.feature_importances_
                                indices = importances.argsort()[::-1]
                                ordered_names = [X_train.columns[i] for i in indices]

                                plt.figure(figsize=(10, 6))
                                model_name = model_type.value.replace("_", " ").title()
                                plt.title(f"Важность признаков ({model_name})")
                                plt.bar(
                                    range(len(importances)), importances[indices], align="center"
                                )
                                plt.xticks(range(len(importances)), ordered_names, rotation=90)
                                plt.tight_layout()
                                plt.savefig(feature_plot_path)
                                plt.close()

                                if (
                                    feature_plot_path.exists()
                                    and feature_plot_path.stat().st_size > 0
                                ):
                                    mlflow.log_artifact(
                                        str(feature_plot_path), artifact_path="plots"
                                    )

                            if df_for_plots is not None and len(df_for_plots) > 0:
                                corr_plot_path = plots_dir / "correlation_heatmap.png"
                                matplotlib.use("Agg")

                                plt.figure(figsize=(12, 9))
                                corr = df_for_plots.corr()
                                sns.heatmap(
                                    corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
                                )
                                plt.title("Матрица корреляций")
                                plt.tight_layout()
                                plt.savefig(corr_plot_path)
                                plt.close()

                                if corr_plot_path.exists() and corr_plot_path.stat().st_size > 0:
                                    mlflow.log_artifact(str(corr_plot_path), artifact_path="plots")
                        finally:
                            try:
                                shutil.rmtree(temp_plots_dir)
                            except Exception:  # nosec B110
                                pass

                if register_model_name:
                    try:
                        client = MlflowClient()
                        try:
                            all_versions = client.search_model_versions(
                                f"name='{register_model_name}'"
                            )
                            versions = [v for v in all_versions if v.run_id == run.info.run_id]
                        except Exception:
                            versions = []

                        if versions:
                            version = str(versions[0].version)
                            model_tags = {
                                "accuracy": float(metrics["accuracy"]),
                                "f1_weighted": float(metrics["f1_weighted"]),
                                "model_type": model_type.value,
                            }
                            if hasattr(model, "n_estimators"):
                                model_tags["n_estimators"] = model.n_estimators
                            set_model_version_tags(register_model_name, version, model_tags)
                            transition_model_stage(register_model_name, version, "Staging")
                    except Exception:  # nosec B110
                        pass

    if save_local:
        save_model(model)

    return {"model": model, "metrics": metrics, "mlflow_run_id": mlflow_run_id}
