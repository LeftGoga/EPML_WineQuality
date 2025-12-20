from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from clearml import OutputModel, Task

    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    Task = None
    OutputModel = None


def get_current_task() -> Task | None:
    """Получает текущую активную задачу ClearML."""
    if not CLEARML_AVAILABLE:
        return None

    try:
        return Task.current_task()
    except Exception:
        return None


def log_params_to_clearml(params: dict[str, Any]) -> None:
    """Логирует параметры в ClearML."""
    if not CLEARML_AVAILABLE:
        logger.debug("ClearML не доступен, пропускаем логирование параметров")
        return

    task = get_current_task()
    if not task:
        logger.debug("Нет активной задачи ClearML")
        return

    try:
        str_params = {k: str(v) for k, v in params.items()}
        task.connect(str_params)
        logger.debug(f"Параметры логированы в ClearML: {list(str_params.keys())}")
    except Exception as e:
        logger.warning(f"Ошибка при логировании параметров в ClearML: {e}")


def log_metrics_to_clearml(metrics: dict[str, float | int]) -> None:
    """Логирует метрики в ClearML."""
    if not CLEARML_AVAILABLE:
        logger.debug("ClearML не доступен, пропускаем логирование метрик")
        return

    task = get_current_task()
    if not task:
        logger.debug("Нет активной задачи ClearML")
        return

    try:
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                task.logger.report_scalar(
                    title="Metrics",
                    series=metric_name,
                    value=float(metric_value),
                    iteration=0,
                )
        logger.debug(f"Метрики логированы в ClearML: {list(metrics.keys())}")
    except Exception as e:
        logger.warning(f"Ошибка при логировании метрик в ClearML: {e}")


def log_plot_to_clearml(
    plot_path: str | Path,
    title: str = "Plot",
    series: str = "default",
    iteration: int = 0,
) -> None:
    """Логирует график в ClearML."""
    if not CLEARML_AVAILABLE:
        logger.debug("ClearML не доступен, пропускаем логирование графика")
        return

    task = get_current_task()
    if not task:
        logger.debug("Нет активной задачи ClearML")
        return

    plot_path = Path(plot_path)
    if not plot_path.exists():
        logger.warning(f"Файл графика не найден: {plot_path}")
        return

    try:
        task.logger.report_image(
            title=title,
            series=series,
            local_path=str(plot_path),
            iteration=iteration,
        )
        logger.debug(f"График логирован в ClearML: {plot_path.name}")
    except Exception as e:
        logger.warning(f"Ошибка при логировании графика в ClearML: {e}")


def log_artifact_to_clearml(
    artifact_path: str | Path,
    artifact_name: str | None = None,
    delete_after_upload: bool = False,
) -> None:
    """Логирует артефакт в ClearML."""
    if not CLEARML_AVAILABLE:
        logger.debug("ClearML не доступен, пропускаем логирование артефакта")
        return

    task = get_current_task()
    if not task:
        logger.debug("Нет активной задачи ClearML")
        return

    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        logger.warning(f"Артефакт не найден: {artifact_path}")
        return

    try:
        artifact_name = artifact_name or artifact_path.name
        if artifact_path.is_file():
            task.upload_artifact(
                name=artifact_name,
                artifact_object=str(artifact_path),
                delete_after_upload=delete_after_upload,
            )
        elif artifact_path.is_dir():
            task.upload_artifact(
                name=artifact_name,
                artifact_object=str(artifact_path),
                delete_after_upload=delete_after_upload,
            )
        logger.debug(f"Артефакт логирован в ClearML: {artifact_name}")
    except Exception as e:
        logger.warning(f"Ошибка при логировании артефакта в ClearML: {e}")


def log_model_to_clearml(  # noqa: PLR0912
    model: Any,
    model_name: str,
    model_path: str | Path | None = None,
    framework: str = "scikit-learn",
    tags: list[str] | None = None,
    labels: dict[str, str] | None = None,
) -> None:
    """Логирует модель в ClearML."""
    if not CLEARML_AVAILABLE:
        logger.debug("ClearML не доступен, пропускаем логирование модели")
        return

    task = get_current_task()
    if not task:
        logger.debug("Нет активной задачи ClearML")
        return

    try:
        if model_path:
            model_path = Path(model_path)
            if model_path.exists():
                output_model = OutputModel(task=task, name=model_name, framework=framework)
                output_model.update_weights(str(model_path))
                if tags:
                    output_model.tags = tags
                if labels:
                    for key, value in labels.items():
                        output_model.labels[key] = value
                logger.debug(f"Модель логирована в ClearML: {model_name}")
                return

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                joblib.dump(model, tmp_path)
                output_model = OutputModel(task=task, name=model_name, framework=framework)
                output_model.update_weights(tmp_path)
                if tags:
                    output_model.tags = tags
                if labels:
                    for key, value in labels.items():
                        output_model.labels[key] = value
                logger.debug(f"Модель логирована в ClearML: {model_name}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    except Exception as e:
        logger.warning(f"Ошибка при логировании модели в ClearML: {e}")


def set_clearml_tags(tags: list[str]) -> None:
    """Устанавливает теги для текущей задачи ClearML."""
    if not CLEARML_AVAILABLE:
        logger.debug("ClearML не доступен, пропускаем установку тегов")
        return

    task = get_current_task()
    if not task:
        logger.debug("Нет активной задачи ClearML")
        return

    try:
        task.add_tags(tags)
        logger.debug(f"Теги установлены в ClearML: {tags}")
    except Exception as e:
        logger.warning(f"Ошибка при установке тегов в ClearML: {e}")


def log_dataframe_to_clearml(
    df: Any,
    name: str = "dataframe",
    title: str = "Data",
) -> None:
    """Логирует DataFrame в ClearML как таблицу."""
    if not CLEARML_AVAILABLE:
        logger.debug("ClearML не доступен, пропускаем логирование DataFrame")
        return

    task = get_current_task()
    if not task:
        logger.debug("Нет активной задачи ClearML")
        return

    try:
        if isinstance(df, pd.DataFrame):
            task.logger.report_table(
                title=title,
                series=name,
                table_plot=df,
                iteration=0,
            )
            logger.debug(f"DataFrame логирован в ClearML: {name}")
        else:
            logger.warning(f"Объект не является pandas DataFrame: {type(df)}")
    except Exception as e:
        logger.warning(f"Ошибка при логировании DataFrame в ClearML: {e}")


def log_confusion_matrix_to_clearml(
    confusion_matrix: Any,
    title: str = "Confusion Matrix",
    series: str = "metrics",
) -> None:
    """Логирует confusion matrix в ClearML."""
    if not CLEARML_AVAILABLE:
        logger.debug("ClearML не доступен, пропускаем логирование confusion matrix")
        return

    task = get_current_task()
    if not task:
        logger.debug("Нет активной задачи ClearML")
        return

    try:
        if isinstance(confusion_matrix, np.ndarray):
            task.logger.report_confusion_matrix(
                title=title,
                series=series,
                matrix=confusion_matrix,
                iteration=0,
                xaxis="Predicted",
                yaxis="Actual",
            )
            logger.debug("Confusion matrix логирована в ClearML")
        else:
            logger.warning(f"Объект не является numpy array: {type(confusion_matrix)}")
    except Exception as e:
        logger.warning(f"Ошибка при логировании confusion matrix в ClearML: {e}")
