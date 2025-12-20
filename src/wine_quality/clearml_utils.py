from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
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


def get_model_version_info(model_name: str, task: Task | None = None) -> dict[str, Any]:  # noqa: PLR0912
    """Получает информацию о версиях модели из ClearML.

    Args:
        model_name: Имя модели
        task: Задача ClearML (если None, используется текущая задача)

    Returns:
        Словарь с информацией о версиях модели:
        - 'version_count': количество версий
        - 'latest_version': номер последней версии
        - 'next_version': следующий номер версии для инкремента
    """
    if not CLEARML_AVAILABLE:
        return {"version_count": 0, "latest_version": "1.0", "next_version": "1.0"}

    if task is None:
        task = get_current_task()

    if not task:
        return {"version_count": 0, "latest_version": "1.0", "next_version": "1.0"}

    try:
        # Получаем все модели из задачи
        # В ClearML модели хранятся в task.models.output
        models_dict = task.models if hasattr(task, "models") else {}
        output_models = models_dict.get("output", [])

        # Фильтруем модели по имени
        model_versions = []
        for model in output_models:
            if hasattr(model, "name") and model.name == model_name:
                model_versions.append(model)
            elif isinstance(model, dict) and model.get("name") == model_name:
                model_versions.append(model)

        version_count = len(model_versions)

        if version_count == 0:
            return {
                "version_count": 0,
                "latest_version": "1.0",
                "next_version": "1.0",
            }

        # Получаем последнюю версию (последний элемент в списке)
        latest_model = model_versions[-1]

        # Пытаемся получить версию из labels
        latest_version = "1.0"
        if hasattr(latest_model, "labels"):
            latest_version = latest_model.labels.get("version", f"{version_count}.0")
        elif isinstance(latest_model, dict):
            latest_version = latest_model.get("labels", {}).get("version", f"{version_count}.0")

        if isinstance(latest_version, str):
            try:
                # Парсим версию вида "major.minor"
                parts = latest_version.split(".")
                if len(parts) == 2:
                    major, minor = int(parts[0]), int(parts[1])
                    # Инкрементируем минорную версию
                    next_version = f"{major}.{minor + 1}"
                else:
                    # Если формат неверный, используем счетчик
                    next_version = f"{version_count + 1}.0"
            except (ValueError, IndexError):
                next_version = f"{version_count + 1}.0"
        else:
            next_version = f"{version_count + 1}.0"

        return {
            "version_count": version_count,
            "latest_version": str(latest_version),
            "next_version": next_version,
        }
    except Exception as e:
        logger.warning(f"Ошибка при получении информации о версиях модели: {e}")
        # В случае ошибки возвращаем следующую версию на основе счетчика
        return {"version_count": 0, "latest_version": "1.0", "next_version": "1.0"}


def log_model_to_clearml(  # noqa: PLR0912
    model: Any,
    model_name: str,
    model_path: str | Path | None = None,
    framework: str = "scikit-learn",
    tags: list[str] | None = None,
    labels: dict[str, str] | None = None,
    auto_version: bool = True,
) -> str | None:
    """Логирует модель в ClearML с автоматическим версионированием.

    Args:
        model: Модель для логирования
        model_name: Имя модели
        model_path: Путь к файлу модели
        framework: Фреймворк модели (по умолчанию "scikit-learn")
        tags: Список тегов для модели
        labels: Словарь меток для модели
        auto_version: Автоматически инкрементировать версию (по умолчанию True)

    Returns:
        Номер версии модели или None в случае ошибки
    """
    if not CLEARML_AVAILABLE:
        logger.debug("ClearML не доступен, пропускаем логирование модели")
        return None

    task = get_current_task()
    if not task:
        logger.debug("Нет активной задачи ClearML")
        return None

    try:
        # Получаем информацию о версиях модели
        version_info = get_model_version_info(model_name, task) if auto_version else None
        model_version = version_info["next_version"] if version_info else "1.0"

        # Подготавливаем labels с метаданными версии
        model_labels = labels.copy() if labels else {}
        if auto_version and version_info:
            model_labels["version"] = model_version
            model_labels["version_count"] = str(version_info["version_count"] + 1)
            model_labels["created_at"] = datetime.now().isoformat()
            if version_info["version_count"] > 0:
                model_labels["previous_version"] = version_info["latest_version"]

        if model_path:
            model_path = Path(model_path)
            if model_path.exists():
                output_model = OutputModel(task=task, name=model_name, framework=framework)
                output_model.update_weights(str(model_path))
                if tags:
                    output_model.tags = tags
                for key, value in model_labels.items():
                    output_model.labels[key] = str(value)
                logger.info(f"Модель логирована в ClearML: {model_name}, версия: {model_version}")
                return model_version
            else:
                logger.warning(f"Файл модели не найден: {model_path}")

        # Если путь не указан или файл не найден, сохраняем модель во временный файл
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                joblib.dump(model, tmp_path)
                output_model = OutputModel(task=task, name=model_name, framework=framework)
                output_model.update_weights(tmp_path)
                if tags:
                    output_model.tags = tags
                for key, value in model_labels.items():
                    output_model.labels[key] = str(value)
                logger.info(f"Модель логирована в ClearML: {model_name}, версия: {model_version}")
                return model_version
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    except Exception as e:
        logger.warning(f"Ошибка при логировании модели в ClearML: {e}")
        return None


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


def compare_clearml_models(  # noqa: PLR0912
    model_name: str,
    task: Task | None = None,
    metric_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Сравнивает версии модели в ClearML.

    Args:
        model_name: Имя модели для сравнения
        task: Задача ClearML (если None, используется текущая задача)
        metric_keys: Список ключей метрик для сравнения (например, ['accuracy', 'f1_weighted'])

    Returns:
        Словарь с результатами сравнения:
        - 'model_name': имя модели
        - 'versions': список версий с их метриками и метаданными
        - 'best_version': версия с лучшими метриками
    """
    if not CLEARML_AVAILABLE:
        logger.warning("ClearML не доступен для сравнения моделей")
        return {"model_name": model_name, "versions": [], "best_version": None}

    if task is None:
        task = get_current_task()

    if not task:
        logger.warning("Нет активной задачи ClearML для сравнения моделей")
        return {"model_name": model_name, "versions": [], "best_version": None}

    if metric_keys is None:
        metric_keys = ["accuracy", "f1_weighted"]

    try:
        # Получаем все версии модели
        models_dict = task.models if hasattr(task, "models") else {}
        output_models = models_dict.get("output", [])

        # Фильтруем модели по имени
        model_versions = []
        for model in output_models:
            if hasattr(model, "name") and model.name == model_name:
                model_versions.append(model)
            elif isinstance(model, dict) and model.get("name") == model_name:
                model_versions.append(model)

        if not model_versions:
            logger.warning(f"Модель '{model_name}' не найдена")
            return {"model_name": model_name, "versions": [], "best_version": None}

        versions_data = []
        best_version = None
        best_accuracy = -1.0

        for model in model_versions:
            # Получаем labels из модели
            if hasattr(model, "labels"):
                labels = model.labels
            elif isinstance(model, dict):
                labels = model.get("labels", {})
            else:
                labels = {}

            version_data: dict[str, Any] = {
                "version": labels.get("version", "unknown"),
                "created_at": labels.get("created_at", "unknown"),
                "model_type": labels.get("model_type", "unknown"),
                "metrics": {},
                "labels": dict(labels) if labels else {},
            }

            # Извлекаем метрики из labels
            for metric_key in metric_keys:
                metric_value = labels.get(metric_key)
                if metric_value:
                    try:
                        version_data["metrics"][metric_key] = float(metric_value)
                    except (ValueError, TypeError):
                        version_data["metrics"][metric_key] = metric_value

            # Определяем лучшую версию по accuracy
            accuracy = version_data["metrics"].get("accuracy", 0.0)
            if isinstance(accuracy, (int, float)) and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_version = version_data["version"]

            versions_data.append(version_data)

        # Сортируем версии по дате создания (если доступна)
        try:
            versions_data.sort(
                key=lambda x: x.get("created_at", ""),
                reverse=True,
            )
        except Exception:  # nosec B110
            pass

        result = {
            "model_name": model_name,
            "versions": versions_data,
            "best_version": best_version,
            "total_versions": len(versions_data),
        }

        logger.info(
            f"Сравнение моделей '{model_name}': найдено {len(versions_data)} версий, "
            f"лучшая версия: {best_version}"
        )

        return result

    except Exception as e:
        logger.warning(f"Ошибка при сравнении моделей в ClearML: {e}")
        return {"model_name": model_name, "versions": [], "best_version": None}
