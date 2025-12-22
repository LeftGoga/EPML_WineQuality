from __future__ import annotations

import logging
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from clearml import Model, OutputModel, Task

    CLEARML_AVAILABLE = True
    ClearMLTask = Task
except ImportError:
    CLEARML_AVAILABLE = False
    Task = None
    OutputModel = None
    Model = None
    ClearMLTask = None


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


def log_metrics_to_clearml(metrics: dict[str, float | int], iteration: int = 0) -> None:
    """Логирует метрики в ClearML в раздел Scalars.

    Args:
        metrics: Словарь с метриками (имя метрики -> значение)
        iteration: Номер итерации для логирования (по умолчанию 0)
    """
    if not CLEARML_AVAILABLE:
        logger.debug("ClearML не доступен, пропускаем логирование метрик")
        return

    task = get_current_task()
    if not task:
        logger.debug("Нет активной задачи ClearML")
        return

    try:
        # Подготавливаем метрики для логирования через connect (отображаются в UI)
        metrics_for_connect = {}

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                float_value = float(metric_value)

                # Логируем через report_scalar (для раздела Scalars)
                task.logger.report_scalar(
                    title="Metrics",
                    series=metric_name,
                    value=float_value,
                    iteration=iteration,
                )

                # Также логируем через report_single_value для лучшей видимости
                try:
                    task.logger.report_single_value(
                        title="Metrics",
                        series=metric_name,
                        value=float_value,
                        iteration=iteration,
                    )
                except Exception:  # nosec B110
                    # Если метод не поддерживается, просто пропускаем
                    pass

                # Сохраняем для логирования через connect
                metrics_for_connect[f"metrics/{metric_name}"] = float_value
                metrics_for_connect[metric_name] = float_value

        # Логируем метрики через connect для отображения в UI задачи
        if metrics_for_connect:
            task.connect(metrics_for_connect)

        logger.info(f"Метрики логированы в ClearML Scalars: {list(metrics.keys())}")
    except Exception as e:
        logger.warning(f"Ошибка при логировании метрик в ClearML: {e}")
        logger.debug(traceback.format_exc())


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


def get_model_version_info(model_name: str, task: Task | None = None) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
    """Получает информацию о версиях модели из ClearML.

    Ищет модели глобально по проекту во всех задачах.

    Args:
        model_name: Базовое имя модели (без версии)
        task: Задача ClearML (если None, используется текущая задача)

    Returns:
        Словарь с информацией о версиях:
        - version_count: количество найденных версий
        - latest_version: номер последней версии (целое число)
        - next_version: следующий номер версии для инкремента
    """
    if not CLEARML_AVAILABLE or Model is None:
        logger.warning("ClearML недоступен, используем версию 1")
        return {"version_count": 0, "latest_version": 1, "next_version": 1}

    if task is None:
        task = get_current_task()

    project_name = None
    if task:
        try:
            if hasattr(task, "get_project_name"):
                project_name = task.get_project_name()
            elif hasattr(task, "project"):
                project_name = task.project
            elif hasattr(task, "get_project"):
                project_name = task.get_project()
        except Exception:  # nosec B110
            pass

    try:
        all_models = []
        logger.info(f"Поиск моделей с именем '{model_name}' в проекте '{project_name}'")

        if project_name:
            try:
                try:
                    project_tasks = ClearMLTask.get_tasks(
                        project_name=project_name,
                        task_filter={"status": ["completed", "failed", "stopped"]},
                    )
                except (TypeError, ValueError):
                    try:
                        project_tasks = ClearMLTask.get_tasks(project_name=project_name)
                    except Exception:
                        logger.warning(
                            "Не удалось получить задачи через Task.get_tasks, используем версию 1"
                        )
                        project_tasks = []

                logger.info(f"Найдено {len(project_tasks)} задач в проекте")

                for project_task in project_tasks:
                    try:
                        task_models_dict = (
                            project_task.models if hasattr(project_task, "models") else {}
                        )

                        output_models = []
                        if isinstance(task_models_dict, dict):
                            output_models = task_models_dict.get("output", [])
                        elif hasattr(task_models_dict, "output"):
                            output_models = task_models_dict.output

                        for model in output_models:
                            model_obj = None
                            model_name_found = None

                            if hasattr(model, "name"):
                                model_name_found = model.name
                                model_obj = model
                            elif hasattr(model, "get_name"):
                                model_name_found = model.get_name()
                                model_obj = model
                            elif isinstance(model, dict):
                                model_name_found = model.get("name")
                                model_obj = model

                            if model_name_found and model_obj:
                                if model_name_found == model_name or model_name_found.startswith(
                                    f"{model_name}_v"
                                ):
                                    logger.debug(
                                        f"Найдена модель '{model_name_found}' в задаче {project_task.id}"
                                    )
                                    all_models.append(model_obj)
                    except Exception as e:
                        logger.debug(
                            f"Ошибка при обработке задачи {getattr(project_task, 'id', 'unknown')}: {e}"
                        )
                        continue
            except Exception as e:
                logger.warning(f"Не удалось найти модели через Task.get_tasks: {e}")
                logger.debug(traceback.format_exc())

        if not all_models and task:
            logger.info("Модели не найдены в проекте, ищем в текущей задаче")
            try:
                models_dict = task.models if hasattr(task, "models") else {}
                output_models = (
                    models_dict.get("output", []) if isinstance(models_dict, dict) else []
                )
                for model in output_models:
                    model_name_found = None
                    if hasattr(model, "name"):
                        model_name_found = model.name
                    elif isinstance(model, dict):
                        model_name_found = model.get("name")

                    if model_name_found and (
                        model_name_found == model_name
                        or model_name_found.startswith(f"{model_name}_v")
                    ):
                        logger.debug(f"Найдена модель '{model_name_found}' в текущей задаче")
                        all_models.append(model)
            except Exception as e:
                logger.debug(f"Не удалось найти модели в задаче: {e}")

        logger.info(f"Всего найдено моделей с именем '{model_name}': {len(all_models)}")

        version_count = len(all_models)

        if version_count == 0:
            logger.info("Модели не найдены в ClearML, используем версию 1")
            return {
                "version_count": 0,
                "latest_version": 1,
                "next_version": 1,
            }

        versions = []
        for model in all_models:
            version_str = None

            try:
                if hasattr(model, "get_metadata"):
                    version_str = model.get_metadata("version") or model.get_metadata(
                        "model_version"
                    )
                elif hasattr(model, "metadata"):
                    metadata = model.metadata
                    if isinstance(metadata, dict):
                        version_str = metadata.get("version") or metadata.get("model_version")
                    elif hasattr(metadata, "get"):
                        version_str = metadata.get("version") or metadata.get("model_version")
            except Exception as e:
                logger.debug(f"Не удалось получить metadata: {e}")

            if not version_str:
                labels_dict = None
                if hasattr(model, "labels"):
                    labels_dict = model.labels
                    if isinstance(labels_dict, dict):
                        version_str = labels_dict.get("version") or labels_dict.get("model_version")
                    elif hasattr(labels_dict, "get"):
                        version_str = labels_dict.get("version") or labels_dict.get("model_version")
                elif hasattr(model, "get_labels"):
                    labels_dict = model.get_labels()
                    if isinstance(labels_dict, dict):
                        version_str = labels_dict.get("version") or labels_dict.get("model_version")
                elif isinstance(model, dict):
                    labels_dict = model.get("labels", {})
                    version_str = (
                        labels_dict.get("version") or labels_dict.get("model_version")
                        if isinstance(labels_dict, dict)
                        else None
                    )

            if version_str:
                try:
                    if isinstance(version_str, str) and "." in version_str:
                        version = int(float(version_str))
                    else:
                        version = int(version_str)
                    versions.append(version)
                    logger.debug(f"Найдена версия {version} для модели '{model_name}'")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Не удалось распарсить версию '{version_str}': {e}")

        if versions:
            latest_version = max(versions)
            logger.info(f"Найдены версии: {versions}, последняя: {latest_version}")
        else:
            latest_version = version_count
            logger.info(
                f"Версии в metadata не найдены, используем количество моделей: {latest_version}"
            )

        next_version = latest_version + 1
        logger.info(f"Следующая версия для модели '{model_name}': {next_version}")

        result = {
            "version_count": version_count,
            "latest_version": latest_version,
            "next_version": next_version,
        }

        return result
    except Exception as e:
        logger.warning(f"Ошибка при получении информации о версиях модели: {e}")
        return {"version_count": 0, "latest_version": 1, "next_version": 1}


def log_model_to_clearml(  # noqa: PLR0912, PLR0915
    model: Any,
    model_name: str,
    model_path: str | Path | None = None,
    framework: str = "scikit-learn",
    tags: list[str] | None = None,
    labels: dict[str, str] | None = None,
    auto_version: bool = True,
) -> int | None:
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
        Номер версии модели (целое число: 1, 2, 3...) или None в случае ошибки
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
        model_version = version_info["next_version"] if version_info else 1

        model_labels = labels.copy() if labels else {}
        if auto_version and version_info:
            model_labels["version"] = str(model_version)
            model_labels["model_version"] = str(model_version)
            model_labels["version_count"] = str(version_info["version_count"] + 1)
            model_labels["created_at"] = datetime.now().isoformat()
            if version_info["version_count"] > 0:
                model_labels["previous_version"] = str(version_info["latest_version"])

        if model_path:
            model_path = Path(model_path)
            if model_path.exists():
                model_name_with_version = f"{model_name}_v{model_version}"
                output_model = OutputModel(
                    task=task, name=model_name_with_version, framework=framework
                )
                output_model.update_weights(str(model_path))

                model_tags = list(tags) if tags else []
                model_tags.append(f"version_{model_version}")
                output_model.tags = model_tags

                for key, value in model_labels.items():
                    try:
                        output_model.set_metadata(key, str(value))
                    except Exception as e:
                        logger.debug(f"Не удалось установить metadata {key}: {e}")
                        try:
                            if hasattr(output_model, "labels"):
                                output_model.labels[key] = str(value)
                        except Exception:  # nosec B110
                            pass

                # Сохраняем метрики для модели (без логирования в Scalars, чтобы избежать дублирования)
                # Метрики логируются только в главной задаче пайплайна
                metrics_for_model = {}
                for key, value in model_labels.items():
                    # Извлекаем числовые метрики (accuracy, f1_weighted и т.д.)
                    if key in ["accuracy", "f1_weighted", "f1", "precision", "recall", "roc_auc"]:
                        try:
                            metric_value = float(value)
                            metrics_for_model[key] = metric_value
                            # Не логируем через report_scalar, чтобы избежать дублирования
                            # Метрики уже логируются в главной задаче пайплайна
                        except (ValueError, TypeError):
                            pass

                # Логируем метрики через connect модели для отображения в UI модели (но не в Scalars)
                if metrics_for_model:
                    try:
                        output_model.connect(metrics_for_model)
                        logger.debug(
                            f"Метрики модели сохранены в metadata: {list(metrics_for_model.keys())}"
                        )
                    except Exception as e:
                        logger.debug(f"Не удалось сохранить метрики через connect модели: {e}")

                logger.info(
                    f"Модель логирована в ClearML: {model_name_with_version}, версия: {model_version}"
                )
                logger.info(
                    f"Версия сохранена в metadata: version={model_version}, model_version={model_version}"
                )
                logger.info(f"Всего метаданных сохранено: {len(model_labels)}")
                return model_version
            else:
                logger.warning(f"Файл модели не найден: {model_path}")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                joblib.dump(model, tmp_path)
                model_name_with_version = f"{model_name}_v{model_version}"
                output_model = OutputModel(
                    task=task, name=model_name_with_version, framework=framework
                )
                output_model.update_weights(tmp_path)

                model_tags = list(tags) if tags else []
                model_tags.append(f"version_{model_version}")
                output_model.tags = model_tags

                for key, value in model_labels.items():
                    try:
                        output_model.set_metadata(key, str(value))
                    except Exception as e:
                        logger.debug(f"Не удалось установить metadata {key}: {e}")
                        try:
                            if hasattr(output_model, "labels"):
                                output_model.labels[key] = str(value)
                        except Exception:  # nosec B110
                            pass

                # Сохраняем метрики для модели (без логирования в Scalars, чтобы избежать дублирования)
                # Метрики логируются только в главной задаче пайплайна
                metrics_for_model = {}
                for key, value in model_labels.items():
                    # Извлекаем числовые метрики (accuracy, f1_weighted и т.д.)
                    if key in ["accuracy", "f1_weighted", "f1", "precision", "recall", "roc_auc"]:
                        try:
                            metric_value = float(value)
                            metrics_for_model[key] = metric_value
                            # Не логируем через report_scalar, чтобы избежать дублирования
                            # Метрики уже логируются в главной задаче пайплайна
                        except (ValueError, TypeError):
                            pass

                # Логируем метрики через connect модели для отображения в UI модели (но не в Scalars)
                if metrics_for_model:
                    try:
                        output_model.connect(metrics_for_model)
                        logger.debug(
                            f"Метрики модели сохранены в metadata: {list(metrics_for_model.keys())}"
                        )
                    except Exception as e:
                        logger.debug(f"Не удалось сохранить метрики через connect модели: {e}")

                logger.info(
                    f"Модель логирована в ClearML: {model_name_with_version}, версия: {model_version}"
                )
                logger.info(
                    f"Версия сохранена в metadata: version={model_version}, model_version={model_version}"
                )
                logger.info(f"Всего метаданных сохранено: {len(model_labels)}")
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


def compare_clearml_models(  # noqa: PLR0912, PLR0915
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
            if hasattr(model, "labels"):
                labels = model.labels
            elif isinstance(model, dict):
                labels = model.get("labels", {})
            else:
                labels = {}

            version_str = labels.get("version", "1")
            try:
                if isinstance(version_str, str) and "." in version_str:
                    version = int(float(version_str))
                else:
                    version = int(version_str)
            except (ValueError, TypeError):
                version = 1

            version_data: dict[str, Any] = {
                "version": version,
                "version_str": str(version),
                "created_at": labels.get("created_at", "unknown"),
                "model_type": labels.get("model_type", "unknown"),
                "metrics": {},
                "labels": dict(labels) if labels else {},
            }

            for metric_key in metric_keys:
                metric_value = labels.get(metric_key)
                if metric_value:
                    try:
                        version_data["metrics"][metric_key] = float(metric_value)
                    except (ValueError, TypeError):
                        version_data["metrics"][metric_key] = metric_value

            accuracy = version_data["metrics"].get("accuracy", 0.0)
            if isinstance(accuracy, (int, float)) and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_version = version_data["version"]

            versions_data.append(version_data)

        try:
            versions_data.sort(
                key=lambda x: x.get("version", 0),
                reverse=True,
            )
        except Exception:  # nosec B110
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
            f"лучшая версия: {best_version} (accuracy: {best_accuracy:.4f})"
        )

        return result

    except Exception as e:
        logger.warning(f"Ошибка при сравнении моделей в ClearML: {e}")
        return {"model_name": model_name, "versions": [], "best_version": None}
