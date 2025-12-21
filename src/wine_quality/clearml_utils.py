from __future__ import annotations

import json
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
    # Алиас для использования внутри функций
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


def _get_version_from_file(model_name: str) -> dict[str, Any] | None:
    """Получает информацию о версиях из локального файла (fallback).

    Args:
        model_name: Имя модели

    Returns:
        Словарь с информацией о версиях или None, если файл не найден
    """
    try:
        # Используем директорию models для хранения версий
        version_file = (
            Path(__file__).parent.parent.parent / "models" / f"{model_name}_versions.json"
        )
        logger.debug(f"Проверка файла версий: {version_file}")
        if version_file.exists():
            with open(version_file, encoding="utf-8") as f:
                data = json.load(f)
                result = {
                    "version_count": data.get("version_count", 0),
                    "latest_version": data.get("latest_version", 1),
                    "next_version": data.get("next_version", 1),
                }
                logger.info(f"Прочитана версия из файла для '{model_name}': {result}")
                return result
        else:
            logger.debug(f"Файл версий не найден: {version_file}")
    except Exception as e:
        logger.warning(f"Не удалось прочитать версию из файла: {e}")
    return None


def _save_version_to_file(model_name: str, version: int) -> None:
    """Сохраняет информацию о версии в локальный файл (fallback).

    Args:
        model_name: Имя модели
        version: Номер версии
    """
    try:
        version_file = (
            Path(__file__).parent.parent.parent / "models" / f"{model_name}_versions.json"
        )
        version_file.parent.mkdir(parents=True, exist_ok=True)

        # Читаем существующие данные
        data = {"version_count": 0, "latest_version": 1, "next_version": 1}
        if version_file.exists():
            try:
                with open(version_file, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.debug(f"Ошибка при чтении файла версий: {e}")

        # Обновляем данные
        old_version = data.get("latest_version", 1)
        data["latest_version"] = version
        data["version_count"] = data.get("version_count", 0) + 1
        data["next_version"] = version + 1

        # Сохраняем
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"Версия сохранена в файл для '{model_name}': {old_version} -> {version}, следующая: {data['next_version']}"
        )
    except Exception as e:
        logger.warning(f"Не удалось сохранить версию в файл: {e}")


def get_model_version_info(model_name: str, task: Task | None = None) -> dict[str, Any]:  # noqa: PLR0911, PLR0912, PLR0915
    """Получает информацию о версиях модели из ClearML.

    Ищет модели глобально по проекту, а не только в текущей задаче.
    Если поиск в ClearML не дает результатов, использует локальный файл как fallback.

    Args:
        model_name: Имя модели
        task: Задача ClearML (если None, используется текущая задача для получения проекта)

    Returns:
        Словарь с информацией о версиях модели:
        - 'version_count': количество версий
        - 'latest_version': номер последней версии (целое число)
        - 'next_version': следующий номер версии для инкремента (целое число)
    """
    if not CLEARML_AVAILABLE or Model is None:
        # Если ClearML недоступен, используем файловое хранилище
        file_version = _get_version_from_file(model_name)
        if file_version:
            return file_version
        return {"version_count": 0, "latest_version": 1, "next_version": 1}

    if task is None:
        task = get_current_task()

    project_name = None
    if task:
        try:
            # Пробуем разные способы получения имени проекта
            if hasattr(task, "get_project_name"):
                project_name = task.get_project_name()
            elif hasattr(task, "project"):
                project_name = task.project
            elif hasattr(task, "get_project"):
                project_name = task.get_project()
        except Exception:  # nosec B110
            pass

    try:
        # Ищем модели глобально по проекту
        # Проходим по всем задачам в проекте и ищем модели с нужным именем
        all_models = []

        logger.info(f"Поиск моделей с именем '{model_name}' в проекте '{project_name}'")

        if project_name:
            try:
                # Получаем все задачи в проекте
                # Пробуем разные варианты вызова get_tasks
                try:
                    project_tasks = ClearMLTask.get_tasks(
                        project_name=project_name,
                        task_filter={"status": ["completed", "failed", "stopped", "running"]},
                    )
                except TypeError:
                    # Если не поддерживается task_filter, пробуем без него
                    try:
                        project_tasks = ClearMLTask.get_tasks(project_name=project_name)
                    except Exception:
                        # Если и это не работает, используем файловое хранилище
                        logger.warning(
                            "Не удалось получить задачи через Task.get_tasks, используем файловое хранилище"
                        )
                        file_version = _get_version_from_file(model_name)
                        if file_version:
                            return file_version
                        project_tasks = []

                logger.info(f"Найдено {len(project_tasks)} задач в проекте")

                # Проходим по всем задачам и ищем модели
                for project_task in project_tasks:
                    try:
                        # Получаем модели из задачи
                        # В ClearML модели хранятся в task.models.output
                        task_models_dict = (
                            project_task.models if hasattr(project_task, "models") else {}
                        )

                        # Модели могут быть в разных форматах
                        output_models = []
                        if isinstance(task_models_dict, dict):
                            output_models = task_models_dict.get("output", [])
                        elif hasattr(task_models_dict, "output"):
                            output_models = task_models_dict.output

                        # Если output_models - это список OutputModel объектов
                        for model in output_models:
                            model_obj = None
                            model_name_found = None

                            # Получаем имя модели разными способами
                            if hasattr(model, "name"):
                                model_name_found = model.name
                                model_obj = model
                            elif hasattr(model, "get_name"):
                                model_name_found = model.get_name()
                                model_obj = model
                            elif isinstance(model, dict):
                                model_name_found = model.get("name")
                                model_obj = model

                            # Если имя совпадает, добавляем модель
                            if model_name_found == model_name and model_obj:
                                logger.debug(
                                    f"Найдена модель '{model_name}' в задаче {project_task.id}"
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

        # Если не нашли через задачи проекта, ищем в текущей задаче как fallback
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

                    if model_name_found == model_name:
                        logger.debug(f"Найдена модель '{model_name}' в текущей задаче")
                        all_models.append(model)
            except Exception as e:
                logger.debug(f"Не удалось найти модели в задаче: {e}")

        logger.info(f"Всего найдено моделей с именем '{model_name}': {len(all_models)}")

        version_count = len(all_models)

        if version_count == 0:
            # Если модели не найдены в ClearML, используем файловое хранилище
            logger.info("Модели не найдены в ClearML, проверяем файловое хранилище")
            file_version = _get_version_from_file(model_name)
            if file_version:
                logger.info(
                    f"Найдена версия в файле: latest_version={file_version['latest_version']}, next_version={file_version['next_version']}"
                )
                return file_version
            # Если и в файле нет, возвращаем версию 1
            logger.info("Версия не найдена ни в ClearML, ни в файле, используем версию 1")
            return {
                "version_count": 0,
                "latest_version": 1,
                "next_version": 1,
            }

        # Получаем версии из metadata и labels всех моделей
        versions = []
        for model in all_models:
            version_str = None

            # Сначала пробуем получить из metadata (правильный способ в ClearML)
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

            # Если не нашли в metadata, пробуем labels (fallback)
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
                    # Преобразуем версию в целое число
                    if isinstance(version_str, str) and "." in version_str:
                        version = int(float(version_str))
                    else:
                        version = int(version_str)
                    versions.append(version)
                    logger.debug(f"Найдена версия {version} для модели '{model_name}'")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Не удалось распарсить версию '{version_str}': {e}")

        # Если нашли версии в labels, используем максимальную
        if versions:
            latest_version = max(versions)
            logger.info(f"Найдены версии: {versions}, последняя: {latest_version}")
        else:
            # Если версий нет в labels, используем количество моделей
            latest_version = version_count
            logger.info(
                f"Версии в labels не найдены, используем количество моделей: {latest_version}"
            )

        # Следующая версия - это последняя версия + 1
        next_version = latest_version + 1
        logger.info(f"Следующая версия для модели '{model_name}': {next_version}")

        result = {
            "version_count": version_count,
            "latest_version": latest_version,
            "next_version": next_version,
        }

        # Сохраняем в файл как backup
        _save_version_to_file(model_name, latest_version)

        return result
    except Exception as e:
        logger.warning(f"Ошибка при получении информации о версиях модели: {e}")
        # В случае ошибки пробуем использовать файловое хранилище
        file_version = _get_version_from_file(model_name)
        if file_version:
            logger.info(f"Используем версию из файла: {file_version}")
            return file_version
        # В случае ошибки возвращаем следующую версию на основе счетчика
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

        # Подготавливаем labels с метаданными версии
        model_labels = labels.copy() if labels else {}
        if auto_version and version_info:
            model_labels["version"] = str(model_version)  # Сохраняем как строку для совместимости
            model_labels["model_version"] = str(model_version)  # Дублируем для удобства поиска
            model_labels["version_count"] = str(version_info["version_count"] + 1)
            model_labels["created_at"] = datetime.now().isoformat()
            if version_info["version_count"] > 0:
                model_labels["previous_version"] = str(version_info["latest_version"])

        if model_path:
            model_path = Path(model_path)
            if model_path.exists():
                # Добавляем версию в имя модели для лучшей видимости
                model_name_with_version = f"{model_name}_v{model_version}"
                output_model = OutputModel(
                    task=task, name=model_name_with_version, framework=framework
                )
                output_model.update_weights(str(model_path))

                # Добавляем теги с версией
                model_tags = list(tags) if tags else []
                model_tags.append(f"version_{model_version}")
                output_model.tags = model_tags

                # Сохраняем метаданные через set_metadata (правильный способ в ClearML)
                for key, value in model_labels.items():
                    try:
                        output_model.set_metadata(key, str(value))
                    except Exception as e:
                        logger.debug(f"Не удалось установить metadata {key}: {e}")
                        # Fallback: пробуем через labels
                        try:
                            if hasattr(output_model, "labels"):
                                output_model.labels[key] = str(value)
                        except Exception:  # nosec B110
                            pass

                logger.info(
                    f"Модель логирована в ClearML: {model_name_with_version}, версия: {model_version}"
                )
                logger.info(
                    f"Версия сохранена в metadata: version={model_version}, model_version={model_version}"
                )
                logger.info(f"Всего метаданных сохранено: {len(model_labels)}")
                # Сохраняем версию в файл как backup
                _save_version_to_file(model_name, model_version)
                return model_version
            else:
                logger.warning(f"Файл модели не найден: {model_path}")

        # Если путь не указан или файл не найден, сохраняем модель во временный файл
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                joblib.dump(model, tmp_path)
                # Добавляем версию в имя модели для лучшей видимости
                model_name_with_version = f"{model_name}_v{model_version}"
                output_model = OutputModel(
                    task=task, name=model_name_with_version, framework=framework
                )
                output_model.update_weights(tmp_path)

                # Добавляем теги с версией
                model_tags = list(tags) if tags else []
                model_tags.append(f"version_{model_version}")
                output_model.tags = model_tags

                # Сохраняем метаданные через set_metadata (правильный способ в ClearML)
                for key, value in model_labels.items():
                    try:
                        output_model.set_metadata(key, str(value))
                    except Exception as e:
                        logger.debug(f"Не удалось установить metadata {key}: {e}")
                        # Fallback: пробуем через labels
                        try:
                            if hasattr(output_model, "labels"):
                                output_model.labels[key] = str(value)
                        except Exception:  # nosec B110
                            pass

                logger.info(
                    f"Модель логирована в ClearML: {model_name_with_version}, версия: {model_version}"
                )
                logger.info(
                    f"Версия сохранена в metadata: version={model_version}, model_version={model_version}"
                )
                logger.info(f"Всего метаданных сохранено: {len(model_labels)}")
                # Сохраняем версию в файл как backup
                _save_version_to_file(model_name, model_version)
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
            # Получаем labels из модели
            if hasattr(model, "labels"):
                labels = model.labels
            elif isinstance(model, dict):
                labels = model.get("labels", {})
            else:
                labels = {}

            # Получаем версию и преобразуем в целое число
            version_str = labels.get("version", "1")
            try:
                # Если версия в формате "1.0", "2.0" и т.д., берем только целую часть
                if isinstance(version_str, str) and "." in version_str:
                    version = int(float(version_str))
                else:
                    version = int(version_str)
            except (ValueError, TypeError):
                version = 1

            version_data: dict[str, Any] = {
                "version": version,
                "version_str": str(version),  # Сохраняем строковое представление для совместимости
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
                best_version = version_data["version"]  # Используем целочисленную версию

            versions_data.append(version_data)

        # Сортируем версии по номеру версии (по убыванию)
        try:
            versions_data.sort(
                key=lambda x: x.get("version", 0),
                reverse=True,
            )
        except Exception:  # nosec B110
            # Если не удалось отсортировать по версии, сортируем по дате
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
