"""
Утилиты для работы с экспериментами MLflow.

Этот модуль предоставляет функции для поиска, фильтрации и сравнения экспериментов.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient


def search_experiments(
    filter_string: str | None = None,
    max_results: int = 1000,
) -> list[Experiment]:
    """
    Поиск экспериментов по фильтру.

    Args:
        filter_string: Строка фильтрации (например, "name = 'my_experiment'")
        max_results: Максимальное количество результатов

    Returns:
        Список экспериментов

    Пример использования:
        experiments = search_experiments(filter_string="name LIKE '%wine%'")
    """
    client = MlflowClient()
    try:
        experiments = client.search_experiments(
            filter_string=filter_string, max_results=max_results
        )
        return list(experiments)
    except Exception as e:
        print(f"Ошибка при поиске экспериментов: {e}")
        return []


def search_runs(
    experiment_ids: list[str] | None = None,
    filter_string: str | None = None,
    run_view_type: int = 1,  # ACTIVE_ONLY
    max_results: int = 1000,
    order_by: list[str] | None = None,
) -> list[Run]:
    """
    Поиск runs по фильтру.

    Args:
        experiment_ids: Список ID экспериментов для поиска
        filter_string: Строка фильтрации (например, "metrics.accuracy > 0.9")
        run_view_type: Тип представления (1 = ACTIVE_ONLY, 2 = DELETED_ONLY, 3 = ALL)
        max_results: Максимальное количество результатов
        order_by: Список полей для сортировки (например, ["metrics.accuracy DESC"])

    Returns:
        Список runs

    Пример использования:
        runs = search_runs(
            experiment_ids=["123"],
            filter_string="metrics.accuracy > 0.9",
            order_by=["metrics.accuracy DESC"]
        )
    """
    client = MlflowClient()
    try:
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            run_view_type=run_view_type,
            max_results=max_results,
            order_by=order_by,
        )
        return list(runs)
    except Exception as e:
        print(f"Ошибка при поиске runs: {e}")
        return []


def filter_runs_by_metrics(
    runs: list[Run],
    metric_filters: dict[str, tuple[float, float] | float],
) -> list[Run]:
    """
    Фильтрует runs по метрикам.

    Args:
        runs: Список runs для фильтрации
        metric_filters: Словарь с фильтрами метрик.
                       Ключ - имя метрики, значение - либо одно число (>=),
                       либо кортеж (min, max) для диапазона

    Returns:
        Отфильтрованный список runs

    Пример использования:
        filtered = filter_runs_by_metrics(
            runs,
            {
                "accuracy": (0.8, 1.0),  # 0.8 <= accuracy <= 1.0
                "f1_score": 0.9,  # f1_score >= 0.9
            }
        )
    """
    filtered_runs = []
    for run in runs:
        match = True
        for metric_name, filter_value in metric_filters.items():
            if metric_name not in run.data.metrics:
                match = False
                break

            metric_value = run.data.metrics[metric_name]

            if isinstance(filter_value, tuple):
                min_val, max_val = filter_value
                if not (min_val <= metric_value <= max_val):
                    match = False
                    break
            elif metric_value < filter_value:
                match = False
                break

        if match:
            filtered_runs.append(run)

    return filtered_runs


def filter_runs_by_params(
    runs: list[Run],
    param_filters: dict[str, str | list[str]],
) -> list[Run]:
    """
    Фильтрует runs по параметрам.

    Args:
        runs: Список runs для фильтрации
        param_filters: Словарь с фильтрами параметров.
                      Ключ - имя параметра, значение - либо строка (точное совпадение),
                      либо список строк (любое из значений)

    Returns:
        Отфильтрованный список runs

    Пример использования:
        filtered = filter_runs_by_params(
            runs,
            {
                "model_type": "random_forest",
                "n_estimators": ["100", "200"],  # n_estimators = 100 или 200
            }
        )
    """
    filtered_runs = []
    for run in runs:
        match = True
        for param_name, filter_value in param_filters.items():
            if param_name not in run.data.params:
                match = False
                break

            param_value = run.data.params[param_name]

            if isinstance(filter_value, list):
                if param_value not in filter_value:
                    match = False
                    break
            elif param_value != filter_value:
                match = False
                break

        if match:
            filtered_runs.append(run)

    return filtered_runs


def compare_runs(
    runs: list[Run],
    metric_names: list[str] | None = None,
    param_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Сравнивает runs и возвращает DataFrame с метриками и параметрами.

    Args:
        runs: Список runs для сравнения
        metric_names: Список имен метрик для включения (если None, включаются все)
        param_names: Список имен параметров для включения (если None, включаются все)

    Returns:
        DataFrame с колонками: run_id, experiment_id, status, start_time,
        и всеми метриками и параметрами

    Пример использования:
        df = compare_runs(
            runs,
            metric_names=["accuracy", "f1_score"],
            param_names=["model_type", "n_estimators"]
        )
    """
    data = []
    for run in runs:
        row: dict[str, Any] = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
            "end_time": (
                datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None
            ),
        }

        # Добавляем метрики
        if metric_names:
            for metric_name in metric_names:
                row[f"metric_{metric_name}"] = run.data.metrics.get(metric_name)
        else:
            for metric_name, metric_value in run.data.metrics.items():
                row[f"metric_{metric_name}"] = metric_value

        # Добавляем параметры
        if param_names:
            for param_name in param_names:
                row[f"param_{param_name}"] = run.data.params.get(param_name)
        else:
            for param_name, param_value in run.data.params.items():
                row[f"param_{param_name}"] = param_value

        # Добавляем теги
        for tag_name, tag_value in run.data.tags.items():
            row[f"tag_{tag_name}"] = tag_value

        data.append(row)

    return pd.DataFrame(data)


def get_best_runs(
    runs: list[Run],
    metric_name: str,
    ascending: bool = False,
    top_k: int = 5,
) -> list[Run]:
    """
    Возвращает лучшие runs по указанной метрике.

    Args:
        runs: Список runs
        metric_name: Имя метрики для сортировки
        ascending: Сортировать по возрастанию (False = по убыванию)
        top_k: Количество лучших runs для возврата

    Returns:
        Список лучших runs

    Пример использования:
        best_runs = get_best_runs(runs, "accuracy", ascending=False, top_k=5)
    """
    # Фильтруем runs, у которых есть указанная метрика
    runs_with_metric = [run for run in runs if metric_name in run.data.metrics]

    # Сортируем по метрике
    sorted_runs = sorted(
        runs_with_metric,
        key=lambda r: r.data.metrics[metric_name],
        reverse=not ascending,
    )

    return sorted_runs[:top_k]


def get_experiment_summary(experiment_id: str) -> dict[str, Any]:
    """
    Получает сводку по эксперименту.

    Args:
        experiment_id: ID эксперимента

    Returns:
        Словарь со сводкой: количество runs, лучшие метрики, средние метрики и т.д.

    Пример использования:
        summary = get_experiment_summary("123")
    """
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id], max_results=1000)

    if not runs:
        return {
            "experiment_id": experiment_id,
            "total_runs": 0,
            "active_runs": 0,
            "finished_runs": 0,
            "failed_runs": 0,
        }

    # Подсчитываем статистику
    total_runs = len(runs)
    active_runs = sum(1 for r in runs if r.info.status == "RUNNING")
    finished_runs = sum(1 for r in runs if r.info.status == "FINISHED")
    failed_runs = sum(1 for r in runs if r.info.status == "FAILED")

    # Собираем все метрики
    all_metrics: dict[str, list[float]] = {}
    for run in runs:
        if run.info.status == "FINISHED":
            for metric_name, metric_value in run.data.metrics.items():
                if isinstance(metric_value, (int, float)):
                    all_metrics.setdefault(metric_name, []).append(float(metric_value))

    # Вычисляем статистику по метрикам
    metrics_summary: dict[str, dict[str, float]] = {}
    for metric_name, values in all_metrics.items():
        if values:
            metrics_summary[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std": (
                    (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5
                    if len(values) > 1
                    else 0.0
                ),
            }

    return {
        "experiment_id": experiment_id,
        "total_runs": total_runs,
        "active_runs": active_runs,
        "finished_runs": finished_runs,
        "failed_runs": failed_runs,
        "metrics_summary": metrics_summary,
    }


def export_runs_to_csv(
    runs: list[Run],
    output_path: str,
    metric_names: list[str] | None = None,
    param_names: list[str] | None = None,
) -> None:
    """
    Экспортирует runs в CSV файл.

    Args:
        runs: Список runs для экспорта
        output_path: Путь к выходному CSV файлу
        metric_names: Список имен метрик для включения
        param_names: Список имен параметров для включения

    Пример использования:
        export_runs_to_csv(runs, "experiments.csv")
    """
    df = compare_runs(runs, metric_names=metric_names, param_names=param_names)
    df.to_csv(output_path, index=False)
    print(f"Runs экспортированы в {output_path}")


def delete_runs(
    run_ids: list[str],
    experiment_id: str | None = None,
) -> None:
    """
    Удаляет указанные runs.

    Args:
        run_ids: Список ID runs для удаления
        experiment_id: ID эксперимента (опционально, для проверки)

    Пример использования:
        delete_runs(["run_id_1", "run_id_2"])
    """
    client = MlflowClient()
    for run_id in run_ids:
        try:
            if experiment_id:
                # Проверяем, что run принадлежит указанному эксперименту
                run = client.get_run(run_id)
                if run.info.experiment_id != experiment_id:
                    print(
                        f"Предупреждение: run {run_id} не принадлежит эксперименту {experiment_id}"
                    )
                    continue

            client.delete_run(run_id)
            print(f"Run {run_id} удалён")
        except Exception as e:
            print(f"Ошибка при удалении run {run_id}: {e}")


def restore_runs(run_ids: list[str]) -> None:
    """
    Восстанавливает удалённые runs.

    Args:
        run_ids: Список ID runs для восстановления

    Пример использования:
        restore_runs(["run_id_1", "run_id_2"])
    """
    client = MlflowClient()
    for run_id in run_ids:
        try:
            client.restore_run(run_id)
            print(f"Run {run_id} восстановлен")
        except Exception as e:
            print(f"Ошибка при восстановлении run {run_id}: {e}")
