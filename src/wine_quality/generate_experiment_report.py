#!/usr/bin/env python3
"""Скрипт для автоматической генерации отчетов об экспериментах из ClearML."""

import argparse
import json
import logging
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from clearml import Task

    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    Task = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt: Any = None
    pd: Any = None
    sns: Any = None

try:
    import plotly.graph_objects as go
    from plotly.io import from_json, to_image

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    from_json = None
    to_image = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DOCUMENTATION_PREFIXES = [
    "clearml_",
    "report_",
    "compare",
    "cookie",
    "dc_",
    "docker",
    "dvc",
    "email",
    "experiments",
    "github",
    "hw1",
    "hydra",
    "login",
    "luigi",
    "model_",
    "poetry",
    "post-",
    "pre-",
    "run",
    "search",
]


def _format_datetime(dt: Any) -> str | None:
    """Форматирует datetime в ISO формат с timezone."""
    if not dt or not hasattr(dt, "strftime"):
        return str(dt) if dt else None

    if hasattr(dt, "tzinfo") and dt.tzinfo:
        tz_str = dt.strftime("%z")
        if len(tz_str) == 5:
            tz_str = f"{tz_str[:3]}:{tz_str[3:]}"
        result: str = dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + tz_str
        return result
    result: str = dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00"
    return result


def _flatten_params(d: dict, prefix: str = "") -> dict:
    """Рекурсивно разворачивает вложенные словари параметров."""
    result = {}
    for key, value in d.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_params(value, full_key))
        else:
            result[full_key] = value
    return result


def _extract_metrics(scalar_metrics: dict) -> dict:
    """Извлекает метрики из структуры ClearML."""
    metrics = {}
    for metric_name, metric_data in scalar_metrics.items():
        if not metric_data:
            continue

        if isinstance(metric_data, dict):
            for series_name, series_data in metric_data.items():
                if isinstance(series_data, dict):
                    value = series_data.get("last", series_data.get("value", 0))
                elif isinstance(series_data, (int, float)):
                    value = series_data
                else:
                    continue

                full_name = (
                    metric_name if series_name == "default" else f"{metric_name}/{series_name}"
                )
                metrics[full_name] = value
        elif isinstance(metric_data, (int, float)):
            metrics[metric_name] = metric_data
    return metrics


def get_task_data(task_id: str) -> dict[str, Any]:
    """Получает данные задачи из ClearML."""
    if not CLEARML_AVAILABLE:
        raise ImportError("ClearML не установлен. Установите: pip install clearml")

    try:
        task = Task.get_task(task_id=task_id)
        logger.info(f"Задача получена: {task.name}")

        params = {}
        try:
            task_params = task.get_parameters()
            if task_params:
                params.update(_flatten_params(task_params))
        except Exception as e:
            logger.warning(f"Не удалось получить параметры: {e}")

        metrics = {}
        try:
            scalar_metrics = task.get_last_scalar_metrics()
            if scalar_metrics:
                metrics = _extract_metrics(scalar_metrics)
        except Exception as e:
            logger.warning(f"Не удалось получить метрики: {e}")

        created_str = None
        try:
            if hasattr(task, "data") and task.data and hasattr(task.data, "created"):
                created_str = _format_datetime(task.data.created)
        except Exception as e:
            logger.debug(f"Не удалось получить дату создания: {e}")

        completed_str = None
        try:
            if hasattr(task, "data") and task.data and hasattr(task.data, "completed"):
                completed_str = _format_datetime(task.data.completed)
        except Exception as e:
            logger.debug(f"Не удалось получить дату завершения: {e}")

        task_info = {
            "id": task.id,
            "name": task.name,
            "project": task.get_project_name() if hasattr(task, "get_project_name") else "Unknown",
            "status": task.status if hasattr(task, "status") else "unknown",
            "created": created_str,
            "completed": completed_str,
            "tags": task.get_tags() if hasattr(task, "get_tags") else [],
        }

        plots = []
        try:
            reported_plots = task.get_reported_plots()
            if reported_plots:
                plots = reported_plots
        except Exception as e:
            logger.warning(f"Не удалось получить графики: {e}")

        return {
            "task_info": task_info,
            "params": params,
            "metrics": metrics,
            "plots": plots,
        }
    except Exception as e:
        logger.error(f"Ошибка при получении данных задачи {task_id}: {e}")
        raise


def get_comparison_data(
    task_ids: list[str],
    project_name: str | None = None,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Получает данные нескольких задач для сравнения."""
    comparison_data = []
    for task_id in task_ids:
        try:
            data = get_task_data(task_id)
            comparison_data.append(data)
        except Exception as e:
            logger.warning(f"Не удалось получить данные для задачи {task_id}: {e}")
            continue
    return comparison_data


def format_value(value: Any) -> str:
    """Форматирует значение для отображения в таблице."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def generate_metrics_table(metrics: dict[str, float]) -> str:
    """Генерирует таблицу метрик в markdown формате."""
    if not metrics:
        return "| Метрика | Значение |\n|---------|----------|\n"

    lines = ["| Метрика | Значение |", "|---------|----------|"]
    for metric_name, metric_value in sorted(metrics.items()):
        lines.append(f"| `{metric_name}` | `{format_value(metric_value)}` |")
    return "\n".join(lines)


def generate_params_table(params: dict[str, Any]) -> str:
    """Генерирует таблицу параметров в markdown формате."""
    if not params:
        return "| Параметр | Значение |\n|----------|----------|\n"

    lines = ["| Параметр | Значение |", "|----------|----------|"]
    for param_name, param_value in sorted(params.items()):
        lines.append(f"| `{param_name}` | `{format_value(param_value)}` |")
    return "\n".join(lines)


def generate_comparison_table(comparison_data: list[dict[str, Any]]) -> str:
    """Генерирует таблицу сравнения экспериментов."""
    if not comparison_data:
        return ""

    all_metrics = set()
    for data in comparison_data:
        all_metrics.update(data.get("metrics", {}).keys())

    if not all_metrics:
        return ""

    header = ["Эксперимент", *sorted(all_metrics)]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for data in comparison_data:
        task_name = data.get("task_info", {}).get("name", "Unknown")
        if len(task_name) > 50:
            task_name = task_name[:47] + "..."
        row = [task_name] + [
            format_value(data.get("metrics", {}).get(metric, "")) for metric in sorted(all_metrics)
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _fix_plotly_deprecated_props(plot_data: dict) -> None:
    """Исправляет устаревшие свойства Plotly."""
    if "layout" not in plot_data:
        return

    layout = plot_data["layout"]

    def fix_titlefont(obj: dict, key: str) -> None:
        if key in obj:
            if "title" not in obj:
                obj["title"] = {}
            if isinstance(obj["title"], dict):
                obj["title"]["font"] = obj[key]
            del obj[key]

    fix_titlefont(layout, "titlefont")

    for axis_key in ["xaxis", "yaxis"]:
        if axis_key in layout and isinstance(layout[axis_key], dict):
            fix_titlefont(layout[axis_key], "titlefont")


def convert_plotly_to_image(  # noqa: PLR0911
    plot_str: str, output_path: Path, width: int = 1200, height: int = 800
) -> bool:
    """Конвертирует Plotly график из JSON строки в изображение."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly не доступен, пропускаем конвертацию")
        return False

    try:
        plot_data = json.loads(plot_str)
        _fix_plotly_deprecated_props(plot_data)

        if "data" in plot_data and "layout" in plot_data:
            try:
                fig = go.Figure(data=plot_data["data"], layout=plot_data["layout"])
            except Exception:
                try:
                    fig = from_json(plot_str)
                except Exception:
                    logger.warning("Не удалось создать figure из plot_str")
                    return False
        else:
            try:
                fig = from_json(plot_str)
            except Exception:
                logger.warning("Не удалось создать figure из plot_str")
                return False

        if hasattr(fig, "update_layout"):
            try:
                fig.update_layout(width=width, height=height)
            except Exception:
                logger.debug("Не удалось обновить layout графика")

        try:
            img_bytes = to_image(fig, format="png", width=width, height=height)
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            logger.info(f"Plotly график конвертирован и сохранен: {output_path}")
            return True
        except Exception as e:
            logger.warning(f"Не удалось конвертировать Plotly график в изображение: {e}")
            return False

    except json.JSONDecodeError as e:
        logger.warning(f"Не удалось распарсить JSON из plot_str: {e}")
        return False
    except Exception as e:
        logger.warning(f"Ошибка при конвертации Plotly графика: {e}")
        return False


def _get_session(task: Task) -> Any:
    """Получает сессию из задачи для авторизованных запросов."""
    if hasattr(task, "_session"):
        return task._session
    if hasattr(task, "session"):
        return task.session
    return None


def _download_image_with_auth(url: str, output_path: Path, session: Any) -> bool:
    """Скачивает изображение с авторизацией."""
    try:
        req = urllib.request.Request(url)

        token = None
        try:
            if hasattr(session, "_get_token"):
                token = session._get_token()
            elif hasattr(session, "get_token"):
                token = session.get_token()
        except Exception:
            logger.debug("Не удалось получить токен из сессии")

        if token:
            req.add_header("Authorization", f"Bearer {token}")
        elif hasattr(session, "auth_headers"):
            for key, value in session.auth_headers.items():
                req.add_header(key, value)
        elif hasattr(session, "headers"):
            for key, value in session.headers.items():
                req.add_header(key, value)

        req.add_header("User-Agent", "ClearML-Python-Client")

        with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310
            with open(output_path, "wb") as f:
                f.write(response.read())
        return True
    except Exception as e:
        logger.debug(f"Не удалось скачать через urllib: {e}")
        return False


def _sanitize_filename(name: str, prefix: str = "") -> str:
    """Очищает имя файла от недопустимых символов."""
    safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)
    safe_name = safe_name.replace(" ", "_")[:50]
    return f"{prefix}_{safe_name}" if prefix else safe_name


def download_plots_from_clearml(
    task: Task, output_dir: Path, max_plots: int = 10, prefix: str = ""
) -> list[str]:
    """Скачивает графики из ClearML задачи."""
    downloaded_plots: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        reported_plots = task.get_reported_plots()
        if not reported_plots:
            logger.info("Графики не найдены в задаче")
            return downloaded_plots

        logger.info(f"Найдено {len(reported_plots)} графиков в задаче")
        all_plots = reported_plots[:max_plots]
        session = _get_session(task)

        for idx, plot in enumerate(all_plots):
            try:
                plot_name = plot.get("metric", f"plot_{idx}")
                safe_name = _sanitize_filename(plot_name, prefix)
                downloaded = False

                if plot.get("source_urls"):
                    for url in plot["source_urls"]:
                        parsed_url = urlparse(url)
                        ext = Path(parsed_url.path).suffix or ".png"
                        filename = f"{safe_name}{ext}"
                        output_path = output_dir / filename

                        logger.info(f"Скачивание графика: {url} -> {output_path}")

                        if session and _download_image_with_auth(url, output_path, session):
                            downloaded_plots.append(filename)
                            logger.info(f"График сохранен: {output_path}")
                            downloaded = True
                            break

                if not downloaded and "plot_str" in plot and plot["plot_str"]:
                    filename = f"{safe_name}.png"
                    output_path = output_dir / filename
                    logger.info(f"Конвертация Plotly графика: {plot_name} -> {output_path}")

                    if convert_plotly_to_image(plot["plot_str"], output_path):
                        downloaded_plots.append(filename)
                        logger.info(f"Plotly график конвертирован и сохранен: {output_path}")
                        downloaded = True

            except Exception as e:
                logger.warning(f"Ошибка при обработке графика {idx}: {e}")
                continue

        logger.info(f"Скачано {len(downloaded_plots)} графиков")
        return downloaded_plots

    except Exception as e:
        logger.warning(f"Ошибка при скачивании графиков: {e}")
        return downloaded_plots


def _cleanup_old_report_plots(output_dir: Path, current_plots: list[str]) -> None:
    """Удаляет старые файлы графиков отчета, оставляя только актуальные."""
    try:
        all_png_files = list(output_dir.glob("*.png"))
        current_plots_set = set(current_plots)

        files_to_delete = []
        for png_file in all_png_files:
            filename = png_file.name

            if any(filename.startswith(prefix) for prefix in DOCUMENTATION_PREFIXES):
                continue

            if filename in current_plots_set:
                continue

            files_to_delete.append(png_file)

        if files_to_delete:
            logger.info(f"Удаление {len(files_to_delete)} старых файлов графиков отчета")
            for file_to_delete in files_to_delete:
                try:
                    file_to_delete.unlink()
                    logger.debug(f"Удален старый файл графика: {file_to_delete.name}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить файл {file_to_delete.name}: {e}")
    except Exception as e:
        logger.warning(f"Ошибка при очистке старых файлов графиков: {e}")


def generate_comparison_plot(comparison_data: list[dict[str, Any]], output_path: Path) -> None:
    """Генерирует график сравнения метрик."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib не доступен, пропускаем генерацию графика")
        return

    if not comparison_data:
        return

    plot_data = []
    for data in comparison_data:
        task_name = data.get("task_info", {}).get("name", "Unknown")
        metrics = data.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                plot_data.append(
                    {
                        "Experiment": task_name[:30],
                        "Metric": metric_name,
                        "Value": float(metric_value),
                    }
                )

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 6))
    unique_metrics = df["Metric"].unique()
    n_metrics = len(unique_metrics)

    for i, metric in enumerate(unique_metrics):
        plt.subplot(1, n_metrics, i + 1)
        metric_data = df[df["Metric"] == metric]
        plt.bar(range(len(metric_data)), metric_data["Value"])
        plt.xticks(range(len(metric_data)), metric_data["Experiment"], rotation=45, ha="right")
        plt.ylabel("Value")
        plt.title(metric)
        plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"График сравнения сохранен: {output_path}")


def generate_report(  # noqa: PLR0912, PLR0915
    task_data: dict[str, Any],
    comparison_data: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
) -> str:
    """Генерирует markdown отчет об эксперименте."""
    task_info = task_data.get("task_info", {})
    params = task_data.get("params", {})
    metrics = task_data.get("metrics", {})

    report_lines = [
        f"# Отчет об эксперименте: {task_info.get('name', 'Unknown')}",
        "",
        f"**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Основная информация",
        "",
        f"- **ID задачи:** `{task_info.get('id', 'Unknown')}`",
        f"- **Проект:** {task_info.get('project', 'Unknown')}",
        f"- **Статус:** {task_info.get('status', 'unknown')}",
    ]

    if task_info.get("created"):
        report_lines.append(f"- **Создано:** {task_info.get('created')}")
    if task_info.get("completed"):
        report_lines.append(f"- **Завершено:** {task_info.get('completed')}")

    tags = task_info.get("tags", [])
    if tags:
        report_lines.append(f"- **Теги:** {', '.join(tags)}")

    report_lines.extend(
        [
            "",
            "## Параметры эксперимента",
            "",
            generate_params_table(params),
            "",
            "## Метрики",
            "",
            "",
            generate_metrics_table(metrics),
        ]
    )

    if comparison_data:
        report_lines.extend(
            [
                "",
                "## Сравнение экспериментов",
                "",
                "### Сравнительная таблица метрик",
                "",
                generate_comparison_table(comparison_data),
            ]
        )

        if output_dir and MATPLOTLIB_AVAILABLE:
            plot_path = output_dir / "metrics_comparison.png"
            generate_comparison_plot(comparison_data, plot_path)
            report_lines.extend(
                [
                    "",
                    "### График сравнения метрик",
                    "",
                    "![Сравнение метрик](metrics_comparison.png)",
                ]
            )

    report_lines.extend(
        [
            "",
            "## Визуализации",
            "",
        ]
    )

    downloaded_plots = task_data.get("downloaded_plots", [])
    plots_info = task_data.get("plots", [])

    all_downloaded_plots = list(downloaded_plots) if downloaded_plots else []

    if comparison_data:
        for comp_data in comparison_data:
            comp_plots = comp_data.get("downloaded_plots", [])
            if comp_plots:
                all_downloaded_plots.extend(comp_plots)

    if all_downloaded_plots and output_dir:
        relative_pics_dir = "pics" if output_dir.name != "pics" else "."

        if comparison_data:
            if downloaded_plots:
                report_lines.append("### Графики основного эксперимента")
                report_lines.append("")
                for plot_file in downloaded_plots:
                    report_lines.append(f">![метрики](./{relative_pics_dir}/{plot_file})")

            for comp_data in comparison_data:
                comp_plots = comp_data.get("downloaded_plots", [])
                if comp_plots:
                    comp_name = comp_data.get("task_info", {}).get("name", "Unknown")
                    if len(comp_name) > 50:
                        comp_name = comp_name[:47] + "..."
                    report_lines.append("")
                    report_lines.append(f"### Графики эксперимента: {comp_name}")
                    report_lines.append("")
                    for plot_file in comp_plots:
                        report_lines.append(f">![метрики](./{relative_pics_dir}/{plot_file})")
        else:
            for plot_file in all_downloaded_plots:
                report_lines.append(f">![метрики](./{relative_pics_dir}/{plot_file})")
    elif plots_info:
        report_lines.append(
            ">⚠️ Графики найдены в эксперименте, но не удалось их скачать автоматически."
        )
        report_lines.append(">Для скачивания графиков используйте веб-интерфейс ClearML или API.")
        report_lines.append("")
        report_lines.append(f">Найдено графиков: {len(plots_info)}")
        for idx, plot in enumerate(plots_info[:5]):
            plot_name = plot.get("metric", f"График {idx + 1}")
            report_lines.append(f">- {plot_name}")
    else:
        report_lines.append(">Графики не найдены в эксперименте")

    return "\n".join(report_lines)


def main() -> None:  # noqa: PLR0912, PLR0915
    """Главная функция скрипта."""
    parser = argparse.ArgumentParser(
        description="Генерация отчетов об экспериментах из ClearML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Генерация отчета по одному эксперименту:
  python scripts/generate_experiment_report.py --task-id aeb5b1e9b2374ced9bfdee23e3bc09d1 --output docs/experiment_report.md

  # Сравнение нескольких экспериментов:
  python scripts/generate_experiment_report.py \\
      --comparison aeb5b1e9b2374ced9bfdee23e3bc09d1 task_id_2 task_id_3 \\
      --output docs/comparison_report.md

  # Отчет с сравнением:
  python scripts/generate_experiment_report.py \\
      --task-id aeb5b1e9b2374ced9bfdee23e3bc09d1 \\
      --comparison task_id_2 task_id_3 \\
      --output docs/experiment_report.md
        """,
    )
    parser.add_argument("--task-id", type=str, help="ID задачи ClearML для генерации отчета")
    parser.add_argument(
        "--comparison", nargs="+", type=str, help="ID задач для сравнения (можно указать несколько)"
    )
    parser.add_argument(
        "--output", type=str, default="experiment_report.md", help="Путь для сохранения отчета"
    )
    parser.add_argument("--project", type=str, help="Имя проекта для фильтрации задач")

    args = parser.parse_args()

    if not args.task_id and not args.comparison:
        parser.error("Необходимо указать --task-id или --comparison")

    try:
        if args.task_id:
            task_data = get_task_data(args.task_id)
        elif args.comparison:
            task_data = get_task_data(args.comparison[0])

        comparison_data = None
        if args.comparison:
            comparison_task_ids = args.comparison.copy()
            if args.task_id and args.task_id not in comparison_task_ids:
                comparison_task_ids.append(args.task_id)
            comparison_data = get_comparison_data(comparison_task_ids, args.project)

        output_path = Path(args.output)
        output_dir = output_path.parent

        if not task_data:
            logger.error("Не удалось получить данные задачи")
            sys.exit(1)

        pics_dir = output_dir / "pics"

        if CLEARML_AVAILABLE and task_data.get("task_info", {}).get("id"):
            try:
                task = Task.get_task(task_id=task_data["task_info"]["id"])
                downloaded_plots = download_plots_from_clearml(task, pics_dir)
                task_data["downloaded_plots"] = downloaded_plots
            except Exception as e:
                logger.warning(f"Не удалось скачать графики основного эксперимента: {e}")
                task_data["downloaded_plots"] = []

        all_downloaded_plots = list(task_data.get("downloaded_plots", []))
        main_task_id = task_data.get("task_info", {}).get("id")

        if comparison_data and CLEARML_AVAILABLE:
            for comp_data in comparison_data:
                comp_task_id = comp_data.get("task_info", {}).get("id")
                if comp_task_id and comp_task_id != main_task_id:
                    try:
                        comp_task = Task.get_task(task_id=comp_task_id)
                        comp_name = comp_data.get("task_info", {}).get("name", "Unknown")
                        safe_prefix = "".join(
                            c if c.isalnum() or c in (" ", "-", "_") else "_" for c in comp_name
                        )
                        safe_prefix = safe_prefix.replace(" ", "_")[:30]

                        logger.info(f"Скачивание графиков для сравнения: {comp_name}")
                        comp_plots = download_plots_from_clearml(
                            comp_task, pics_dir, prefix=safe_prefix
                        )
                        comp_data["downloaded_plots"] = comp_plots
                        all_downloaded_plots.extend(comp_plots)
                    except Exception as e:
                        logger.warning(
                            f"Не удалось скачать графики для эксперимента {comp_task_id}: {e}"
                        )
                        comp_data["downloaded_plots"] = []

        _cleanup_old_report_plots(pics_dir, all_downloaded_plots)

        report = generate_report(task_data, comparison_data, output_dir)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info(f"Отчет сохранен: {output_path}")
        logger.info(f"Размер отчета: {len(report)} символов")

    except Exception as e:
        logger.error(f"Ошибка при генерации отчета: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
