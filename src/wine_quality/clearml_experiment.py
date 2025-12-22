import logging
import sys
from pathlib import Path
from typing import Any

from clearml import Task
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None


def run_simple_experiment(  # noqa: PLR0912, PLR0915
    quality_threshold: int = 7,
    test_size: float = 0.2,
    random_state: int = 42,
    model_type: str = "boosting",
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    boosting_n_estimators: int | None = None,
    boosting_max_depth: int | None = None,
    boosting_learning_rate: float | None = None,
    mlp_hidden_layer_sizes: tuple[int, ...] | None = None,
    mlp_max_iter: int | None = None,
    experiment_name: str | None = None,
    project_name: str = "Wine Quality",
) -> dict[str, Any]:
    project_root: Path | None = None
    try:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
    except Exception:
        project_root = Path.cwd()

    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from wine_quality.clearml_model_registry import register_model_with_version  # noqa: PLC0415
    from wine_quality.config import BASE_DIR, PLOTS_DIR  # noqa: PLC0415
    from wine_quality.data import (  # noqa: PLC0415
        create_target,
        get_features_and_target,
        load_data,
        split_data,
    )
    from wine_quality.features import engineer_features  # noqa: PLC0415
    from wine_quality.model import ModelType, run_experiment  # noqa: PLC0415

    base_task_name = experiment_name or f"Wine Quality - {model_type.title()}"
    task = Task.init(
        project_name=project_name,
        task_name=base_task_name,
        task_type="training",
    )

    tags = [
        "framework:sklearn",
        "task:classification",
        f"model_type:{model_type}",
        "wine_quality",
    ]
    try:
        task.add_tags(tags)
        logger.info(f"Теги установлены: {tags}")
    except Exception as e:
        logger.warning(f"Не удалось установить теги: {e}")

    full_task_name = None
    try:
        model_name_display = model_type.replace("_", " ").title()
        task_name_parts = [f"Wine Quality - {model_name_display}"]

        params_list = []
        if model_type == "boosting":
            if boosting_n_estimators:
                params_list.append(f"n_est={boosting_n_estimators}")
            if boosting_max_depth:
                params_list.append(f"depth={boosting_max_depth}")
            if boosting_learning_rate:
                params_list.append(f"lr={boosting_learning_rate:.3f}")
        elif model_type == "random_forest":
            if rf_n_estimators:
                params_list.append(f"n_est={rf_n_estimators}")
            if rf_max_depth:
                params_list.append(f"depth={rf_max_depth}")
        elif model_type == "mlp":
            if mlp_hidden_layer_sizes:
                hidden_str = "x".join(str(s) for s in mlp_hidden_layer_sizes)
                params_list.append(f"hidden={hidden_str}")
            if mlp_max_iter:
                params_list.append(f"max_iter={mlp_max_iter}")

        if params_list:
            params_str = ", ".join(params_list)
            task_name_parts.append(f"[{params_str}]")

        if experiment_name:
            task_name_parts.append(f"({experiment_name})")

        full_task_name = " | ".join(task_name_parts)
        if len(full_task_name) > 200:
            full_task_name = full_task_name[:197] + "..."

        task.set_name(full_task_name)
        logger.info(f"Имя задачи установлено: {full_task_name}")
    except Exception as e:
        logger.warning(f"Не удалось изменить имя задачи: {e}")

    try:
        logger.info("=" * 80)
        logger.info("НАЧАЛО ЭКСПЕРИМЕНТА")
        logger.info("=" * 80)

        logger.info("Шаг 1: Загрузка и подготовка данных...")
        df = load_data()
        df = create_target(df, threshold=quality_threshold)
        df = engineer_features(df)

        X, y = get_features_and_target(df)
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"Данные загружены: train={len(X_train)}, test={len(X_test)}")

        task.connect(
            {
                "quality_threshold": quality_threshold,
                "test_size": test_size,
                "random_state": random_state,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": len(X_train.columns),
            }
        )

        logger.info("Шаг 2: Обучение модели...")
        model_type_enum = ModelType(model_type.lower())

        result = run_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type=model_type_enum,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            boosting_n_estimators=boosting_n_estimators,
            boosting_max_depth=boosting_max_depth,
            boosting_learning_rate=boosting_learning_rate,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
            mlp_max_iter=mlp_max_iter,
            random_state=random_state,
            use_mlflow=False,
            use_clearml=True,
            experiment_name=None,
            clearml_project_name=project_name,
            model_path=str(BASE_DIR / "models" / f"wine_{model_type}.pkl"),
            save_local=True,
            log_artifacts=True,
            df_for_plots=df,
            register_model_name=None,
        )

        if full_task_name:
            try:
                task.set_name(full_task_name)
                logger.info(f"Финальное имя задачи установлено: {full_task_name}")
            except Exception as e:
                logger.warning(f"Не удалось установить финальное имя задачи: {e}")

        logger.info("Шаг 3: Регистрация модели...")
        try:
            model_file_path = str(BASE_DIR / "models" / f"wine_{model_type}.pkl")
            model_name = f"wine_quality_{model_type}"

            if Path(model_file_path).exists() and register_model_with_version:
                metrics = result["metrics"]
                version_info = register_model_with_version(
                    model=result["model"],
                    model_name=model_name,
                    model_path=model_file_path,
                    framework="scikit-learn",
                    tags=[model_type, "wine_quality", "simple_experiment"],
                    labels={
                        "model_type": model_type,
                        "accuracy": str(metrics.get("accuracy", 0)),
                        "f1_weighted": str(metrics.get("f1_weighted", 0)),
                        "experiment_name": experiment_name or "default",
                    },
                    auto_version=True,
                )
                version = version_info.get("version", "unknown")
                logger.info(
                    f"Модель зарегистрирована с версионированием: {model_name}, версия: {version}"
                )
        except Exception as e:
            logger.warning(f"Не удалось зарегистрировать модель: {e}")

        metrics = result["metrics"]
        task.logger.report_scalar(
            title="Final Metrics",
            series="accuracy",
            value=float(metrics.get("accuracy", 0)),
            iteration=0,
        )
        task.logger.report_scalar(
            title="Final Metrics",
            series="f1_weighted",
            value=float(metrics.get("f1_weighted", 0)),
            iteration=0,
        )

        params_dict = {
            "model_type": model_type,
            "random_state": random_state,
            "quality_threshold": quality_threshold,
            "test_size": test_size,
        }
        if model_type == "boosting":
            params_dict.update(
                {
                    "boosting_n_estimators": boosting_n_estimators or "default",
                    "boosting_max_depth": boosting_max_depth or "default",
                    "boosting_learning_rate": boosting_learning_rate or "default",
                }
            )
        elif model_type == "random_forest":
            params_dict.update(
                {
                    "rf_n_estimators": rf_n_estimators or "default",
                    "rf_max_depth": rf_max_depth or "default",
                }
            )
        elif model_type == "mlp":
            params_dict.update(
                {
                    "mlp_hidden_layer_sizes": str(mlp_hidden_layer_sizes)
                    if mlp_hidden_layer_sizes
                    else "default",
                    "mlp_max_iter": mlp_max_iter or "default",
                }
            )

        task.connect(params_dict)

        model_file_path = str(BASE_DIR / "models" / f"wine_{model_type}.pkl")
        if Path(model_file_path).exists():
            try:
                task.upload_artifact(
                    name="model.pkl",
                    artifact_object=model_file_path,
                )
                logger.info(f"Модель залогирована как артефакт: {model_file_path}")
            except Exception as e:
                logger.warning(f"Не удалось залогировать модель: {e}")

        if MATPLOTLIB_AVAILABLE:
            try:
                PLOTS_DIR.mkdir(parents=True, exist_ok=True)

                feature_plot_path = PLOTS_DIR / "feature_importances.png"
                if feature_plot_path.exists() and feature_plot_path.stat().st_size > 0:
                    task.logger.report_image(
                        title="Plots",
                        series="Feature Importances",
                        local_path=str(feature_plot_path),
                        iteration=0,
                    )
                    logger.info(f"График важности признаков залогирован: {feature_plot_path}")

                corr_plot_path = PLOTS_DIR / "correlation_heatmap.png"
                if corr_plot_path.exists() and corr_plot_path.stat().st_size > 0:
                    task.logger.report_image(
                        title="Plots",
                        series="Correlation Heatmap",
                        local_path=str(corr_plot_path),
                        iteration=0,
                    )
                    logger.info(f"График корреляций залогирован: {corr_plot_path}")
            except Exception as e:
                logger.warning(f"Не удалось залогировать графики: {e}")

        logger.info("=" * 80)
        logger.info("ЭКСПЕРИМЕНТ ЗАВЕРШЕН УСПЕШНО")
        logger.info("=" * 80)
        logger.info(f"Модель: {model_type}")
        logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"F1-score: {metrics.get('f1_weighted', 0):.4f}")

        return {
            "model": result["model"],
            "metrics": metrics,
            "model_path": model_file_path,
            "task_id": task.id,
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"Ошибка выполнения эксперимента: {error_message}")
        task.mark_failed(status_message=error_message)
        raise
    finally:
        task.close()
