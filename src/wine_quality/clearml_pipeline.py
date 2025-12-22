import importlib.util
import logging
import os
import pickle  # nosec B403
import smtplib
import ssl
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from clearml import PipelineDecorator, Task
from dotenv import load_dotenv

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_pipeline_queue = os.getenv("CLEARML_PIPELINE_QUEUE", "wine").strip()
PIPELINE_QUEUE = _pipeline_queue if _pipeline_queue else None

_controller_queue = os.getenv("CLEARML_CONTROLLER_QUEUE", "").strip()
if _controller_queue and _controller_queue.lower() != "default":
    CONTROLLER_QUEUE: str | None = _controller_queue
else:
    CONTROLLER_QUEUE = PIPELINE_QUEUE


def _get_project_root() -> Path:
    project_root = None
    try:
        task = Task.current_task()
        if task:
            project_root = (
                Path(task.get_script().working_dir)
                if hasattr(task.get_script(), "working_dir")
                else None
            )
    except Exception:  # nosec B110
        pass

    if not project_root or not project_root.exists():
        try:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
        except Exception:
            project_root = Path.cwd()
            if not (project_root / "src" / "wine_quality").exists():
                if (project_root.parent / "src" / "wine_quality").exists():
                    project_root = project_root.parent
                else:
                    raise ImportError("Не удалось определить путь к проекту") from None

    return project_root


class NotificationSystem:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.email_enabled = self._check_email_enabled()
        self.smtp_config = self._load_smtp_config()

    def _check_email_enabled(self) -> bool:
        email_enabled = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false").lower()
        return email_enabled in ("true", "1", "yes", "on")

    def _load_smtp_config(self) -> dict[str, str | int | None]:
        smtp_host = os.getenv("SMTP_HOST", "smtp.mail.ru")
        smtp_port_str = os.getenv("SMTP_PORT", "465")
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        recipient_email = os.getenv("RECIPIENT_EMAIL")

        try:
            smtp_port = int(smtp_port_str)
        except (ValueError, TypeError):
            smtp_port = 465

        return {
            "host": smtp_host,
            "port": smtp_port,
            "user": smtp_user,
            "password": smtp_password,
            "recipient": recipient_email,
        }

    def _send_email(self, subject: str, body: str, is_html: bool = False) -> None:
        if not self.email_enabled:
            return

        config = self.smtp_config
        if not all([config["user"], config["password"], config["recipient"]]):
            logger.warning(
                "Email уведомления отключены: не указаны SMTP_USER, SMTP_PASSWORD или RECIPIENT_EMAIL"
            )
            return

        smtp_host = config["host"]
        smtp_port = config["port"]
        smtp_user = config["user"]
        smtp_password = config["password"]
        smtp_recipient = config["recipient"]

        if not smtp_host or not smtp_user or not smtp_password or not smtp_recipient:
            return

        host = str(smtp_host)
        port = int(smtp_port) if isinstance(smtp_port, int) else 465
        user = str(smtp_user)
        password = str(smtp_password)
        recipient = str(smtp_recipient)

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = user
            msg["To"] = recipient

            if is_html:
                msg.attach(MIMEText(body, "html", "utf-8"))
            else:
                msg.attach(MIMEText(body, "plain", "utf-8"))

            if port == 465:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(host, port, context=context) as server:
                    server.login(user, password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(host, port) as server:
                    server.starttls()
                    server.login(user, password)
                    server.send_message(msg)

            logger.info(f"Email уведомление отправлено на {recipient}")
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Ошибка аутентификации SMTP: {e}")
        except smtplib.SMTPException as e:
            logger.error(f"Ошибка отправки email: {e}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при отправке email: {e}")

    def _format_metrics_html(self, metrics: dict[str, Any] | None) -> str:
        if not metrics:
            return ""
        rows = "".join(
            f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>{key}</strong></td>"
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{value}</td></tr>"
            for key, value in metrics.items()
        )
        return f"""
        <table style='border-collapse: collapse; width: 100%; margin: 10px 0;'>
            <thead>
                <tr style='background-color: #f2f2f2;'>
                    <th style='padding: 8px; border: 1px solid #ddd; text-align: left;'>Параметр</th>
                    <th style='padding: 8px; border: 1px solid #ddd; text-align: left;'>Значение</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """

    def notify_success(self, message: str, metrics: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        logger.info(f"✓ УСПЕХ: {message}")
        if metrics:
            logger.info(f"  Метрики: {metrics}")

        if self.email_enabled:
            metrics_html = self._format_metrics_html(metrics)
            email_body = f"""
            <html>
                <body style='font-family: Arial, sans-serif;'>
                    <div style='background-color: #f8f9fa; padding: 10px; margin-bottom: 15px; border-left: 4px solid #007bff;'>
                        <p style='margin: 0; color: #007bff; font-weight: bold;'>📧 Уведомление из ClearML</p>
                    </div>
                    <h2 style='color: #28a745;'>✓ Успешное выполнение</h2>
                    <p>{message}</p>
                    {metrics_html if metrics else ""}
                    <p style='color: #666; font-size: 12px; margin-top: 20px;'>
                        Время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </p>
                </body>
            </html>
            """
            self._send_email(f"[ClearML] ✓ Успех: {message[:50]}", email_body, is_html=True)

    def notify_failure(self, message: str, error: str | None = None) -> None:
        if not self.enabled:
            return
        logger.error(f"✗ ОШИБКА: {message}")
        if error:
            logger.error(f"  Детали: {error}")

        if self.email_enabled:
            error_details = (
                f"<p><strong>Детали ошибки:</strong><br><code style='background-color: #f5f5f5; padding: 5px;'>{error}</code></p>"
                if error
                else ""
            )
            email_body = f"""
            <html>
                <body style='font-family: Arial, sans-serif;'>
                    <div style='background-color: #f8f9fa; padding: 10px; margin-bottom: 15px; border-left: 4px solid #007bff;'>
                        <p style='margin: 0; color: #007bff; font-weight: bold;'>📧 Уведомление из ClearML</p>
                    </div>
                    <h2 style='color: #dc3545;'>✗ Ошибка выполнения</h2>
                    <p>{message}</p>
                    {error_details}
                    <p style='color: #666; font-size: 12px; margin-top: 20px;'>
                        Время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </p>
                </body>
            </html>
            """
            self._send_email(f"[ClearML] ✗ Ошибка: {message[:50]}", email_body, is_html=True)

    def notify_completion(self, summary: dict[str, Any]) -> None:
        if not self.enabled:
            return
        logger.info("=" * 80)
        logger.info("ИТОГИ ВЫПОЛНЕНИЯ ПАЙПЛАЙНА")
        logger.info("=" * 80)
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)

        if self.email_enabled:
            summary_html = self._format_metrics_html(summary)
            email_body = f"""
            <html>
                <body style='font-family: Arial, sans-serif;'>
                    <div style='background-color: #f8f9fa; padding: 10px; margin-bottom: 15px; border-left: 4px solid #007bff;'>
                        <p style='margin: 0; color: #007bff; font-weight: bold;'>📧 Уведомление из ClearML</p>
                    </div>
                    <h2 style='color: #007bff;'>📊 Итоги выполнения пайплайна</h2>
                    {summary_html}
                    <p style='color: #666; font-size: 12px; margin-top: 20px;'>
                        Время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </p>
                </body>
            </html>
            """
            self._send_email("[ClearML] 📊 Итоги выполнения пайплайна", email_body, is_html=True)


@PipelineDecorator.component(
    cache=False,
    execution_queue=PIPELINE_QUEUE,
    name="load_data",
    return_values=["data_path"],
)
def load_and_prepare_data(
    quality_threshold: int = 7,
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    logging.basicConfig(level=logging.INFO)
    component_logger = logging.getLogger(__name__)

    project_root = _get_project_root()
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from wine_quality.config import BASE_DIR  # noqa: PLC0415
        from wine_quality.data import (  # noqa: PLC0415
            create_target,
            get_features_and_target,
            load_data,
            split_data,
        )
        from wine_quality.features import engineer_features  # noqa: PLC0415
    except ImportError:
        config_path = src_path / "wine_quality" / "config.py"
        if config_path.exists():
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Не удалось загрузить config из {config_path}") from None
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            BASE_DIR = config_module.BASE_DIR

            for module_name, file_name in [
                ("data", "data.py"),
                ("features", "features.py"),
            ]:
                module_path = src_path / "wine_quality" / file_name
                if module_path.exists():
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if module_name == "data":
                        load_data = module.load_data
                        create_target = module.create_target
                        get_features_and_target = module.get_features_and_target
                        split_data = module.split_data
                    elif module_name == "features":
                        engineer_features = module.engineer_features
        else:
            raise ImportError(f"Не удалось найти модули. Искал в: {config_path}") from None

    component_logger.info("Загрузка данных...")
    df = load_data()
    df = create_target(df, threshold=quality_threshold)
    df = engineer_features(df)

    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    data_dir = BASE_DIR / "data" / "clearml_pipeline"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "prepared_data.pkl"

    with open(data_path, "wb") as f:
        pickle.dump(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "df": df,
            },
            f,
        )

    component_logger.info(f"Данные загружены и сохранены: train={len(X_train)}, test={len(X_test)}")

    return str(data_path)


@PipelineDecorator.component(
    cache=False,
    execution_queue=PIPELINE_QUEUE,
    name="train",
    return_values=["results_path"],
)
def train_model_component(  # noqa: PLR0915
    data_path: str,
    model_type: str = "boosting",
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    boosting_n_estimators: int | None = None,
    boosting_max_depth: int | None = None,
    boosting_learning_rate: float | None = None,
    mlp_hidden_layer_sizes: tuple[int, ...] | None = None,
    mlp_max_iter: int | None = None,
    random_state: int = 42,
    experiment_name: str | None = None,
) -> str:
    logging.basicConfig(level=logging.INFO)
    component_logger = logging.getLogger(__name__)

    project_root = _get_project_root()
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from wine_quality.clearml_model_registry import register_model_with_version  # noqa: PLC0415
        from wine_quality.config import BASE_DIR  # noqa: PLC0415
        from wine_quality.model import ModelType, run_experiment  # noqa: PLC0415
    except ImportError:
        config_path = src_path / "wine_quality" / "config.py"
        model_path = src_path / "wine_quality" / "model.py"

        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Не удалось загрузить config из {config_path}") from None
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        BASE_DIR = config_module.BASE_DIR

        spec = importlib.util.spec_from_file_location("model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Не удалось загрузить model из {model_path}") from None
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        ModelType = model_module.ModelType
        run_experiment = model_module.run_experiment

        registry_path = src_path / "wine_quality" / "clearml_model_registry.py"
        if registry_path.exists():
            spec = importlib.util.spec_from_file_location("clearml_model_registry", registry_path)
            if spec is not None and spec.loader is not None:
                registry_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(registry_module)
                register_model_with_version = registry_module.register_model_with_version
            else:
                register_model_with_version = None
        else:
            register_model_with_version = None

    component_logger.info(f"Обучение модели: {model_type}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)  # nosec B301

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    df = data["df"]

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
        use_clearml=False,
        experiment_name=experiment_name,
        clearml_project_name="Wine Quality",
        model_path=str(BASE_DIR / "models" / f"wine_{model_type}.pkl"),
        save_local=True,
        log_artifacts=True,
        df_for_plots=df,
        register_model_name=None,
    )

    results_dir = BASE_DIR / "data" / "clearml_pipeline"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"results_{model_type}.pkl"

    with open(results_path, "wb") as f:
        pickle.dump(
            {
                "model": result["model"],
                "metrics": result["metrics"],
                "model_path": str(BASE_DIR / "models" / f"wine_{model_type}.pkl"),
            },
            f,
        )

    component_logger.info(f"Модель обучена: accuracy={result['metrics'].get('accuracy', 0):.4f}")

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
                tags=[model_type, "wine_quality", "pipeline"],
                labels={
                    "model_type": model_type,
                    "accuracy": str(metrics.get("accuracy", 0)),
                    "f1_weighted": str(metrics.get("f1_weighted", 0)),
                    "experiment_name": experiment_name or "default",
                },
                auto_version=True,
            )
            version = version_info.get("version", "unknown")
            component_logger.info(
                f"Модель зарегистрирована с версионированием: {model_name}, версия: {version}"
            )
    except Exception as e:
        component_logger.warning(f"Не удалось зарегистрировать модель: {e}")

    return str(results_path)


@PipelineDecorator.component(
    cache=False,
    execution_queue=PIPELINE_QUEUE,
    name="evaluate",
    return_values=["evaluation_path"],
)
def evaluate_model_component(
    results_path: str,
    data_path: str,  # noqa: ARG001
) -> str:
    logging.basicConfig(level=logging.INFO)
    component_logger = logging.getLogger(__name__)

    project_root = _get_project_root()
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        from wine_quality.config import BASE_DIR  # noqa: PLC0415
    except ImportError:
        config_path = src_path / "wine_quality" / "config.py"
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Не удалось загрузить config из {config_path}") from None
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        BASE_DIR = config_module.BASE_DIR

    component_logger.info("Оценка модели...")

    with open(results_path, "rb") as f:
        results = pickle.load(f)  # nosec B301

    model = results["model"]
    metrics = results["metrics"]

    evaluation_results = {
        "accuracy": metrics.get("accuracy", 0),
        "f1_weighted": metrics.get("f1_weighted", 0),
        "model_type": type(model).__name__,
    }

    eval_dir = BASE_DIR / "data" / "clearml_pipeline"
    eval_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = eval_dir / "evaluation_results.pkl"

    with open(evaluation_path, "wb") as f:
        pickle.dump(evaluation_results, f)

    component_logger.info(f"Оценка завершена: {evaluation_results}")

    return str(evaluation_path)


def _get_plots_dir() -> Path:
    project_root = _get_project_root()
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from wine_quality.config import PLOTS_DIR  # noqa: PLC0415

    return Path(PLOTS_DIR)


@PipelineDecorator.pipeline(
    name="Wine Quality Training Pipeline",
    project="Wine Quality",
    version="1.0",
)
def wine_quality_pipeline(  # noqa: PLR0912, PLR0915
    quality_threshold: int = 7,
    test_size: float = 0.2,
    random_state: int = 42,
    model_type: str = "boosting",
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    boosting_n_estimators: int | None = 100,
    boosting_max_depth: int | None = 2,
    boosting_learning_rate: float | None = 0.1,
    mlp_hidden_layer_sizes: tuple[int, ...] | None = None,
    mlp_max_iter: int | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    logger.info("Запуск пайплайна Wine Quality...")
    notifications = NotificationSystem(enabled=True)

    try:
        main_task = Task.current_task()
        if main_task:
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

            main_task.set_name(full_task_name)
            logger.info(f"Имя задачи установлено: {full_task_name}")

            tags = [
                "framework:sklearn",
                "task:classification",
                f"model_type:{model_type}",
                "wine_quality",
                "pipeline",
            ]
            try:
                main_task.add_tags(tags)
                logger.info(f"Теги установлены: {tags}")
            except Exception as e:
                logger.warning(f"Не удалось установить теги: {e}")
    except Exception as e:
        logger.warning(f"Не удалось изменить имя задачи: {e}")

    try:
        data_path = load_and_prepare_data(
            quality_threshold=quality_threshold,
            test_size=test_size,
            random_state=random_state,
        )

        results_path = train_model_component(
            data_path=data_path,
            model_type=model_type,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            boosting_n_estimators=boosting_n_estimators,
            boosting_max_depth=boosting_max_depth,
            boosting_learning_rate=boosting_learning_rate,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
            mlp_max_iter=mlp_max_iter,
            random_state=random_state,
            experiment_name=experiment_name,
        )

        evaluation_path = evaluate_model_component(
            results_path=results_path,
            data_path=data_path,
        )

        eval_path_str = str(evaluation_path).strip("'\"")
        results_path_str = str(results_path).strip("'\"")
        data_path_str = str(data_path).strip("'\"")

        with open(eval_path_str, "rb") as f:
            evaluation_results: dict[str, Any] = pickle.load(f)  # nosec B301

        with open(results_path_str, "rb") as f:
            results_data = pickle.load(f)  # nosec B301
        model_path_from_results = results_data.get("model_path")
        trained_model = results_data.get("model")

        with open(data_path_str, "rb") as f:
            data_dict = pickle.load(f)  # nosec B301
        X_train = data_dict.get("X_train")
        df_for_plots = data_dict.get("df")

        if MATPLOTLIB_AVAILABLE and trained_model is not None:
            try:
                PLOTS_DIR = _get_plots_dir()
                PLOTS_DIR.mkdir(parents=True, exist_ok=True)

                if hasattr(trained_model, "feature_importances_") and X_train is not None:
                    feature_plot_path = PLOTS_DIR / "feature_importances.png"
                    importances = trained_model.feature_importances_
                    indices = importances.argsort()[::-1]
                    ordered_names = [X_train.columns[i] for i in indices]

                    plt.figure(figsize=(10, 6))
                    model_name_display = model_type.replace("_", " ").title()
                    plt.title(f"Важность признаков ({model_name_display})")
                    plt.bar(range(len(importances)), importances[indices], align="center")
                    plt.xticks(range(len(importances)), ordered_names, rotation=90)
                    plt.tight_layout()
                    plt.savefig(feature_plot_path)
                    plt.close()
                    logger.info(f"График важности признаков создан: {feature_plot_path}")

                if df_for_plots is not None and len(df_for_plots) > 0:
                    corr_plot_path = PLOTS_DIR / "correlation_heatmap.png"
                    plt.figure(figsize=(12, 9))
                    corr = df_for_plots.corr()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
                    plt.title("Матрица корреляций")
                    plt.tight_layout()
                    plt.savefig(corr_plot_path)
                    plt.close()
                    logger.info(f"График корреляций создан: {corr_plot_path}")

            except Exception as e:
                logger.warning(f"Не удалось создать графики: {e}")

        logger.info("Пайплайн завершен успешно!")

        try:
            main_task = Task.current_task()
            if main_task:
                main_task.logger.report_scalar(
                    title="Final Metrics",
                    series="accuracy",
                    value=float(evaluation_results.get("accuracy", 0)),
                    iteration=0,
                )
                main_task.logger.report_scalar(
                    title="Final Metrics",
                    series="f1_weighted",
                    value=float(evaluation_results.get("f1_weighted", 0)),
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

                main_task.connect(params_dict)

                if model_path_from_results and Path(model_path_from_results).exists():
                    try:
                        main_task.upload_artifact(
                            name="model.pkl",
                            artifact_object=model_path_from_results,
                        )
                        logger.info(f"Модель залогирована как артефакт: {model_path_from_results}")
                    except Exception as e:
                        logger.warning(f"Не удалось залогировать модель: {e}")

                try:
                    PLOTS_DIR = _get_plots_dir()

                    feature_plot_path = PLOTS_DIR / "feature_importances.png"
                    if feature_plot_path.exists() and feature_plot_path.stat().st_size > 0:
                        main_task.logger.report_image(
                            title="Plots",
                            series="Feature Importances",
                            local_path=str(feature_plot_path),
                            iteration=0,
                        )
                        logger.info(f"График важности признаков залогирован: {feature_plot_path}")
                    else:
                        logger.warning(f"График важности признаков не найден: {feature_plot_path}")

                    corr_plot_path = PLOTS_DIR / "correlation_heatmap.png"
                    if corr_plot_path.exists() and corr_plot_path.stat().st_size > 0:
                        main_task.logger.report_image(
                            title="Plots",
                            series="Correlation Heatmap",
                            local_path=str(corr_plot_path),
                            iteration=0,
                        )
                        logger.info(f"График корреляций залогирован: {corr_plot_path}")
                    else:
                        logger.warning(f"График корреляций не найден: {corr_plot_path}")

                except Exception as e:
                    logger.warning(f"Не удалось залогировать графики: {e}")

                logger.info("Итоги эксперимента залогированы в главную задачу пайплайна")
        except Exception as e:
            logger.warning(f"Не удалось залогировать итоги в главную задачу: {e}")

        summary = {
            "Модель": model_type,
            "Accuracy": evaluation_results.get("accuracy", "N/A"),
            "F1-score": evaluation_results.get("f1_weighted", "N/A"),
            "Эксперимент": experiment_name or "Не указан",
            "Статус": "Успешно завершен",
        }
        notifications.notify_completion(summary)

        return evaluation_results
    except Exception as e:
        error_message = str(e)
        logger.error(f"Ошибка выполнения пайплайна: {error_message}")
        notifications.notify_failure(
            f"Пайплайн завершился с ошибкой: {model_type}",
            error_message,
        )
        raise


if __name__ == "__main__":
    PipelineDecorator.run_locally()

    wine_quality_pipeline(
        quality_threshold=7,
        test_size=0.2,
        random_state=42,
        model_type="boosting",
        boosting_n_estimators=100,
        boosting_max_depth=2,
        boosting_learning_rate=0.1,
        experiment_name="Wine Quality - Boosting",
    )
