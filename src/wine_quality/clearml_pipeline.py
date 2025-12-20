"""ClearML Pipeline для обучения моделей Wine Quality."""

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationSystem:
    """Система уведомлений о результатах выполнения с поддержкой email."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.email_enabled = self._check_email_enabled()
        self.smtp_config = self._load_smtp_config()

    def _check_email_enabled(self) -> bool:
        """Проверяет, включены ли email уведомления через переменную окружения."""
        email_enabled = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false").lower()
        return email_enabled in ("true", "1", "yes", "on")

    def _load_smtp_config(self) -> dict[str, str | int | None]:
        """Загружает настройки SMTP из переменных окружения."""
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
        """Отправляет email уведомление через SMTP."""
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
        """Форматирует метрики в HTML таблицу."""
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
        """Отправляет уведомление об успешном выполнении."""
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
                    <h2 style='color: #28a745;'>✓ Успешное выполнение</h2>
                    <p>{message}</p>
                    {metrics_html if metrics else ""}
                    <p style='color: #666; font-size: 12px; margin-top: 20px;'>
                        Время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </p>
                </body>
            </html>
            """
            self._send_email(f"✓ Успех: {message[:50]}", email_body, is_html=True)

    def notify_failure(self, message: str, error: str | None = None) -> None:
        """Отправляет уведомление об ошибке."""
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
                    <h2 style='color: #dc3545;'>✗ Ошибка выполнения</h2>
                    <p>{message}</p>
                    {error_details}
                    <p style='color: #666; font-size: 12px; margin-top: 20px;'>
                        Время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </p>
                </body>
            </html>
            """
            self._send_email(f"✗ Ошибка: {message[:50]}", email_body, is_html=True)

    def notify_completion(self, summary: dict[str, Any]) -> None:
        """Отправляет итоговое уведомление."""
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
                    <h2 style='color: #007bff;'>📊 Итоги выполнения пайплайна</h2>
                    {summary_html}
                    <p style='color: #666; font-size: 12px; margin-top: 20px;'>
                        Время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </p>
                </body>
            </html>
            """
            self._send_email("📊 Итоги выполнения пайплайна", email_body, is_html=True)


@PipelineDecorator.component(
    cache=True,
    execution_queue="default",
    name="Load and Prepare Data",
    return_values=["data_path"],
)
def load_and_prepare_data(  # noqa: PLR0912, PLR0915
    quality_threshold: int = 7,
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """Загружает и подготавливает данные."""
    logging.basicConfig(level=logging.INFO)
    component_logger = logging.getLogger(__name__)

    project_root: Path | None = None
    try:
        task = Task.current_task()
        if task:
            project_root = (
                Path(task.get_script().working_dir)
                if hasattr(task.get_script(), "working_dir")
                else None
            )
    except Exception:
        project_root = None

    if not project_root or not project_root.exists():
        try:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
        except Exception:
            project_root = Path.cwd()
            if (project_root / "src" / "wine_quality").exists():
                pass
            elif (project_root.parent / "src" / "wine_quality").exists():
                project_root = project_root.parent
            else:
                raise ImportError("Не удалось определить путь к проекту") from None

    # Добавляем путь к модулям
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

    # Сохраняем данные в файл для передачи между компонентами
    data_dir = BASE_DIR / "data" / "clearml_pipeline"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "prepared_data.pkl"

    with open(data_path, "wb") as f:
        pickle.dump(  # nosec B301
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

    # Логируем метрики в ClearML
    try:
        task = Task.current_task()
        if task:
            task.logger.report_scalar(
                title="Data Info",
                series="train_size",
                value=len(X_train),
                iteration=0,
            )
            task.logger.report_scalar(
                title="Data Info",
                series="test_size",
                value=len(X_test),
                iteration=0,
            )
            task.logger.report_scalar(
                title="Data Info",
                series="features_count",
                value=len(X_train.columns) if hasattr(X_train, "columns") else 0,
                iteration=0,
            )
    except Exception as e:
        component_logger.warning(f"Не удалось залогировать метрики: {e}")

    return str(data_path)


@PipelineDecorator.component(
    cache=True,
    execution_queue="default",
    name="Train Model",
    return_values=["results_path"],
)
def train_model_component(  # noqa: PLR0912, PLR0915
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
    """Обучает модель."""
    logging.basicConfig(level=logging.INFO)
    component_logger = logging.getLogger(__name__)

    project_root: Path | None = None
    try:
        task = Task.current_task()
        if task:
            project_root = (
                Path(task.get_script().working_dir)
                if hasattr(task.get_script(), "working_dir")
                else None
            )
    except Exception:
        project_root = None

    if not project_root or not project_root.exists():
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

    try:
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
        use_mlflow=False,  # MLflow в отдельном пайплайне
        use_clearml=True,  # ClearML логирование внутри run_experiment
        experiment_name=experiment_name,
        clearml_project_name="Wine Quality",
        model_path=str(BASE_DIR / "models" / f"wine_{model_type}.pkl"),
        save_local=True,
        log_artifacts=True,
        df_for_plots=df,
    )

    # Сохраняем результаты
    results_dir = BASE_DIR / "data" / "clearml_pipeline"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"results_{model_type}.pkl"

    with open(results_path, "wb") as f:
        pickle.dump(  # nosec B301
            {
                "model": result["model"],
                "metrics": result["metrics"],
                "model_path": str(BASE_DIR / "models" / f"wine_{model_type}.pkl"),
            },
            f,
        )

    component_logger.info(f"Модель обучена: accuracy={result['metrics'].get('accuracy', 0):.4f}")

    # Логируем метрики и модель в ClearML
    try:
        task = Task.current_task()
        if task:
            # Логируем метрики
            metrics = result["metrics"]
            task.logger.report_scalar(
                title="Metrics",
                series="accuracy",
                value=float(metrics.get("accuracy", 0)),
                iteration=0,
            )
            task.logger.report_scalar(
                title="Metrics",
                series="f1_weighted",
                value=float(metrics.get("f1_weighted", 0)),
                iteration=0,
            )

            # Логируем параметры модели
            task.connect(
                {
                    "model_type": model_type,
                    "random_state": random_state,
                    "boosting_n_estimators": boosting_n_estimators or "default",
                    "boosting_max_depth": boosting_max_depth or "default",
                    "boosting_learning_rate": boosting_learning_rate or "default",
                }
            )

            # Логируем модель как артефакт
            model_file_path = str(BASE_DIR / "models" / f"wine_{model_type}.pkl")
            if Path(model_file_path).exists():
                task.upload_artifact(
                    name=f"model_{model_type}",
                    artifact_object=model_file_path,
                )
                component_logger.info(f"Модель загружена как артефакт: {model_file_path}")
            else:
                component_logger.warning(f"Файл модели не найден: {model_file_path}")
    except Exception as e:
        component_logger.warning(f"Не удалось залогировать метрики/артефакты: {e}")

    return str(results_path)


@PipelineDecorator.component(
    cache=True,
    execution_queue="default",
    name="Evaluate Model",
    return_values=["evaluation_path"],
)
def evaluate_model_component(
    results_path: str,
    data_path: str,  # noqa: ARG001
) -> str:
    """Оценивает модель."""
    logging.basicConfig(level=logging.INFO)
    component_logger = logging.getLogger(__name__)

    project_root: Path | None = None
    try:
        task = Task.current_task()
        if task:
            project_root = (
                Path(task.get_script().working_dir)
                if hasattr(task.get_script(), "working_dir")
                else None
            )
    except Exception:
        project_root = None

    if not project_root or not project_root.exists():
        try:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
        except Exception:
            project_root = Path.cwd()

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
        pickle.dump(evaluation_results, f)  # nosec B301

    component_logger.info(f"Оценка завершена: {evaluation_results}")

    try:
        task = Task.current_task()
        if task:
            task.logger.report_scalar(
                title="Final Evaluation",
                series="accuracy",
                value=float(evaluation_results.get("accuracy", 0)),
                iteration=0,
            )
            task.logger.report_scalar(
                title="Final Evaluation",
                series="f1_weighted",
                value=float(evaluation_results.get("f1_weighted", 0)),
                iteration=0,
            )
            task.upload_artifact(
                name="evaluation_results",
                artifact_object=str(evaluation_path),
            )
    except Exception as e:
        component_logger.warning(f"Не удалось залогировать результаты оценки: {e}")

    return str(evaluation_path)


@PipelineDecorator.pipeline(
    name="Wine Quality Training Pipeline",
    project="Wine Quality",
    version="1.0",
)
def wine_quality_pipeline(
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
    """Главный пайплайн для обучения модели Wine Quality."""
    logger.info("Запуск пайплайна Wine Quality...")
    notifications = NotificationSystem(enabled=True)

    try:
        # Шаг 1: Загрузка и подготовка данных
        data_path = load_and_prepare_data(
            quality_threshold=quality_threshold,
            test_size=test_size,
            random_state=random_state,
        )

        # Шаг 2: Обучение модели
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

        # Шаг 3: Оценка модели
        evaluation_path = evaluate_model_component(
            results_path=results_path,
            data_path=data_path,
        )

        # Загружаем результаты для возврата
        # Преобразуем proxy объект в строку
        eval_path_str = str(evaluation_path)
        # Убираем возможные кавычки или другие символы
        eval_path_str = eval_path_str.strip("'\"")

        with open(eval_path_str, "rb") as f:
            evaluation_results: dict[str, Any] = pickle.load(f)  # nosec B301

        logger.info("Пайплайн завершен успешно!")

        # Отправляем итоговое уведомление об успешном завершении
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
    # Для локального запуска (отладка)
    # Для запуска на сервере закомментируйте следующую строку
    PipelineDecorator.run_locally()

    # Запуск пайплайна
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
