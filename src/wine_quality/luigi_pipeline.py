from __future__ import annotations

import ast
import json
import logging
import os
import smtplib
import ssl
import sys
import time
import traceback
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, cast

import hydra
import luigi
import pandas as pd
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .config import BASE_DIR
from .data import create_target, get_features_and_target, load_data, split_data
from .features import engineer_features, scale_features
from .model import ModelType, run_experiment, save_model

load_dotenv()

# === Очистка от дублирующих handlers Luigi ===
for logger_name in ["", "luigi", "luigi-interface"]:
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False

# Настройка единого красивого вывода
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
handler.setFormatter(formatter)

logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

# Уровни для специфических логгеров
logging.getLogger("luigi-interface").setLevel(logging.INFO)  # или WARNING, если совсем тихо
logging.getLogger("luigi").setLevel(logging.INFO)
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

# Создаем logger для использования в модуле
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
            smtp_port = 465  # Порт по умолчанию для mail.ru SSL

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

        # Извлекаем значения с проверкой типов для mypy
        smtp_host = config["host"]
        smtp_port = config["port"]
        smtp_user = config["user"]
        smtp_password = config["password"]
        smtp_recipient = config["recipient"]

        # Проверка типов после проверки на None
        if not smtp_host or not smtp_user or not smtp_password or not smtp_recipient:
            return

        # Приведение типов для mypy
        host = str(smtp_host)
        port = int(smtp_port) if isinstance(smtp_port, int) else 465
        user = str(smtp_user)
        password = str(smtp_password)
        recipient = str(smtp_recipient)

        try:
            # Создаем сообщение
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = user
            msg["To"] = recipient

            # Добавляем тело сообщения
            if is_html:
                msg.attach(MIMEText(body, "html", "utf-8"))
            else:
                msg.attach(MIMEText(body, "plain", "utf-8"))

            # Подключение к SMTP серверу
            # Для mail.ru обычно используется SSL на порту 465
            if port == 465:
                # SSL соединение
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(host, port, context=context) as server:
                    server.login(user, password)
                    server.send_message(msg)
            else:
                # STARTTLS для других портов (например, 587)
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

        # Отправка email
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

        # Отправка email
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

        # Отправка email
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


# Настройка логирования для уменьшения вывода
logging.getLogger("luigi").setLevel(logging.INFO)
logging.getLogger("luigi-interface").setLevel(logging.INFO)
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("mlflow.store.db.utils").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)


class PrepareData(luigi.Task):
    """Задача для подготовки исходных данных."""

    output_path = luigi.Parameter(default=str(BASE_DIR / "data" / "winequality-red.csv"))
    config_path = luigi.OptionalParameter(default=None)
    enable_notifications = luigi.BoolParameter(default=True)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.output_path)

    def run(self) -> None:
        notifications = NotificationSystem(enabled=self.enable_notifications)
        start_time = time.time()
        logger.info(f"Начало подготовки данных: {self.output_path}")
        try:
            df = load_data()
            # Сохраняем с разделителем ";", как в исходных данных
            df.to_csv(self.output_path, index=False, sep=";")
            elapsed = time.time() - start_time
            logger.info(f"Данные подготовлены за {elapsed:.2f} секунд")
            notifications.notify_success(
                f"Данные успешно подготовлены за {elapsed:.2f} секунд",
                {"output_path": self.output_path, "rows": len(df), "columns": len(df.columns)},
            )
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных: {e}")
            notifications.notify_failure(
                f"Ошибка при подготовке данных: {self.output_path}", str(e)
            )
            raise


class GenerateFeatures(luigi.Task):
    """Задача для генерации признаков из исходных данных."""

    input_path = luigi.Parameter(default=str(BASE_DIR / "data" / "winequality-red.csv"))
    output_path = luigi.Parameter(default=str(BASE_DIR / "data" / "features.csv"))
    quality_threshold = luigi.IntParameter(default=7)
    config_path = luigi.OptionalParameter(default=None)
    enable_notifications = luigi.BoolParameter(default=True)

    def requires(self) -> PrepareData:
        return PrepareData(
            output_path=self.input_path,
            config_path=self.config_path,
            enable_notifications=self.enable_notifications,
        )

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.output_path)

    def run(self) -> None:
        notifications = NotificationSystem(enabled=self.enable_notifications)
        start_time = time.time()
        logger.info(f"Начало генерации признаков: {self.output_path}")
        try:
            # Загружаем данные из файла, созданного задачей PrepareData
            # Пробуем разные разделители, так как исходные данные используют ";"
            try:
                df = pd.read_csv(self.input_path, sep=";")
            except Exception:
                # Если не получилось с ";", пробуем ","
                df = pd.read_csv(self.input_path, sep=",")
            # Убеждаемся, что колонки имеют правильные имена (без пробелов)
            df.columns = df.columns.str.replace(" ", "_")
            # Проверяем наличие колонки quality
            if "quality" not in df.columns:
                raise ValueError(
                    f"Колонка 'quality' не найдена в данных. Доступные колонки: {list(df.columns)}"
                )
            df = create_target(df, threshold=self.quality_threshold)
            df = engineer_features(df)
            df.to_csv(self.output_path, index=False)
            elapsed = time.time() - start_time
            logger.info(f"Признаки сгенерированы за {elapsed:.2f} секунд")
            notifications.notify_success(
                f"Признаки успешно сгенерированы за {elapsed:.2f} секунд",
                {
                    "output_path": self.output_path,
                    "quality_threshold": self.quality_threshold,
                    "rows": len(df),
                    "features": len(df.columns) - 1,  # исключаем target
                },
            )
        except Exception as e:
            logger.error(f"Ошибка при генерации признаков: {e}")
            notifications.notify_failure(
                f"Ошибка при генерации признаков: {self.output_path}", str(e)
            )
            raise


class TrainModel(luigi.Task):
    """Задача для обучения модели с поддержкой конфигураций Hydra."""

    features_path = luigi.Parameter(default=str(BASE_DIR / "data" / "features.csv"))
    model_path = luigi.Parameter(default=str(BASE_DIR / "models" / "wine_rf.pkl"))
    model_type = luigi.Parameter(default="random_forest")
    test_size = luigi.FloatParameter(default=0.2)
    random_state = luigi.IntParameter(default=42)
    use_mlflow = luigi.BoolParameter(default=True)
    experiment_name = luigi.Parameter(default="wine_quality_experiments")
    no_mlflow = luigi.BoolParameter(default=False)
    rf_n_estimators = luigi.OptionalParameter(default=None)
    rf_max_depth = luigi.OptionalParameter(default=None)
    boosting_n_estimators = luigi.OptionalParameter(default=None)
    boosting_max_depth = luigi.OptionalParameter(default=None)
    boosting_learning_rate = luigi.OptionalParameter(default=None)
    mlp_hidden_layer_sizes = luigi.OptionalParameter(default=None)
    mlp_max_iter = luigi.OptionalParameter(default=None)
    config_path = luigi.OptionalParameter(default=None)
    hydra_config_path = luigi.OptionalParameter(default=None)
    hydra_config_name = luigi.OptionalParameter(default=None)
    hydra_overrides = luigi.OptionalParameter(default=None)
    enable_notifications = luigi.BoolParameter(default=True)

    def requires(self) -> GenerateFeatures:
        return GenerateFeatures(
            output_path=self.features_path,
            config_path=self.config_path,
            enable_notifications=self.enable_notifications,
        )

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.model_path)

    def _load_hydra_config(self) -> DictConfig | None:
        """Загружает конфигурацию Hydra используя hydra.initialize() и hydra.compose()."""
        if not self.hydra_config_path or not self.hydra_config_name:
            return None
        try:
            # Парсим overrides если они переданы
            overrides = []
            if self.hydra_overrides:
                if isinstance(self.hydra_overrides, str):
                    try:
                        overrides = json.loads(self.hydra_overrides)
                    except json.JSONDecodeError:
                        # Если это не JSON, пробуем как строку с разделителями
                        overrides = [
                            o.strip() for o in self.hydra_overrides.split(",") if o.strip()
                        ]
                elif isinstance(self.hydra_overrides, list):
                    overrides = self.hydra_overrides

            # Используем относительный путь от BASE_DIR
            # Hydra.initialize() работает с относительными путями от точки входа
            config_path = self.hydra_config_path
            if Path(config_path).is_absolute():
                # Если путь абсолютный, пытаемся сделать его относительным от BASE_DIR
                try:
                    config_path = str(Path(config_path).relative_to(BASE_DIR))
                except ValueError:
                    # Если не получается сделать относительным, используем как есть
                    pass

            # Деинициализируем Hydra, если он уже инициализирован
            # Это необходимо для повторной инициализации в задачах Luigi
            try:
                hydra_instance = GlobalHydra.instance()
                if hydra_instance.is_initialized():
                    hydra_instance.clear()
            except Exception:  # nosec B110
                pass  # Игнорируем ошибки деинициализации

            # Инициализируем Hydra и загружаем конфигурацию
            # Меняем рабочую директорию на BASE_DIR для правильной работы с относительными путями
            original_cwd = os.getcwd()
            try:
                os.chdir(BASE_DIR)
                with hydra.initialize(config_path=config_path, version_base=None):
                    cfg = hydra.compose(config_name=self.hydra_config_name, overrides=overrides)
                return cfg
            finally:
                os.chdir(original_cwd)
        except Exception as e:
            logger.warning(f"Не удалось загрузить конфигурацию Hydra: {e}")
            logger.debug(f"Детали ошибки Hydra: {traceback.format_exc()}")
            return None

    def _parse_int_parameter(self, value: str | None) -> int | None:
        if not value:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _parse_float_parameter(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_mlp_hidden_layers(self, value: str | None) -> tuple[int, ...] | None:
        if not value:
            return None
        try:
            result = ast.literal_eval(value)
            if isinstance(result, tuple) and all(isinstance(x, int) for x in result):
                return cast(tuple[int, ...], result)
            if isinstance(result, list) and all(isinstance(x, int) for x in result):
                return cast(tuple[int, ...], tuple(result))
            return (100, 50)
        except (ValueError, SyntaxError):
            return (100, 50)

    def _prepare_data(self) -> tuple:
        # Загружаем данные из файла features.csv, созданного задачей GenerateFeatures
        # Файл сохранен с разделителем "," (запятая) по умолчанию в to_csv()
        # Пробуем сначала запятую, потом точку с запятой
        df = None
        for sep in [",", ";"]:
            try:
                df = pd.read_csv(self.features_path, sep=sep)
                # Проверяем, что файл прочитан правильно (больше одной колонки)
                if len(df.columns) > 1:
                    break
            except Exception:  # nosec B112
                continue

        if df is None or len(df.columns) <= 1:
            raise ValueError(
                f"Не удалось правильно прочитать файл {self.features_path}. Проверьте разделитель."
            )

        # Убеждаемся, что колонки имеют правильные имена (без пробелов)
        df.columns = df.columns.str.replace(" ", "_")

        # Проверяем наличие необходимых колонок
        if "good_quality" not in df.columns:
            raise ValueError(
                f"Колонка 'good_quality' не найдена в данных. Доступные колонки: {list(df.columns)}"
            )

        X, y = get_features_and_target(df)
        # Преобразуем параметры в правильные типы
        # Luigi.FloatParameter и IntParameter должны автоматически преобразовывать,
        # но на всякий случай делаем явное преобразование
        test_size = float(self.test_size)
        random_state = int(self.random_state)
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train_sc, X_test_sc, _scaler = scale_features(X_train, X_test)
        return X_train_sc, X_test_sc, y_train, y_test

    def _parse_model_parameters(self, hydra_cfg: DictConfig | None = None) -> dict:
        """Парсит параметры модели, приоритет у Hydra конфигурации."""
        params = {
            "rf_n_estimators": self._parse_int_parameter(self.rf_n_estimators),
            "rf_max_depth": self._parse_int_parameter(self.rf_max_depth),
            "boosting_n_estimators": self._parse_int_parameter(self.boosting_n_estimators),
            "boosting_max_depth": self._parse_int_parameter(self.boosting_max_depth),
            "boosting_learning_rate": self._parse_float_parameter(self.boosting_learning_rate),
            "mlp_hidden_layer_sizes": self._parse_mlp_hidden_layers(self.mlp_hidden_layer_sizes),
            "mlp_max_iter": self._parse_int_parameter(self.mlp_max_iter),
        }

        # Если есть Hydra конфигурация, используем её значения
        if hydra_cfg:
            model_cfg = OmegaConf.select(hydra_cfg, "model", default=None)
            if model_cfg:
                if model_cfg.get("n_estimators") is not None:
                    if self.model_type == "random_forest":
                        params["rf_n_estimators"] = model_cfg.get("n_estimators")
                    elif self.model_type == "boosting":
                        params["boosting_n_estimators"] = model_cfg.get("n_estimators")

                if model_cfg.get("max_depth") is not None:
                    if self.model_type == "random_forest":
                        params["rf_max_depth"] = model_cfg.get("max_depth")
                    elif self.model_type == "boosting":
                        params["boosting_max_depth"] = model_cfg.get("max_depth")

                if model_cfg.get("learning_rate") is not None:
                    params["boosting_learning_rate"] = model_cfg.get("learning_rate")

                if model_cfg.get("hidden_layer_sizes") is not None:
                    hidden_sizes = model_cfg.get("hidden_layer_sizes")
                    if isinstance(hidden_sizes, list):
                        params["mlp_hidden_layer_sizes"] = tuple(hidden_sizes)

                if model_cfg.get("max_iter") is not None:
                    params["mlp_max_iter"] = model_cfg.get("max_iter")

        return params

    def run(self) -> None:
        notifications = NotificationSystem(enabled=self.enable_notifications)
        start_time = time.time()
        logger.info(f"Начало обучения модели: {self.model_type}")
        try:
            hydra_cfg = self._load_hydra_config()
            X_train_sc, X_test_sc, y_train, y_test = self._prepare_data()
            model_type = ModelType(self.model_type.lower())
            params = self._parse_model_parameters(hydra_cfg)

            # Преобразуем random_state в int (Luigi может передавать как строку)
            random_state = (
                int(self.random_state) if isinstance(self.random_state, str) else self.random_state
            )

            result = run_experiment(
                X_train_sc,
                y_train,
                X_test=X_test_sc,
                y_test=y_test,
                model_type=model_type,
                rf_n_estimators=params["rf_n_estimators"],
                rf_max_depth=params["rf_max_depth"],
                boosting_n_estimators=params["boosting_n_estimators"],
                boosting_max_depth=params["boosting_max_depth"],
                boosting_learning_rate=params["boosting_learning_rate"],
                mlp_hidden_layer_sizes=params["mlp_hidden_layer_sizes"],
                mlp_max_iter=params["mlp_max_iter"],
                random_state=random_state,
                use_mlflow=self.use_mlflow and not self.no_mlflow,
                experiment_name=self.experiment_name,
                save_local=True,
                register_model_name=f"Wine{model_type.value.title()}",
            )

            model = result["model"]
            save_model(model, self.model_path)

            # Сохранение метрик для мониторинга
            metrics = result.get("metrics", {})
            metrics_path = (
                Path(self.model_path).parent / f"{Path(self.model_path).stem}_metrics.json"
            )
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "accuracy": float(metrics.get("accuracy", 0.0)),
                        "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
                        "model_type": self.model_type,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            elapsed = time.time() - start_time
            logger.info(
                f"Модель обучена за {elapsed:.2f} секунд. "
                f"Accuracy: {metrics.get('accuracy', 0.0):.4f}"
            )
            notifications.notify_success(
                f"Модель {self.model_type} успешно обучена за {elapsed:.2f} секунд",
                {
                    "model_type": self.model_type,
                    "model_path": self.model_path,
                    "accuracy": metrics.get("accuracy", 0.0),
                    "f1_weighted": metrics.get("f1_weighted", 0.0),
                },
            )
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            notifications.notify_failure(
                f"Ошибка при обучении модели {self.model_type}: {self.model_path}",
                str(e),
            )
            raise


class EvaluateModel(luigi.Task):
    """Задача для оценки модели и генерации отчетов."""

    model_path = luigi.Parameter(default=str(BASE_DIR / "models" / "wine_rf.pkl"))
    report_path = luigi.Parameter(default=str(BASE_DIR / "outputs" / "evaluation_report.json"))
    enable_notifications = luigi.BoolParameter(default=True)

    def requires(self) -> TrainModel:
        return TrainModel(
            model_path=self.model_path, enable_notifications=self.enable_notifications
        )

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.report_path)

    def run(self) -> None:
        notifications = NotificationSystem(enabled=self.enable_notifications)
        start_time = time.time()
        logger.info(f"Начало оценки модели: {self.model_path}")
        try:
            # Загрузка метрик из предыдущего этапа
            metrics_path = (
                Path(self.model_path).parent / f"{Path(self.model_path).stem}_metrics.json"
            )
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
            else:
                metrics = {}

            # Создание отчета
            report = {
                "model_path": self.model_path,
                "metrics": metrics,
                "evaluation_timestamp": datetime.now().isoformat(),
                "status": "completed",
            }

            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=2)

            elapsed = time.time() - start_time
            logger.info(f"Оценка завершена за {elapsed:.2f} секунд")
            notifications.notify_success(
                f"Оценка модели завершена за {elapsed:.2f} секунд",
                {
                    "model_path": self.model_path,
                    "report_path": self.report_path,
                    "accuracy": metrics.get("accuracy", "N/A"),
                    "f1_weighted": metrics.get("f1_weighted", "N/A"),
                },
            )
        except Exception as e:
            logger.error(f"Ошибка при оценке модели: {e}")
            notifications.notify_failure(f"Ошибка при оценке модели: {self.model_path}", str(e))
            raise


class WineQualityPipeline(luigi.WrapperTask):
    """Главный пайплайн, объединяющий все задачи."""

    features_path = luigi.Parameter(default=str(BASE_DIR / "data" / "features.csv"))
    model_path = luigi.Parameter(default=str(BASE_DIR / "models" / "wine_rf.pkl"))
    model_type = luigi.Parameter(default="random_forest")
    quality_threshold = luigi.IntParameter(default=7)
    test_size = luigi.FloatParameter(default=0.2)
    random_state = luigi.IntParameter(default=42)
    use_mlflow = luigi.BoolParameter(default=True)
    experiment_name = luigi.Parameter(default="wine_quality_experiments")
    no_mlflow = luigi.BoolParameter(default=False)
    rf_n_estimators = luigi.OptionalParameter(default=None)
    rf_max_depth = luigi.OptionalParameter(default=None)
    boosting_n_estimators = luigi.OptionalParameter(default=None)
    boosting_max_depth = luigi.OptionalParameter(default=None)
    boosting_learning_rate = luigi.OptionalParameter(default=None)
    mlp_hidden_layer_sizes = luigi.OptionalParameter(default=None)
    mlp_max_iter = luigi.OptionalParameter(default=None)
    config_path = luigi.OptionalParameter(default=None)
    hydra_config_path = luigi.OptionalParameter(default=None)
    hydra_config_name = luigi.OptionalParameter(default=None)
    hydra_overrides = luigi.OptionalParameter(default=None)
    run_evaluation = luigi.BoolParameter(default=True)
    enable_notifications = luigi.BoolParameter(default=True)

    def requires(self) -> list[luigi.Task]:
        tasks = [
            TrainModel(
                features_path=self.features_path,
                model_path=self.model_path,
                model_type=self.model_type,
                test_size=self.test_size,
                random_state=self.random_state,
                use_mlflow=self.use_mlflow,
                experiment_name=self.experiment_name,
                no_mlflow=self.no_mlflow,
                rf_n_estimators=self.rf_n_estimators,
                rf_max_depth=self.rf_max_depth,
                boosting_n_estimators=self.boosting_n_estimators,
                boosting_max_depth=self.boosting_max_depth,
                boosting_learning_rate=self.boosting_learning_rate,
                mlp_hidden_layer_sizes=self.mlp_hidden_layer_sizes,
                mlp_max_iter=self.mlp_max_iter,
                config_path=self.config_path,
                hydra_config_path=self.hydra_config_path,
                hydra_config_name=self.hydra_config_name,
                hydra_overrides=self.hydra_overrides,
                enable_notifications=self.enable_notifications,
            )
        ]
        if self.run_evaluation:
            report_path = str(
                Path(self.model_path).parent / f"{Path(self.model_path).stem}_report.json"
            )
            tasks.append(
                EvaluateModel(
                    model_path=self.model_path,
                    report_path=report_path,
                    enable_notifications=self.enable_notifications,
                )
            )
        return tasks

    def on_success(self) -> None:
        """Вызывается при успешном завершении всех задач пайплайна."""
        if self.enable_notifications:
            notifications = NotificationSystem(enabled=True)
            try:
                # Загрузка метрик из файла
                metrics_path = (
                    Path(self.model_path).parent / f"{Path(self.model_path).stem}_metrics.json"
                )
                metrics = {}
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        metrics = json.load(f)

                notifications.notify_completion(
                    {
                        "Модель": self.model_type,
                        "Accuracy": metrics.get("accuracy", "N/A"),
                        "F1-score": metrics.get("f1_weighted", "N/A"),
                        "Путь к модели": self.model_path,
                        "Статус": "Успешно завершен",
                    }
                )
            except Exception as e:
                logger.warning(f"Не удалось загрузить метрики для итогового уведомления: {e}")
                notifications.notify_completion(
                    {
                        "Модель": self.model_type,
                        "Путь к модели": self.model_path,
                        "Статус": "Успешно завершен",
                    }
                )

    def on_failure(self, exception: Exception) -> None:
        """Вызывается при ошибке выполнения пайплайна."""
        if self.enable_notifications:
            notifications = NotificationSystem(enabled=True)
            notifications.notify_failure(
                f"Пайплайн завершился с ошибкой: {self.model_type}",
                str(exception),
            )


if __name__ == "__main__":
    luigi.run()
