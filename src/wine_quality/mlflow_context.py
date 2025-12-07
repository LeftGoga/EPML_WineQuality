"""
Контекстные менеджеры для работы с MLflow.

Этот модуль предоставляет удобные контекстные менеджеры для автоматизации
логирования экспериментов в MLflow.
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

# Импорты для логирования моделей (опциональные, импортируются по требованию)
try:
    import mlflow.sklearn
except ImportError:
    pass

try:
    import mlflow.pytorch
except ImportError:
    pass

try:
    import mlflow.tensorflow
except ImportError:
    pass


class MLflowExperimentContext:
    """
    Контекстный менеджер для работы с экспериментом MLflow.

    Автоматически создаёт эксперимент, если он не существует,
    и устанавливает его как текущий.

    Пример использования:
        with MLflowExperimentContext("my_experiment") as exp:
            mlflow.log_param("param1", "value1")
            mlflow.log_metric("accuracy", 0.95)
    """

    def __init__(
        self,
        experiment_name: str,
        create_if_not_exists: bool = True,
        tags: dict[str, str] | None = None,
    ):
        """
        Инициализирует контекстный менеджер для эксперимента.

        Args:
            experiment_name: Имя эксперимента
            create_if_not_exists: Создавать эксперимент, если не существует
            tags: Теги для эксперимента
        """
        self.experiment_name = experiment_name
        self.create_if_not_exists = create_if_not_exists
        self.tags = tags or {}
        self._previous_experiment_id = None

    def __enter__(self) -> MLflowExperimentContext:
        """Входит в контекст и настраивает эксперимент."""
        # Сохраняем текущий эксперимент
        try:
            current_experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if current_experiment:
                self._previous_experiment_id = current_experiment.experiment_id
            else:
                self._previous_experiment_id = None
        except Exception:
            self._previous_experiment_id = None

        # Создаём или получаем эксперимент
        if self.create_if_not_exists:
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name, tags=self.tags)
            except Exception:
                # Эксперимент уже существует
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment:
                    experiment_id = experiment.experiment_id
                    # Обновляем теги, если они предоставлены
                    if self.tags:
                        client = MlflowClient()
                        for key, value in self.tags.items():
                            client.set_experiment_tag(experiment_id, key, value)
                else:
                    raise
        else:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                raise ValueError(f"Эксперимент '{self.experiment_name}' не существует")
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(self.experiment_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Выходит из контекста."""
        # Восстанавливать предыдущий эксперимент не нужно,
        # так как MLflow управляет этим автоматически


class MLflowRunContext:
    """
    Контекстный менеджер для работы с MLflow run.

    Автоматически создаёт и завершает run, логирует параметры и метрики.

    Пример использования:
        with MLflowRunContext(
            experiment_name="my_experiment",
            run_name="test_run",
            params={"param1": "value1"},
            tags={"tag1": "value1"}
        ) as run:
            mlflow.log_metric("accuracy", 0.95)
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        log_system_info: bool = True,
    ):
        """
        Инициализирует контекстный менеджер для run.

        Args:
            experiment_name: Имя эксперимента (если None, используется текущий)
            run_name: Имя run
            params: Параметры для логирования
            tags: Теги для run
            log_system_info: Логировать системную информацию
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.params = params or {}
        self.tags = tags or {}
        self.log_system_info = log_system_info
        self._run = None

    def __enter__(self) -> mlflow.entities.Run:
        """Входит в контекст и создаёт run."""
        # Устанавливаем эксперимент, если указан
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        # Создаём run
        self._run = mlflow.start_run(run_name=self.run_name, tags=self.tags)

        # Логируем параметры
        if self.params:
            mlflow.log_params(self.params)

        # Логируем системную информацию
        if self.log_system_info:
            try:
                mlflow.log_params(
                    {
                        "python_version": sys.version.split()[0],
                        "platform": platform.platform(),
                    }
                )
            except Exception:  # nosec B110
                pass  # Игнорируем ошибки логирования системной информации

        return self._run

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Выходит из контекста и завершает run."""
        if self._run:
            mlflow.end_run()


@contextmanager
def mlflow_artifact_context(
    artifact_name: str, artifact_path: str | None = None
) -> Generator[Path, None, None]:
    """
    Контекстный менеджер для работы с артефактами.

    Создаёт временную директорию для артефактов и автоматически
    логирует их в MLflow при выходе из контекста.

    Пример использования:
        with mlflow_artifact_context("plots", "visualizations") as artifact_dir:
            plot_path = artifact_dir / "plot.png"
            # Создаём файл
            plt.savefig(plot_path)
            # Файл автоматически залогируется при выходе из контекста
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = Path(temp_dir) / artifact_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            yield artifact_dir
        finally:
            # Логируем все файлы из директории
            if artifact_dir.exists():
                for file_path in artifact_dir.rglob("*"):
                    if file_path.is_file():
                        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)


@contextmanager
def mlflow_model_context(
    model: Any,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
    signature: Any = None,
    input_example: Any = None,
) -> Generator[None, None, None]:
    """
    Контекстный менеджер для логирования модели в MLflow.

    Автоматически определяет тип модели и использует соответствующий
    метод логирования.

    Пример использования:
        with mlflow_model_context(
            model=my_model,
            artifact_path="model",
            registered_model_name="MyModel"
        ):
            # Модель будет залогирована при выходе из контекста
            pass
    """
    try:
        yield
    finally:
        # Определяем тип модели и логируем
        model_type = type(model).__module__

        if "sklearn" in model_type or hasattr(model, "fit"):
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    signature=signature,
                    input_example=input_example,
                )
            except Exception as e:
                print(f"Ошибка при логировании sklearn модели: {e}")
        elif "pytorch" in model_type or "torch" in model_type:
            try:
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                )
            except Exception as e:
                print(f"Ошибка при логировании PyTorch модели: {e}")
        elif "tensorflow" in model_type or "keras" in model_type:
            try:
                mlflow.tensorflow.log_model(
                    model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                )
            except Exception as e:
                print(f"Ошибка при логировании TensorFlow модели: {e}")
        else:
            # Пытаемся использовать общий метод
            try:
                mlflow.pyfunc.log_model(
                    artifact_path=artifact_path,
                    python_model=model,
                    registered_model_name=registered_model_name,
                )
            except Exception as e:
                print(f"Ошибка при логировании модели: {e}")


class MLflowTrackingContext:
    """
    Контекстный менеджер для настройки tracking URI и аутентификации.

    Пример использования:
        with MLflowTrackingContext(
            tracking_uri="http://localhost:5000",
            username="user",
            password="pass"
        ):
            # Все операции MLflow будут использовать указанный tracking URI
            mlflow.log_param("param1", "value1")
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        """
        Инициализирует контекстный менеджер для tracking URI.

        Args:
            tracking_uri: URI для MLflow tracking server
            username: Имя пользователя для аутентификации
            password: Пароль для аутентификации
        """
        self.tracking_uri = tracking_uri
        self.username = username
        self.password = password
        self._previous_uri = None

    def __enter__(self) -> MLflowTrackingContext:
        """Входит в контекст и настраивает tracking URI."""
        # Сохраняем текущий URI
        try:
            self._previous_uri = mlflow.get_tracking_uri()
        except Exception:
            self._previous_uri = None

        # Устанавливаем новый URI
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        # Настраиваем аутентификацию через переменные окружения
        if self.username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
        if self.password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Выходит из контекста и восстанавливает предыдущий URI."""
        # Восстанавливаем предыдущий URI
        if self._previous_uri:
            mlflow.set_tracking_uri(self._previous_uri)

        # Очищаем переменные окружения
        if self.username:
            os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        if self.password:
            os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
