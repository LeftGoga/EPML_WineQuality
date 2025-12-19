from __future__ import annotations

import logging
import os
import platform
import sys
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

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
    def __init__(
        self,
        experiment_name: str,
        create_if_not_exists: bool = True,
        tags: dict[str, str] | None = None,
    ):
        self.experiment_name = experiment_name
        self.create_if_not_exists = create_if_not_exists
        self.tags = tags or {}
        self._previous_experiment_id = None

    def __enter__(self) -> MLflowExperimentContext:
        try:
            current_experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if current_experiment:
                self._previous_experiment_id = current_experiment.experiment_id
            else:
                self._previous_experiment_id = None
        except Exception:
            self._previous_experiment_id = None

        if self.create_if_not_exists:
            try:
                client = MlflowClient()
                try:
                    experiment = mlflow.get_experiment_by_name(self.experiment_name)
                    if experiment:
                        if experiment.lifecycle_stage == "deleted":
                            try:
                                client.restore_experiment(experiment.experiment_id)
                                logger.debug(
                                    f"Восстановлен удаленный эксперимент: {self.experiment_name}"
                                )
                                experiment_id = experiment.experiment_id
                            except Exception:
                                new_name = f"{self.experiment_name}_{int(time.time())}"
                                experiment_id = mlflow.create_experiment(new_name, tags=self.tags)
                                logger.debug(
                                    f"Создан новый эксперимент '{new_name}' (не удалось восстановить '{self.experiment_name}')"
                                )
                                self.experiment_name = new_name
                        else:
                            experiment_id = experiment.experiment_id
                            if self.tags:
                                for key, value in self.tags.items():
                                    client.set_experiment_tag(experiment_id, key, value)
                    else:
                        experiment_id = mlflow.create_experiment(
                            self.experiment_name, tags=self.tags
                        )
                except Exception:
                    experiment_id = mlflow.create_experiment(self.experiment_name, tags=self.tags)
            except Exception as e:
                error_msg = str(e)
                if "deleted" in error_msg.lower():
                    try:
                        client = MlflowClient()
                        experiments = client.search_experiments(view_type=3)
                        deleted_exp = None
                        for exp in experiments:
                            if (
                                exp.name == self.experiment_name
                                and exp.lifecycle_stage == "deleted"
                            ):
                                deleted_exp = exp
                                break

                        if deleted_exp:
                            client.restore_experiment(deleted_exp.experiment_id)
                            experiment_id = deleted_exp.experiment_id
                            logger.debug(
                                f"Восстановлен удаленный эксперимент: {self.experiment_name}"
                            )
                        else:
                            new_name = f"{self.experiment_name}_{int(time.time())}"
                            experiment_id = mlflow.create_experiment(new_name, tags=self.tags)
                            logger.debug(
                                f"Создан новый эксперимент '{new_name}' (старый '{self.experiment_name}' был удален)"
                            )
                            self.experiment_name = new_name
                    except Exception:
                        new_name = f"{self.experiment_name}_{int(time.time())}"
                        experiment_id = mlflow.create_experiment(new_name, tags=self.tags)
                        logger.debug(
                            f"Создан новый эксперимент '{new_name}' (не удалось восстановить '{self.experiment_name}')"
                        )
                        self.experiment_name = new_name
                else:
                    experiment = mlflow.get_experiment_by_name(self.experiment_name)
                    if experiment:
                        if experiment.lifecycle_stage == "deleted":
                            try:
                                client = MlflowClient()
                                client.restore_experiment(experiment.experiment_id)
                                logger.debug(
                                    f"Восстановлен удаленный эксперимент: {self.experiment_name}"
                                )
                                experiment_id = experiment.experiment_id
                            except Exception:
                                new_name = f"{self.experiment_name}_{int(time.time())}"
                                experiment_id = mlflow.create_experiment(new_name, tags=self.tags)
                                logger.debug(
                                    f"Создан новый эксперимент '{new_name}' (не удалось восстановить '{self.experiment_name}')"
                                )
                                self.experiment_name = new_name
                        else:
                            experiment_id = experiment.experiment_id
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
            if experiment.lifecycle_stage == "deleted":
                raise ValueError(
                    f"Эксперимент '{self.experiment_name}' был удален. "
                    "Используйте create_if_not_exists=True для автоматического восстановления."
                )
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(self.experiment_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class MLflowRunContext:
    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        log_system_info: bool = True,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.params = params or {}
        self.tags = tags or {}
        self.log_system_info = log_system_info
        self._run = None

    def __enter__(self) -> mlflow.entities.Run:
        if self.experiment_name:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment and experiment.lifecycle_stage == "deleted":
                try:
                    client = MlflowClient()
                    client.restore_experiment(experiment.experiment_id)
                    logger.debug(f"Восстановлен удаленный эксперимент: {self.experiment_name}")
                except Exception as e:
                    raise ValueError(
                        f"Эксперимент '{self.experiment_name}' был удален и не может быть восстановлен: {e}"
                    ) from e
            mlflow.set_experiment(self.experiment_name)

        self._run = mlflow.start_run(run_name=self.run_name, tags=self.tags)

        if self.params:
            mlflow.log_params(self.params)

        if self.log_system_info:
            try:
                mlflow.log_params(
                    {
                        "python_version": sys.version.split()[0],
                        "platform": platform.platform(),
                    }
                )
            except Exception:  # nosec B110
                pass

        return self._run

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._run:
            mlflow.end_run()


@contextmanager
def mlflow_artifact_context(
    artifact_name: str, artifact_path: str | None = None
) -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = Path(temp_dir) / artifact_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            yield artifact_dir
        finally:
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
    try:
        yield
    finally:
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
                if registered_model_name:
                    logger.debug(
                        f"Модель {registered_model_name} успешно зарегистрирована в MLflow"
                    )
            except Exception as e:
                logger.error(f"Ошибка при логировании sklearn модели: {e}")
                if registered_model_name:
                    logger.warning(
                        f"Модель {registered_model_name} не была зарегистрирована из-за ошибки"
                    )
        elif "pytorch" in model_type or "torch" in model_type:
            try:
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                )
            except Exception as e:
                logger.error(f"Ошибка при логировании PyTorch модели: {e}")
        elif "tensorflow" in model_type or "keras" in model_type:
            try:
                mlflow.tensorflow.log_model(
                    model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                )
            except Exception as e:
                logger.error(f"Ошибка при логировании TensorFlow модели: {e}")
        else:
            try:
                mlflow.pyfunc.log_model(
                    artifact_path=artifact_path,
                    python_model=model,
                    registered_model_name=registered_model_name,
                )
            except Exception as e:
                logger.error(f"Ошибка при логировании модели: {e}")


class MLflowTrackingContext:
    def __init__(
        self,
        tracking_uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        self.tracking_uri = tracking_uri
        self.username = username
        self.password = password
        self._previous_uri = None

    def __enter__(self) -> MLflowTrackingContext:
        try:
            self._previous_uri = mlflow.get_tracking_uri()
        except Exception:
            self._previous_uri = None

        final_tracking_uri = self.tracking_uri
        if self.tracking_uri and self.username and self.password:
            parsed = urlparse(self.tracking_uri)
            netloc = f"{self.username}:{self.password}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            final_tracking_uri = urlunparse(
                (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
            )

        if final_tracking_uri:
            mlflow.set_tracking_uri(final_tracking_uri)

        if self.username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
        if self.password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._previous_uri:
            mlflow.set_tracking_uri(self._previous_uri)

        if self.username:
            os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        if self.password:
            os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
