from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, cast

import hydra
import luigi
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .config import BASE_DIR
from .config_schema import AppConfig
from .luigi_pipeline import NotificationSystem, WineQualityPipeline

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Класс для мониторинга выполнения пайплайна."""

    def __init__(self, log_dir: Path | str | None = None):
        if log_dir is None:
            log_dir = BASE_DIR / "outputs" / "pipeline_logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_task_start(self, task_name: str, params: dict[str, Any]) -> None:
        """Логирует начало выполнения задачи."""
        log_entry = {
            "task": task_name,
            "status": "started",
            "params": params,
            "timestamp": str(Path(__file__).stat().st_mtime),
        }
        self._write_log(log_entry)

    def log_task_complete(self, task_name: str, metrics: dict[str, Any] | None = None) -> None:
        """Логирует завершение задачи."""
        log_entry = {
            "task": task_name,
            "status": "completed",
            "metrics": metrics or {},
        }
        self._write_log(log_entry)

    def log_task_failure(self, task_name: str, error: str) -> None:
        """Логирует ошибку выполнения задачи."""
        log_entry = {
            "task": task_name,
            "status": "failed",
            "error": error,
        }
        self._write_log(log_entry)

    def _write_log(self, entry: dict[str, Any]) -> None:
        """Записывает лог в файл."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = self.log_dir / f"pipeline_{timestamp}.json"
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def _prepare_model_params(model_type: str, model_config: Any) -> dict[str, str]:
    """Подготавливает параметры модели для Luigi задачи."""
    model_params = {}

    if model_type == "random_forest":
        if model_config.n_estimators:
            model_params["rf_n_estimators"] = str(model_config.n_estimators)
        if model_config.max_depth:
            model_params["rf_max_depth"] = str(model_config.max_depth)
    elif model_type == "boosting":
        if model_config.n_estimators:
            model_params["boosting_n_estimators"] = str(model_config.n_estimators)
        if model_config.max_depth:
            model_params["boosting_max_depth"] = str(model_config.max_depth)
        if model_config.learning_rate:
            model_params["boosting_learning_rate"] = str(model_config.learning_rate)
    elif model_type == "mlp":
        if model_config.hidden_layer_sizes:
            model_params["mlp_hidden_layer_sizes"] = str(tuple(model_config.hidden_layer_sizes))
        if model_config.max_iter:
            model_params["mlp_max_iter"] = str(model_config.max_iter)

    return model_params


def _prepare_luigi_params(validated_cfg: AppConfig, cfg: DictConfig) -> dict[str, str]:
    """Подготавливает параметры для Luigi задачи."""
    model_type = validated_cfg.model_type
    model_path = str(BASE_DIR / validated_cfg.paths.models_dir / f"wine_{model_type}.pkl")
    features_path = str(BASE_DIR / validated_cfg.paths.data_dir / "features.csv")

    model_params = _prepare_model_params(model_type, validated_cfg.model)

    # Получаем параметры Hydra из текущего контекста
    # Используем относительный путь от корня проекта (относительно точки входа)
    hydra_config_path = "conf"
    hydra_config_name = "config"
    hydra_overrides = []

    # Пытаемся получить overrides из текущего контекста Hydra
    try:
        hydra_instance = GlobalHydra.instance()
        if hydra_instance.is_initialized():
            hydra_cfg = hydra_instance.hydra
            # Получаем overrides из контекста Hydra
            if hasattr(hydra_cfg, "overrides"):
                hydra_overrides = hydra_cfg.overrides.task_overrides
            elif hasattr(hydra_cfg, "config_loader") and hasattr(
                hydra_cfg.config_loader, "overrides"
            ):
                hydra_overrides = hydra_cfg.config_loader.overrides
    except Exception as e:
        logger.debug(f"Не удалось получить overrides из Hydra (это нормально): {e}")

    luigi_params = {
        "features_path": features_path,
        "model_path": model_path,
        "model_type": model_type,
        "quality_threshold": str(validated_cfg.data.quality_threshold),
        "test_size": str(validated_cfg.data.test_size),
        "random_state": str(validated_cfg.random_state),
        "use_mlflow": str(not validated_cfg.no_mlflow),
        "experiment_name": validated_cfg.mlflow.experiment_name,
        "no_mlflow": str(validated_cfg.no_mlflow),
        "hydra_config_path": hydra_config_path,
        "hydra_config_name": hydra_config_name,
        "hydra_overrides": json.dumps(hydra_overrides) if hydra_overrides else None,
        "run_evaluation": "True",
        "enable_notifications": str(not cfg.get("no_notifications", False)),
        **model_params,
    }

    return {k: v for k, v in luigi_params.items() if v is not None}


def _prepare_build_params(cfg: DictConfig) -> dict[str, Any]:
    """Подготавливает параметры для luigi.build()."""
    pipeline_cfg = cfg.get("pipeline", {})
    workers = pipeline_cfg.get("workers", 1)
    parallel = pipeline_cfg.get("parallel", False)
    scheduler_type = pipeline_cfg.get("scheduler", "local")

    build_params = {
        "local_scheduler": scheduler_type == "local",
    }

    if parallel and workers > 1:
        build_params["workers"] = workers
        logger.info(f"Запуск пайплайна с параллельным выполнением: {workers} workers")
    else:
        logger.info("Запуск пайплайна в последовательном режиме")

    return build_params


def _load_metrics(model_path: str) -> dict[str, Any]:
    """Загружает метрики из файла."""
    metrics_path = Path(model_path).parent / f"{Path(model_path).stem}_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))
    return {}


@hydra.main(
    config_path=str(BASE_DIR / "conf"),
    config_name="config",
    version_base=None,
)
def run_pipeline_with_hydra(cfg: DictConfig) -> None:
    """Запускает Luigi пайплайн с конфигурацией из Hydra."""
    monitor = PipelineMonitor()
    notifications = NotificationSystem(enabled=not cfg.get("no_notifications", False))

    try:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if "env" in cfg_dict and cfg_dict["env"] is None:
            del cfg_dict["env"]
        validated_cfg = AppConfig(**cfg_dict)
        logger.info("✓ Конфигурация успешно валидирована")

        luigi_params = _prepare_luigi_params(validated_cfg, cfg)
        monitor.log_task_start("WineQualityPipeline", luigi_params)

        logger.info("Запуск Luigi пайплайна...")
        task = WineQualityPipeline(**luigi_params)
        build_params = _prepare_build_params(cfg)
        success = luigi.build([task], **build_params)

        if success:
            model_path = luigi_params["model_path"]
            metrics = _load_metrics(model_path)

            monitor.log_task_complete("WineQualityPipeline", metrics)
            notifications.notify_success(
                f"Пайплайн успешно завершен. Модель: {validated_cfg.model_type}",
                metrics,
            )
            notifications.notify_completion(
                {
                    "Модель": validated_cfg.model_type,
                    "Accuracy": metrics.get("accuracy", "N/A"),
                    "F1-score": metrics.get("f1_weighted", "N/A"),
                    "Путь к модели": model_path,
                }
            )
        else:
            error_msg = "Пайплайн завершился с ошибками"
            monitor.log_task_failure("WineQualityPipeline", error_msg)
            notifications.notify_failure(error_msg)
            raise RuntimeError(error_msg)

    except Exception as e:
        error_msg = str(e)
        monitor.log_task_failure("WineQualityPipeline", error_msg)
        notifications.notify_failure("Ошибка при выполнении пайплайна", error_msg)
        raise


if __name__ == "__main__":
    run_pipeline_with_hydra()
