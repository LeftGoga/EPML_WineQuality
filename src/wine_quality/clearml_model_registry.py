from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from clearml import Task

    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    Task = None

from .clearml_utils import (  # noqa: E402
    compare_clearml_models,
    get_model_version_info,
    log_model_to_clearml,
)


def register_model_with_version(
    model: Any,
    model_name: str,
    model_path: str | None = None,
    framework: str = "scikit-learn",
    tags: list[str] | None = None,
    labels: dict[str, str] | None = None,
    auto_version: bool = True,
) -> dict[str, Any]:
    if not CLEARML_AVAILABLE:
        logger.warning("ClearML не доступен")
        return {"version": None, "model_name": model_name, "version_info": None}

    task = Task.current_task() if CLEARML_AVAILABLE else None
    version_info = get_model_version_info(model_name, task) if auto_version else None

    version = log_model_to_clearml(
        model=model,
        model_name=model_name,
        model_path=model_path,
        framework=framework,
        tags=tags,
        labels=labels,
        auto_version=auto_version,
    )

    return {
        "version": version,
        "model_name": model_name,
        "version_info": version_info,
    }


def get_model_comparison(
    model_name: str,
    metric_keys: list[str] | None = None,
) -> dict[str, Any]:
    if not CLEARML_AVAILABLE:
        logger.warning("ClearML не доступен")
        return {"model_name": model_name, "versions": [], "best_version": None}

    task = Task.current_task() if CLEARML_AVAILABLE else None
    if not task:
        logger.warning("Нет активной задачи ClearML")
        return {"model_name": model_name, "versions": [], "best_version": None}

    return compare_clearml_models(model_name=model_name, task=task, metric_keys=metric_keys)


def list_model_versions(model_name: str) -> list[dict[str, Any]]:
    comparison = get_model_comparison(model_name)
    versions = comparison.get("versions", [])
    return versions if isinstance(versions, list) else []
