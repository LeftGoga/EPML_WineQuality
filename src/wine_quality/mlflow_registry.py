from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, cast

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def ensure_experiment(name: str) -> None:
    """Создаёт эксперимент, если не существует, и устанавливает его текущим."""
    experiments = [e.name for e in mlflow.list_experiments()]
    if name not in experiments:
        mlflow.create_experiment(name)
    mlflow.set_experiment(name)


def log_metadata(metadata: dict[str, Any], artifact_path: str = "metadata") -> None:
    """Логирует JSON-метаданные как артефакт текущего run-а."""
    with tempfile.TemporaryDirectory() as td:
        meta_path = os.path.join(td, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(meta_path, artifact_path=artifact_path)


def register_model_from_run(
    run_id: str,
    artifact_path: str,
    model_name: str,
    wait_for_ready: bool = True,
    timeout_sec: int = 300,
    client: MlflowClient | None = None,
) -> str:
    """
    Создаёт Model Version, используя существующий артефакт внутри run (`runs:/<run_id>/<artifact_path>`).

    ВАЖНО: эту функцию НЕ стоит вызывать одновременно с логированием модели через
    `mlflow.sklearn.log_model(..., registered_model_name=...)`, потому что это приводит
    к созданию дублирующих версий (источники вида `models:/m-...`).

    Возвращает номер версии (строка).
    """
    client = client or MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    except MlflowException as e:
        err = str(e)
        # Если RegisteredModel не существует, создаём и повторяем
        if "RESOURCE_DOES_NOT_EXIST" in err or "does not exist" in err or "Registered model" in err:
            try:
                client.create_registered_model(model_name)
            except MlflowException:
                # возможно, модель создалась параллельно — игнорируем
                pass
            mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
        else:
            # Не делаем mlflow.register_model (он создаёт models:/... ссылки).
            # Поднимем исключение, чтобы вызывающий код мог выбрать fallback-сценарий.
            raise

    version = str(mv.version)

    # polling до READY
    if wait_for_ready:
        waited = 0
        while waited < timeout_sec:
            mv = client.get_model_version(name=model_name, version=version)
            status = getattr(mv, "status", None)
            if status == "READY" or mv.current_stage is not None:
                break
            time.sleep(2)
            waited += 2

    return version


def transition_model_stage(
    model_name: str, version: str, stage: str, archive_existing_versions: bool = True
) -> None:
    """
    Переводит указанную версию в stage. Обёртка над MlflowClient.
    """
    client = MlflowClient()
    # Note: transition_model_version_stage может быть deprecated в будущих версиях.
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )


def set_model_version_tags(model_name: str, version: str, tags: dict[str, Any]) -> None:
    """Устанавливает теги (metadata) на модельную версию."""
    client = MlflowClient()
    for k, v in tags.items():
        client.set_model_version_tag(name=model_name, version=version, key=str(k), value=str(v))


def list_model_versions(model_name: str) -> list[ModelVersion]:
    client = MlflowClient()
    versions = client.get_latest_versions(name=model_name)
    return cast(list[ModelVersion], versions)


def compare_versions(model_name: str, versions: list[str]) -> dict[str, dict[str, Any]]:
    """
    Сравнивает указанные версии модели по метрикам, параметрам и тегам.
    """
    client = MlflowClient()
    result: dict[str, dict[str, Any]] = {}
    for v in versions:
        mv = client.get_model_version(name=model_name, version=v)
        run_id = mv.run_id
        if run_id is None:
            continue
        run = client.get_run(run_id)
        data = run.data
        result[v] = {
            "metrics": dict(data.metrics),
            "params": dict(data.params),
            "tags": dict(data.tags),
            "source": mv.source,
            "current_stage": mv.current_stage,
        }
    return result


def find_duplicate_versions(model_name: str) -> list[dict[str, Any]]:
    """Находит возможные дубликаты (same run_id but different source types).

    Возвращает список словарей с полями version, run_id, source, current_stage.
    """
    client = MlflowClient()
    duplicates = []
    versions = client.get_latest_versions(
        model_name, stages=["None", "Staging", "Production", "Archived"]
    )
    by_run: dict[str, list[dict[str, Any]]] = {}
    for v in versions:
        entry = {
            "version": v.version,
            "run_id": v.run_id,
            "source": v.source,
            "stage": v.current_stage,
        }
        by_run.setdefault(v.run_id, []).append(entry)

    for _run_id, items in by_run.items():
        if len(items) > 1:
            duplicates.extend(items)
    return duplicates


def cleanup_duplicate_versions(model_name: str, dry_run: bool = True) -> list[dict[str, Any]]:
    """
    Удаляет версии с source типа `models:/m-...`, оставляя версии с `runs:/...` если они есть.
    Внимание: удаление необратимо. По умолчанию dry_run=True — функция только пройдёт и вернёт, что бы сделала.

    Возвращает список действий (версия, action).
    """
    client = MlflowClient()
    actions = []
    versions = client.get_latest_versions(
        model_name, stages=["None", "Staging", "Production", "Archived"]
    )

    # сгруппируем по run_id
    by_run: dict[str, list[ModelVersion]] = {}
    for v in versions:
        by_run.setdefault(v.run_id, []).append(v)

    for _run_id, items in by_run.items():
        if len(items) <= 1:
            continue
        # если есть runs:/ и models:/ — удаляем models:/
        runs_versions = [v for v in items if v.source and v.source.startswith("runs:/")]
        models_versions = [v for v in items if v.source and v.source.startswith("models:/")]
        if runs_versions and models_versions:
            # удаляем models_versions
            for mv in models_versions:
                actions.append({"version": mv.version, "action": "delete_models_source"})
                if not dry_run:
                    client.delete_model_version(name=model_name, version=mv.version)
    return actions
