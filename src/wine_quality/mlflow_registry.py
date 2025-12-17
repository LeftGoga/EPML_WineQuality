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
    experiments = [e.name for e in mlflow.list_experiments()]
    if name not in experiments:
        mlflow.create_experiment(name)
    mlflow.set_experiment(name)


def log_metadata(metadata: dict[str, Any], artifact_path: str = "metadata") -> None:
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
    client = client or MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    except MlflowException as e:
        err = str(e)

        if "RESOURCE_DOES_NOT_EXIST" in err or "does not exist" in err or "Registered model" in err:
            try:
                client.create_registered_model(model_name)
            except MlflowException:
                pass
            mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
        else:
            raise

    version = str(mv.version)

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
    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )


def set_model_version_tags(model_name: str, version: str, tags: dict[str, Any]) -> None:
    client = MlflowClient()
    for k, v in tags.items():
        client.set_model_version_tag(name=model_name, version=version, key=str(k), value=str(v))


def list_model_versions(model_name: str) -> list[ModelVersion]:
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    return cast(list[ModelVersion], versions)


def compare_versions(model_name: str, versions: list[str]) -> dict[str, dict[str, Any]]:
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
    client = MlflowClient()
    duplicates = []

    versions = client.search_model_versions(f"name='{model_name}'")
    by_run: dict[str, list[dict[str, Any]]] = {}
    for v in versions:
        entry = {
            "version": v.version,
            "run_id": v.run_id,
            "source": v.source,
            "stage": getattr(v, "current_stage", "None"),
        }
        by_run.setdefault(v.run_id, []).append(entry)

    for _run_id, items in by_run.items():
        if len(items) > 1:
            duplicates.extend(items)
    return duplicates


def cleanup_duplicate_versions(model_name: str, dry_run: bool = True) -> list[dict[str, Any]]:
    client = MlflowClient()
    actions = []

    versions = client.search_model_versions(f"name='{model_name}'")

    by_run: dict[str, list[ModelVersion]] = {}
    for v in versions:
        by_run.setdefault(v.run_id, []).append(v)

    for _run_id, items in by_run.items():
        if len(items) <= 1:
            continue

        runs_versions = [v for v in items if v.source and v.source.startswith("runs:/")]
        models_versions = [v for v in items if v.source and v.source.startswith("models:/")]
        if runs_versions and models_versions:
            for mv in models_versions:
                actions.append({"version": mv.version, "action": "delete_models_source"})
                if not dry_run:
                    client.delete_model_version(name=model_name, version=mv.version)
    return actions
