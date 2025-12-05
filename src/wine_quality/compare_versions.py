# compare_versions.py
import sys
from typing import Any

from mlflow_registry import compare_versions, list_model_versions


def pretty_print_cmp(cmp_dict: dict[str, dict[str, Any]]) -> None:
    for v, info in cmp_dict.items():
        print(f"\n=== Version: {v} (stage={info.get('current_stage')}) ===")
        print("Metrics:")
        for k, val in info["metrics"].items():
            print(f"  {k}: {val}")
        print("Params:")
        for k, val in info["params"].items():
            print(f"  {k}: {val}")
        print("Tags:")
        for k, val in info["tags"].items():
            print(f"  {k}: {val}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_versions.py <ModelName> [v1 v2 ...]")
        sys.exit(1)
    model_name = sys.argv[1]
    versions = sys.argv[2:]
    if not versions:
        vers = list_model_versions(model_name)
        versions = [v.version for v in vers]
        print(f"No versions provided — будет сравнение всех доступных: {versions}")
    cmp = compare_versions(model_name, versions)
    pretty_print_cmp(cmp)
