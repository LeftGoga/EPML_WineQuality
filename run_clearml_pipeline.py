#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from wine_quality.clearml_pipeline import PipelineDecorator, wine_quality_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск ClearML пайплайна для обучения моделей Wine Quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Запуск с boosting моделью (по умолчанию)
  poetry run python run_clearml_pipeline.py

  # Запуск с Random Forest
  poetry run python run_clearml_pipeline.py --model-type random_forest --rf-n-estimators 200 --rf-max-depth 5

  # Запуск с MLP
  poetry run python run_clearml_pipeline.py --model-type mlp --mlp-hidden-layer-sizes 100 50 --mlp-max-iter 500

  # Запуск с глубоким boosting
  poetry run python run_clearml_pipeline.py --model-type boosting --boosting-n-estimators 300 --boosting-max-depth 5 --boosting-learning-rate 0.05

  # Изменение параметров данных
  poetry run python run_clearml_pipeline.py --quality-threshold 6 --test-size 0.3 --random-state 123
        """,
    )

    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=7,
        help="Порог качества для бинарной классификации (по умолчанию: 7)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля тестовой выборки (по умолчанию: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Случайное состояние для воспроизводимости (по умолчанию: 42)",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["boosting", "random_forest", "mlp"],
        default="boosting",
        help="Тип модели для обучения (по умолчанию: boosting)",
    )

    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=None,
        help="Количество деревьев для Random Forest",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Максимальная глубина для Random Forest",
    )

    parser.add_argument(
        "--boosting-n-estimators",
        type=int,
        default=100,
        help="Количество бустинговых итераций (по умолчанию: 100)",
    )
    parser.add_argument(
        "--boosting-max-depth",
        type=int,
        default=2,
        help="Максимальная глубина для бустинга (по умолчанию: 2)",
    )
    parser.add_argument(
        "--boosting-learning-rate",
        type=float,
        default=0.1,
        help="Скорость обучения для бустинга (по умолчанию: 0.1)",
    )

    parser.add_argument(
        "--mlp-hidden-layer-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Размеры скрытых слоев для MLP (например: --mlp-hidden-layer-sizes 100 50)",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=None,
        help="Максимальное количество итераций для MLP",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Имя эксперимента в ClearML (по умолчанию генерируется автоматически)",
    )

    parser.add_argument(
        "--task-name",
        type=str,
        default="WineQuality",
        help="Имя задачи контроллера в ClearML (по умолчанию: WineQuality)",
    )

    parser.add_argument(
        "--queue",
        type=str,
        default="wine",
        help="Очередь для выполнения компонентов пайплайна (по умолчанию: wine)",
    )

    return parser.parse_args()


def get_default_experiment_name(model_type: str) -> str:
    model_names = {
        "boosting": "Boosting",
        "random_forest": "Random Forest",
        "mlp": "MLP",
    }
    return f"Wine Quality - {model_names.get(model_type, model_type.title())}"


if __name__ == "__main__":
    args = parse_args()

    experiment_name = args.experiment_name or get_default_experiment_name(args.model_type)

    mlp_hidden_layer_sizes = (
        tuple(args.mlp_hidden_layer_sizes) if args.mlp_hidden_layer_sizes else None
    )

    from wine_quality.clearml_pipeline import PipelineDecorator

    PipelineDecorator.run_locally()

    wine_quality_pipeline(
        quality_threshold=args.quality_threshold,
        test_size=args.test_size,
        random_state=args.random_state,
        model_type=args.model_type,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        boosting_n_estimators=args.boosting_n_estimators,
        boosting_max_depth=args.boosting_max_depth,
        boosting_learning_rate=args.boosting_learning_rate,
        mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
        mlp_max_iter=args.mlp_max_iter,
        experiment_name=experiment_name,
    )
