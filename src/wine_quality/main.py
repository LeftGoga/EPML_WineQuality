import argparse
import os

import mlflow
from config import MLFLOW_EXPERIMENT_NAME
from features import engineer_features, scale_features
from mlflow_utils import (
    compare_runs,
    export_runs_to_csv,
    get_best_runs,
    get_experiment_summary,
    search_runs,
)
from model import ModelType, run_experiment
from utils import plot_correlation_heatmap, plot_feature_importances, plot_quality_distribution

from data import create_target, get_features_and_target, load_data, split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение модели для предсказания качества вина")
    parser.add_argument(
        "--model-type",
        type=str,
        default="boosting",
        choices=["random_forest", "boosting", "mlp"],
        help="Тип модели для обучения (по умолчанию: boosting)",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=None,
        help="Количество деревьев для RandomForest",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Максимальная глубина для RandomForest",
    )
    parser.add_argument(
        "--boosting-n-estimators",
        type=int,
        default=None,
        help="Количество деревьев для Boosting",
    )
    parser.add_argument(
        "--boosting-max-depth",
        type=int,
        default=None,
        help="Максимальная глубина для Boosting",
    )
    parser.add_argument(
        "--boosting-learning-rate",
        type=float,
        default=None,
        help="Скорость обучения для Boosting",
    )
    parser.add_argument(
        "--mlp-hidden-layer-sizes",
        type=str,
        default=None,
        help="Размеры скрытых слоев для MLP (формат: '100,50')",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=None,
        help="Максимальное количество итераций для MLP",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Случайное состояние для воспроизводимости",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Имя эксперимента в MLflow",
    )
    parser.add_argument(
        "--register-model-name",
        type=str,
        default=None,
        help="Имя модели для регистрации в MLflow",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Отключить логирование в MLflow",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять модель локально",
    )
    parser.add_argument(
        "--analyze-experiments",
        action="store_true",
        help="Показать анализ экспериментов после обучения",
    )
    parser.add_argument(
        "--export-comparison",
        type=str,
        default=None,
        help="Экспортировать сравнение экспериментов в CSV файл",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    from mlflow_context import MLflowTrackingContext

    with MLflowTrackingContext(tracking_uri=tracking_uri, username=username, password=password):
        if not args.no_mlflow:
            try:
                experiments = mlflow.search_experiments(max_results=1)
                print(f"✓ Подключение к MLflow успешно: {tracking_uri}")
                if username:
                    print(f"  Используется аутентификация: {username}")
            except Exception as e:
                print(f"⚠ Предупреждение: не удалось подключиться к MLflow: {e}")
                print(f"  Tracking URI: {tracking_uri}")
                print("  Продолжаем без логирования в MLflow...")

        df = load_data()
        print(df.head())
        plot_quality_distribution(df)
        df = create_target(df)
        df = engineer_features(df)

        X, y = get_features_and_target(df)

        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

        mlp_hidden_layer_sizes = None
        if args.mlp_hidden_layer_sizes:
            mlp_hidden_layer_sizes = tuple(
                int(x.strip()) for x in args.mlp_hidden_layer_sizes.split(",")
            )

        model_type = ModelType(args.model_type)
        experiment_name = args.experiment_name or MLFLOW_EXPERIMENT_NAME
        register_model_name = args.register_model_name or f"Wine{model_type.value.title()}"

        res = run_experiment(
            X_train_sc,
            y_train,
            X_test=X_test_sc,
            y_test=y_test,
            model_type=model_type,
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=args.rf_max_depth,
            boosting_n_estimators=args.boosting_n_estimators,
            boosting_max_depth=args.boosting_max_depth,
            boosting_learning_rate=args.boosting_learning_rate,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
            mlp_max_iter=args.mlp_max_iter,
            random_state=args.random_state,
            use_mlflow=not args.no_mlflow,
            experiment_name=experiment_name,
            model_artifact_path="model",
            save_local=not args.no_save,
            register_model_name=register_model_name,
            log_artifacts=True,
            df_for_plots=df.drop(columns=["quality", "good_quality"], errors="ignore"),
        )
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        try:
            registered_models = client.search_registered_models()
            print("All registered models:")
            for model in registered_models:
                print(f"- {model.name}")

            versions = client.search_model_versions("name='WineRF'")
            print("Versions for WineRF:")
            for v in versions:
                print(
                    f"Version: {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id}, Source: {v.source}, Status: {v.status}"
                )

            if not versions:
                print("No versions found for WineRF—model not registered or registry query failed.")
        except Exception as e:
            print("Error querying registry:", str(e))

        model = res["model"]
        metrics = res["metrics"]
        mlflow_run_id = res.get("mlflow_run_id")
        if mlflow_run_id:
            print(f"MLflow run id: {mlflow_run_id}")
            try:
                tracking_uri = mlflow.get_tracking_uri()
                print(f"MLflow tracking URI: {tracking_uri}")
            except Exception as exc:
                print(f"Warning: could not get tracking URI: {exc}")

        plot_correlation_heatmap(df.drop(columns=["quality", "good_quality"]))
        plot_feature_importances(model, X.columns.tolist())

        if args.analyze_experiments and mlflow_run_id:
            print("\n" + "=" * 80)
            print("АНАЛИЗ ЭКСПЕРИМЕНТОВ")
            print("=" * 80)

            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    summary = get_experiment_summary(experiment.experiment_id)
                    print(f"\nСводка по эксперименту '{experiment_name}':")
                    print(f"  Всего runs: {summary['total_runs']}")
                    print(f"  Завершённых: {summary['finished_runs']}")
                    print(f"  Неудачных: {summary['failed_runs']}")

                    if summary["metrics_summary"]:
                        print("\n  Статистика по метрикам:")
                        for metric_name, stats in summary["metrics_summary"].items():
                            print(
                                f"    {metric_name}: "
                                f"mean={stats['mean']:.4f}, "
                                f"min={stats['min']:.4f}, "
                                f"max={stats['max']:.4f}, "
                                f"std={stats['std']:.4f}"
                            )

                runs = search_runs(
                    experiment_ids=[experiment.experiment_id] if experiment else None,
                    filter_string="metrics.accuracy > 0.7",
                    order_by=["metrics.accuracy DESC"],
                    max_results=10,
                )

                if runs:
                    print(f"\n  Найдено {len(runs)} runs с accuracy > 0.7")
                    best_runs = get_best_runs(runs, "accuracy", ascending=False, top_k=5)
                    print("\n  Топ-5 runs по accuracy:")
                    for i, run in enumerate(best_runs, 1):
                        acc = run.data.metrics.get("accuracy", "N/A")
                        f1 = run.data.metrics.get("f1", "N/A")
                        model_t = run.data.params.get("model_type", "N/A")
                        print(
                            f"    {i}. Run {run.info.run_id[:8]}...: "
                            f"accuracy={acc:.4f}, f1={f1:.4f}, model={model_t}"
                        )

                    if args.export_comparison:
                        comparison_df = compare_runs(
                            runs[:10],
                            metric_names=["accuracy", "f1"],
                            param_names=["model_type", "n_estimators", "max_depth"],
                        )
                        export_runs_to_csv(
                            runs[:10],
                            args.export_comparison,
                            metric_names=["accuracy", "f1"],
                            param_names=["model_type", "n_estimators", "max_depth"],
                        )
                        print(
                            f"\n  Сравнение экспериментов экспортировано в {args.export_comparison}"
                        )

            except Exception as e:
                print(f"\n  Ошибка при анализе экспериментов: {e}")

            print("=" * 80)
