import argparse

import mlflow
from features import engineer_features, scale_features
from model import ModelType, run_experiment
from utils import plot_correlation_heatmap, plot_feature_importances, plot_quality_distribution

from data import create_target, get_features_and_target, load_data, split_data


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
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
    experiment_name = args.experiment_name or f"wine_{model_type.value}"
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
    )

    model = res["model"]
    metrics = res["metrics"]

    # Визуализации — как раньше
    plot_correlation_heatmap(df.drop(columns=["quality", "good_quality"]))
    plot_feature_importances(model, X.columns.tolist())
