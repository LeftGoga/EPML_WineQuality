# main.py
import mlflow
from features import engineer_features, scale_features
from model import run_experiment
from utils import plot_correlation_heatmap, plot_feature_importances, plot_quality_distribution

from data import create_target, get_features_and_target, load_data, split_data

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    df = load_data()
    print(df.head())
    plot_quality_distribution(df)
    df = create_target(df)
    df = engineer_features(df)

    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    res = run_experiment(
        X_train_sc,
        y_train,
        X_test=X_test_sc,
        y_test=y_test,
        n_estimators=300,
        max_depth=None,
        random_state=None,
        use_mlflow=True,  # включаем логирование в MLflow
        experiment_name="wine_rf",  # имя эксперимента (можете поменять)
        model_artifact_path="model",  # где модель хранится в артефактах MLflow
        save_local=True,  # дополнительно сохраняет локальную joblib копию
        register_model_name="WineRF",
    )
    from mlflow.tracking import MlflowClient

    client = MlflowClient()  # Uses the tracking URI you've set
    try:
        # List all registered models
        registered_models = client.search_registered_models()
        print("All registered models:")
        for model in registered_models:
            print(f"- {model.name}")

        # Specifically search for versions of 'WineRF'
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
        # если хотите — можно напечатать ссылку (при локальном tracking server URL в MLFLOW_TRACKING_URI)
        try:
            tracking_uri = mlflow.get_tracking_uri()
            print(f"MLflow tracking URI: {tracking_uri}")
        except Exception as exc:
            # Игнорируем ошибки при получении tracking URI - это не критично
            print(f"Warning: could not get tracking URI: {exc}")

    # Визуализации — как раньше
    plot_correlation_heatmap(df.drop(columns=["quality", "good_quality"]))
    plot_feature_importances(model, X.columns.tolist())
