import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from config import (
    BOOSTING_LEARNING_RATE,
    BOOSTING_MAX_DEPTH,
    BOOSTING_N_ESTIMATORS,
    DATA_URL,
    MLFLOW_EXPERIMENT_NAME,
    MLP_HIDDEN_LAYER_SIZES,
    MLP_MAX_ITER,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    TEST_SIZE,
)
from features import engineer_features, scale_features
from mlflow_context import MLflowTrackingContext
from model import ModelType, run_experiment
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from data import create_target, get_features_and_target, load_data, split_data

if "mlflow_initialized" not in st.session_state:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    from urllib.parse import urlparse, urlunparse

    import mlflow

    final_tracking_uri = tracking_uri
    if username and password:
        parsed = urlparse(tracking_uri)
        netloc = f"{username}:{password}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        final_tracking_uri = urlunparse(
            (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
        )

    mlflow.set_tracking_uri(final_tracking_uri)
    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    st.session_state["mlflow_initialized"] = True
    st.session_state["mlflow_tracking_uri"] = final_tracking_uri

st.set_page_config(page_title="Wine Quality Analyzer", layout="wide")

st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел",
    ["Загрузка и просмотр данных", "Визуализация", "Обучение модели", "Предсказание качества"],
)


@st.cache_data
def get_data(url: str = DATA_URL, threshold: int = 7) -> pd.DataFrame:
    df = load_data(url)
    df = create_target(df, threshold)
    return df


if page == "Загрузка и просмотр данных":
    st.title("Загрузка и просмотр данных о качестве вина")

    data_url = st.sidebar.text_input("URL датасета", DATA_URL)
    quality_threshold = st.sidebar.slider("Порог для 'good_quality' (quality >= ?)", 5, 8, 7)

    if st.sidebar.button("Загрузить данные"):
        with st.spinner("Загрузка данных..."):
            df = get_data(data_url, quality_threshold)
            st.session_state["df"] = df

    if "df" in st.session_state:
        df = st.session_state["df"]
        st.subheader("Первые строки датасета")
        st.dataframe(df.head())

        st.subheader("Статистика")
        st.dataframe(df.describe())

        st.subheader("Распределение 'good_quality'")
        st.bar_chart(df["good_quality"].value_counts())

elif page == "Визуализация":
    st.title("Визуализация данных")

    if "df" not in st.session_state:
        st.warning("Сначала загрузите данные в разделе 'Загрузка и просмотр данных'")
    else:
        df = st.session_state["df"]

        if st.button("Показать распределение качества"):
            fig, ax = plt.subplots(figsize=(8, 5))
            counts = df["quality"].value_counts().sort_index()
            counts.plot(kind="bar", ax=ax)
            ax.set_title("Распределение оригинального качества вина")
            ax.set_xlabel("Оценка")
            ax.set_ylabel("Количество")
            st.pyplot(fig)

        if st.button("Показать матрицу корреляций"):
            fig, ax = plt.subplots(figsize=(12, 9))
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Матрица корреляций")
            st.pyplot(fig)

elif page == "Обучение модели":
    st.title("Обучение модели")

    if "df" not in st.session_state:
        st.warning("Сначала загрузите данные в разделе 'Загрузка и просмотр данных'")
    else:
        df = st.session_state["df"]

        test_size = st.sidebar.slider("Размер тестовой выборки", 0.1, 0.5, TEST_SIZE)

        model_type_str = st.sidebar.selectbox(
            "Тип модели",
            ["random_forest", "boosting", "mlp"],
            format_func=lambda x: {
                "random_forest": "Random Forest",
                "boosting": "Gradient Boosting",
                "mlp": "MLP (Neural Network)",
            }[x],
        )
        model_type = ModelType(model_type_str)

        rf_n_estimators = None
        rf_max_depth = None
        boosting_n_estimators = None
        boosting_max_depth = None
        boosting_learning_rate = None
        mlp_hidden_layer_sizes = None
        mlp_max_iter = None

        if model_type == ModelType.RANDOM_FOREST:
            st.sidebar.subheader("Параметры Random Forest")
            rf_n_estimators = st.sidebar.number_input(
                "Количество деревьев", 50, 500, RF_N_ESTIMATORS
            )
            max_depth_input = st.sidebar.number_input(
                "Максимальная глубина (0 для None)",
                0,
                50,
                RF_MAX_DEPTH if RF_MAX_DEPTH is not None else 0,
            )
            rf_max_depth = None if max_depth_input == 0 else max_depth_input

        elif model_type == ModelType.BOOSTING:
            st.sidebar.subheader("Параметры Gradient Boosting")
            boosting_n_estimators = st.sidebar.number_input(
                "Количество деревьев", 50, 500, BOOSTING_N_ESTIMATORS
            )
            boosting_max_depth = st.sidebar.number_input(
                "Максимальная глубина", 1, 10, BOOSTING_MAX_DEPTH
            )
            boosting_learning_rate = st.sidebar.number_input(
                "Learning rate", 0.01, 1.0, BOOSTING_LEARNING_RATE, step=0.01
            )

        elif model_type == ModelType.MLP:
            st.sidebar.subheader("Параметры MLP")
            hidden_layers_str = st.sidebar.text_input(
                "Размеры скрытых слоев (через запятую)", ",".join(map(str, MLP_HIDDEN_LAYER_SIZES))
            )
            try:
                mlp_hidden_layer_sizes = tuple(
                    int(x.strip()) for x in hidden_layers_str.split(",") if x.strip()
                )
            except ValueError:
                st.sidebar.error(
                    "Неверный формат. Используйте числа через запятую, например: 100, 50"
                )
                mlp_hidden_layer_sizes = MLP_HIDDEN_LAYER_SIZES
            mlp_max_iter = st.sidebar.number_input(
                "Максимальное количество итераций", 100, 2000, MLP_MAX_ITER
            )

        st.sidebar.subheader("MLflow настройки")
        use_mlflow = st.sidebar.checkbox("Логировать в MLflow", value=True)
        experiment_name = st.sidebar.text_input("Имя эксперимента", value=MLFLOW_EXPERIMENT_NAME)
        register_model = st.sidebar.checkbox("Зарегистрировать модель в MLflow", value=True)
        register_model_name = None
        if register_model:
            model_name_map = {
                "random_forest": "WineRandomForest",
                "boosting": "WineBoosting",
                "mlp": "WineMLP",
            }
            default_model_name = model_name_map.get(model_type_str, f"Wine{model_type_str.title()}")
            register_model_name = st.sidebar.text_input(
                "Имя модели для регистрации",
                value=default_model_name,
            )

        if st.button("Обучить модель"):
            with st.spinner("Подготовка данных..."):
                X, y = get_features_and_target(df)
                X = engineer_features(X)
                X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
                X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
                st.session_state["scaler"] = scaler

            with st.spinner("Обучение модели..."):
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
                username = os.getenv("MLFLOW_TRACKING_USERNAME")
                password = os.getenv("MLFLOW_TRACKING_PASSWORD")

                with MLflowTrackingContext(
                    tracking_uri=tracking_uri,
                    username=username,
                    password=password,
                ):
                    result = run_experiment(
                        X_train=X_train_sc,
                        y_train=y_train,
                        X_test=X_test_sc,
                        y_test=y_test,
                        model_type=model_type,
                        rf_n_estimators=rf_n_estimators,
                        rf_max_depth=rf_max_depth,
                        boosting_n_estimators=boosting_n_estimators,
                        boosting_max_depth=boosting_max_depth,
                        boosting_learning_rate=boosting_learning_rate,
                        mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
                        mlp_max_iter=mlp_max_iter,
                        use_mlflow=use_mlflow,
                        experiment_name=experiment_name,
                        model_artifact_path="model",
                        save_local=True,
                        register_model_name=register_model_name if register_model else None,
                        log_artifacts=True,
                        df_for_plots=df.drop(columns=["quality", "good_quality"], errors="ignore"),
                    )

                model = result["model"]
                metrics = result["metrics"]
                mlflow_run_id = result.get("mlflow_run_id")

                st.session_state["model"] = model
                st.session_state["model_type"] = model_type_str
                st.session_state["X_test_sc"] = X_test_sc
                st.session_state["y_test"] = y_test
                st.session_state["feature_names"] = X_train_sc.columns.tolist()
                st.session_state["mlflow_run_id"] = mlflow_run_id

            st.success("Модель обучена!")

            if use_mlflow and mlflow_run_id:
                st.info(f"✓ Эксперимент залогирован в MLflow. Run ID: {mlflow_run_id[:8]}...")
                try:
                    import mlflow

                    tracking_uri = mlflow.get_tracking_uri()
                    st.info(f"MLflow UI: {tracking_uri}")
                except Exception:  # nosec B110
                    pass

            st.subheader("Метрики модели")
            st.write(f"Тип модели: **{model_type_str.replace('_', ' ').title()}**")
            st.write(f"Accuracy: {metrics['accuracy']:.4f}")
            st.write(f"F1-score: {metrics['f1_weighted']:.4f}")

            st.subheader("Classification Report")
            st.text(classification_report(y_test, metrics["y_pred"]))

            st.subheader("Confusion Matrix")
            st.text(metrics["confusion_matrix"])

            if hasattr(model, "feature_importances_"):
                if st.button("Показать важность признаков"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importances = model.feature_importances_
                    indices = importances.argsort()[::-1]
                    ordered_names = [st.session_state["feature_names"][i] for i in indices]
                    ax.bar(range(len(importances)), importances[indices], align="center")
                    model_name = model_type_str.replace("_", " ").title()
                    ax.set_title(f"Важность признаков ({model_name})")
                    ax.set_xticks(range(len(importances)))
                    ax.set_xticklabels(ordered_names, rotation=90)
                    st.pyplot(fig)

elif page == "Предсказание качества":
    st.title("Предсказание качества вина")

    if "model" not in st.session_state or "scaler" not in st.session_state:
        st.warning("Сначала обучите модель в разделе 'Обучение модели'")
    else:
        model = st.session_state["model"]
        scaler: StandardScaler = st.session_state["scaler"]
        feature_names = st.session_state["feature_names"]
        model_type_str = st.session_state.get("model_type", "unknown")

        st.info(f"Используется модель: **{model_type_str.replace('_', ' ').title()}**")

        st.subheader("Введите характеристики вина")

        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            with cols[i % 3]:
                input_data[feat] = st.number_input(feat.replace("_", " ").title(), value=0.0)

        if st.button("Предсказать"):
            input_df = pd.DataFrame([input_data])
            input_sc = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
            prediction = model.predict(input_sc)[0]
            prob = model.predict_proba(input_sc)[0][1]

            st.subheader("Результат")
            quality = "Хорошее" if prediction == 1 else "Плохое"
            st.write(f"Предсказанное качество: **{quality}** (вероятность хорошего: {prob:.2f})")
