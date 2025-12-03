import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from config import (
    DATA_URL,
    N_ESTIMATORS,
    TEST_SIZE,
)
from features import engineer_features, scale_features
from model import evaluate_model, save_model, train_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from data import create_target, get_features_and_target, load_data, split_data

st.set_page_config(page_title="Wine Quality Analyzer", layout="wide")

st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел",
    ["Загрузка и просмотр данных", "Визуализация", "Обучение модели", "Предсказание качества"],
)


# Функция для загрузки данных (с кэшированием для производительности)
@st.cache_data
def get_data(url: str = DATA_URL, threshold: int = 7) -> pd.DataFrame:
    df = load_data(url)
    df = create_target(df, threshold)
    return df


# Раздел 1: Загрузка и просмотр данных
if page == "Загрузка и просмотр данных":
    st.title("Загрузка и просмотр данных о качестве вина")

    # Опции в сайдбаре
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

# Раздел 2: Визуализация
elif page == "Визуализация":
    st.title("Визуализация данных")

    if "df" not in st.session_state:
        st.warning("Сначала загрузите данные в разделе 'Загрузка и просмотр данных'")
    else:
        df = st.session_state["df"]

        # Визуализация распределения качества
        if st.button("Показать распределение качества"):
            fig, ax = plt.subplots(figsize=(8, 5))
            counts = df["quality"].value_counts().sort_index()
            counts.plot(kind="bar", ax=ax)
            ax.set_title("Распределение оригинального качества вина")
            ax.set_xlabel("Оценка")
            ax.set_ylabel("Количество")
            st.pyplot(fig)

        # Heatmap корреляций
        if st.button("Показать матрицу корреляций"):
            fig, ax = plt.subplots(figsize=(12, 9))
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Матрица корреляций")
            st.pyplot(fig)

# Раздел 3: Обучение модели
elif page == "Обучение модели":
    st.title("Обучение модели Random Forest")

    if "df" not in st.session_state:
        st.warning("Сначала загрузите данные в разделе 'Загрузка и просмотр данных'")
    else:
        df = st.session_state["df"]

        # Опции в сайдбаре
        test_size = st.sidebar.slider("Размер тестовой выборки", 0.1, 0.5, TEST_SIZE)
        n_estimators = st.sidebar.number_input("Количество деревьев", 50, 500, N_ESTIMATORS)
        max_depth = st.sidebar.number_input("Максимальная глубина (0 для None)", 0, 50, 0)
        max_depth = None if max_depth == 0 else max_depth

        if st.button("Обучить модель"):
            with st.spinner("Подготовка данных..."):
                X, y = get_features_and_target(df)
                X = engineer_features(X)
                X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
                X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
                st.session_state["scaler"] = scaler  # Сохраняем scaler для предсказаний

            with st.spinner("Обучение модели..."):
                model = train_model(
                    X_train_sc, y_train, n_estimators=n_estimators, max_depth=max_depth
                )
                st.session_state["model"] = model
                st.session_state["X_test_sc"] = X_test_sc
                st.session_state["y_test"] = y_test
                st.session_state["feature_names"] = X_train_sc.columns.tolist()

            st.success("Модель обучена!")

            # Оценка
            metrics = evaluate_model(model, X_test_sc, y_test)
            st.subheader("Метрики модели")
            st.write(f"Accuracy: {metrics['accuracy']:.4f}")
            st.write(f"F1-score: {metrics['f1']:.4f}")

            st.subheader("Classification Report")
            st.text(classification_report(y_test, metrics["y_pred"]))

            st.subheader("Confusion Matrix")
            st.text(metrics["confusion_matrix"])

            # Важность признаков
            if st.button("Показать важность признаков"):
                fig, ax = plt.subplots(figsize=(10, 6))
                importances = model.feature_importances_
                indices = importances.argsort()[::-1]
                ordered_names = [st.session_state["feature_names"][i] for i in indices]
                ax.bar(range(len(importances)), importances[indices], align="center")
                ax.set_title("Важность признаков (Random Forest)")
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels(ordered_names, rotation=90)
                st.pyplot(fig)

            # Сохранение модели
            if st.button("Сохранить модель"):
                save_model(model)

# Раздел 4: Предсказание качества
elif page == "Предсказание качества":
    st.title("Предсказание качества вина")

    if "model" not in st.session_state or "scaler" not in st.session_state:
        st.warning("Сначала обучите модель в разделе 'Обучение модели'")
    else:
        model: RandomForestClassifier = st.session_state["model"]
        scaler: StandardScaler = st.session_state["scaler"]
        feature_names = st.session_state["feature_names"]

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
