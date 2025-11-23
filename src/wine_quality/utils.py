from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import PLOT_CORR_PATH, PLOT_FEATURES_PATH, PLOT_QUALITY_PATH, PLOTS_DIR
from sklearn.ensemble import RandomForestClassifier

# create dirs early
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_quality_distribution(df: pd.DataFrame, save_path: str | None = None) -> None:
    if save_path is None:
        save_path = str(PLOT_QUALITY_PATH)
    plt.figure(figsize=(8, 5))
    counts = df["quality"].value_counts().sort_index()
    counts.plot(kind="bar")
    plt.title("Распределение оригинального качества вина")
    plt.xlabel("Оценка")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"График сохранён: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | None = None) -> None:
    if save_path is None:
        save_path = str(PLOT_CORR_PATH)
    plt.figure(figsize=(12, 9))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Матрица корреляций")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"График сохранён: {save_path}")


def plot_feature_importances(
    model: RandomForestClassifier,
    feature_names: list[str],
    save_path: str | None = None,
) -> None:
    if save_path is None:
        save_path = str(PLOT_FEATURES_PATH)

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("model does not have attribute 'feature_importances_'")

    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    ordered_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title("Важность признаков (Random Forest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), ordered_names, rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"График сохранён: {save_path}")
