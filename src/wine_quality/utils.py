# src/wine_quality/utils.py
import matplotlib

matplotlib.use("Agg")  # ← ЭТО ГЛАВНОЕ — отключает Tkinter полностью
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Создаём папку для графиков один раз
os.makedirs("plots", exist_ok=True)


def plot_quality_distribution(
    df: pd.DataFrame, save_path: str = "plots/quality_distribution.png"
) -> None:
    plt.figure(figsize=(8, 5))
    df["quality"].value_counts().sort_index().plot(kind="bar", color="steelblue")
    plt.title("Распределение оригинального качества вина")
    plt.xlabel("Оценка")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"График сохранён: {save_path}")


def plot_correlation_heatmap(
    df: pd.DataFrame, save_path: str = "plots/correlation_heatmap.png"
) -> None:
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Матрица корреляций")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"График сохранён: {save_path}")


def plot_feature_importances(
    model, feature_names: list[str], save_path: str = "plots/feature_importances.png"
) -> None:
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Важность признаков (Random Forest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"График сохранён: {save_path}")
