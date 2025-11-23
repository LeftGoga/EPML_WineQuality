from __future__ import annotations

from typing import cast

import pandas as pd
from config import DATA_URL, RANDOM_STATE, TEST_SIZE
from sklearn.model_selection import train_test_split


def load_data(url: str | None = None) -> pd.DataFrame:
    """
    Загружает датасет вина (red wine) из DATA_URL (по умолчанию)
    Возвращает DataFrame c колонками, где пробелы заменены на '_'.
    """
    if url is None:
        url = DATA_URL
    df = pd.read_csv(url, sep=";")
    df.columns = df.columns.str.replace(" ", "_")
    return df


def create_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """
    Добавляет колонку 'good_quality' — бинарная метка (1,0)
    по порогу качества (quality >= threshold).
    Возвращает копию DataFrame.
    """
    df = df.copy()
    df["good_quality"] = (df["quality"] >= threshold).astype(int)
    return df


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Возвращает X (DataFrame) и y (Series) — признаковую матрицу и целевую переменную.
    Удаляет колонки 'quality' и 'good_quality' из X.
    """
    X = df.drop(columns=["quality", "good_quality"])
    y = df["good_quality"]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE
    return cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y),
    )
