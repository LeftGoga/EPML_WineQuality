from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создает новые признаки на основе существующих.

    Добавляет следующие производные признаки:
    - total_acidity: сумма fixed_acidity и volatile_acidity
    - density_per_alcohol: отношение density к alcohol
    - sulfates_to_chlorides: отношение sulfates к chlorides

    Args:
        df: Исходный DataFrame с признаками вина.

    Returns:
        DataFrame с добавленными новыми признаками.

    Example:
        >>> df = load_data()
        >>> df = engineer_features(df)
        >>> print(df.columns.tolist())
    """
    df = df.copy()
    cols = set(df.columns)
    if {"fixed_acidity", "volatile_acidity"}.issubset(cols):
        df["total_acidity"] = df["fixed_acidity"] + df["volatile_acidity"]
    if {"density", "alcohol"}.issubset(cols):
        df["density_per_alcohol"] = df["density"] / (df["alcohol"] + 1e-6)
    if {"sulfates", "chlorides"}.issubset(cols):
        df["sulfates_to_chlorides"] = df["sulfates"] / (df["chlorides"] + 1e-6)
    return df


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Масштабирует признаки с помощью StandardScaler.

    Обучает StandardScaler на обучающей выборке и применяет
    масштабирование к обеим выборкам (train и test).

    Args:
        X_train: DataFrame с обучающими признаками.
        X_test: DataFrame с тестовыми признаками.

    Returns:
        tuple: Кортеж (X_train_scaled, X_test_scaled, scaler).

    Example:
        >>> X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
        >>> print(f"Среднее X_train_sc: {X_train_sc.mean().mean():.2f}")
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_sc, X_test_sc, scaler
