from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_sc, X_test_sc, scaler
