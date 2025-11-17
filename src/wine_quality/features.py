import pandas as pd
from numpy import ndarray
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_acidity"] = df["fixed_acidity"] + df["volatile_acidity"] + df["citric_acid"]
    df["free_so2_ratio"] = df["free_sulfur_dioxide"] / (df["total_sulfur_dioxide"] + 1)
    df["alcohol_ph"] = df["alcohol"] * df["pH"]
    return df


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[ndarray, ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
