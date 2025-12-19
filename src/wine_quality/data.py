from __future__ import annotations

import urllib.request
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .config import BASE_DIR, DATA_URL, RANDOM_STATE, TEST_SIZE
    from .features import engineer_features
except ImportError:
    from config import BASE_DIR, DATA_URL, RANDOM_STATE, TEST_SIZE
    from features import engineer_features


def load_data(url: str | None = None) -> pd.DataFrame:
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    local_path = data_dir / "winequality-red.csv"

    if url is not None:
        df = pd.read_csv(url, sep=";")
    else:
        if not local_path.exists():
            urllib.request.urlretrieve(DATA_URL, local_path)  # nosec B310
        df = pd.read_csv(local_path, sep=";")

    df.columns = df.columns.str.replace(" ", "_")
    return df


def create_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    df = df.copy()
    df["good_quality"] = (df["quality"] >= threshold).astype(int)
    return df


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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


def save_features(output_path: str | None = None) -> pd.DataFrame:
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    if output_path is None:
        output_path = str(data_dir / "features.csv")

    df = load_data()
    df = create_target(df)
    df = engineer_features(df)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    save_features()
