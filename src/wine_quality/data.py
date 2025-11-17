import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, sep=";")
    df.columns = df.columns.str.replace(" ", "_")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["good_quality"] = (df["quality"] >= 7).astype(int)
    return df


def get_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=["quality", "good_quality"])
    y = df["good_quality"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
