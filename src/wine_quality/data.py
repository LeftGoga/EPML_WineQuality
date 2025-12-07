from __future__ import annotations

import urllib.request
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

# Используем try/except для поддержки как абсолютных, так и относительных импортов
try:
    from .config import BASE_DIR, DATA_URL, RANDOM_STATE, TEST_SIZE
    from .features import engineer_features
except ImportError:
    from config import BASE_DIR, DATA_URL, RANDOM_STATE, TEST_SIZE
    from features import engineer_features


def load_data(url: str | None = None) -> pd.DataFrame:
    """
    Загружает датасет вина (red wine) из DATA_URL (по умолчанию)
    Возвращает DataFrame c колонками, где пробелы заменены на '_'.
    Если url is None, данные загружаются в папку data/ в корне проекта, если файл отсутствует.
    """
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    local_path = data_dir / "winequality-red.csv"

    if url is not None:
        # Загрузка из указанного URL без сохранения локально
        df = pd.read_csv(url, sep=";")
    else:
        # Использование дефолтного URL: скачать в data/, если не существует, затем загрузить
        if not local_path.exists():
            urllib.request.urlretrieve(DATA_URL, local_path)  # nosec B310
        df = pd.read_csv(local_path, sep=";")

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


def save_features(output_path: str | None = None) -> pd.DataFrame:
    """
    Загружает данные, создает целевую переменную, применяет feature engineering
    и сохраняет фичи в CSV файл.

    Args:
        output_path: Путь для сохранения CSV файла. Если None, используется data/features.csv

    Returns:
        DataFrame с обработанными фичами
    """
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    if output_path is None:
        output_path = str(data_dir / "features.csv")

    # Загружаем данные
    df = load_data()

    # Создаем целевую переменную
    df = create_target(df)

    # Применяем feature engineering
    df = engineer_features(df)

    # Сохраняем фичи (включая целевую переменную для удобства)
    df.to_csv(output_path, index=False)
    print(f"Фичи сохранены в {output_path}")

    return df


if __name__ == "__main__":
    save_features()
