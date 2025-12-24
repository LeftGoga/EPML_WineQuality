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
    """Загружает данные о качестве вина.

    Загружает датасет Wine Quality из локального файла или по указанному URL.
    Если файл отсутствует локально, он будет автоматически скачан.

    Args:
        url: URL для загрузки данных. Если None, используется URL из конфигурации
            или локальный файл data/winequality-red.csv.

    Returns:
        DataFrame с данными о вине. Колонки с пробелами заменены на подчеркивания.

    Example:
        >>> df = load_data()
        >>> print(df.head())
        >>> print(df.shape)
    """
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
    """Создает бинарную целевую переменную на основе порога качества.

    Создает колонку 'good_quality' где 1 означает качество >= threshold,
    а 0 означает качество < threshold.

    Args:
        df: Исходный DataFrame с колонкой 'quality'.
        threshold: Порог качества для разделения на хорошее/плохое вино.
            По умолчанию 7.

    Returns:
        DataFrame с добавленной колонкой 'good_quality'.

    Example:
        >>> df = load_data()
        >>> df = create_target(df, threshold=7)
        >>> print(df['good_quality'].value_counts())
    """
    df = df.copy()
    df["good_quality"] = (df["quality"] >= threshold).astype(int)
    return df


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Разделяет данные на признаки и целевую переменную.

    Удаляет колонки 'quality' и 'good_quality' из признаков,
    возвращает 'good_quality' как целевую переменную.

    Args:
        df: DataFrame с данными, должен содержать колонки 'quality' и 'good_quality'.

    Returns:
        tuple: Кортеж (X, y) где X - DataFrame с признаками, y - Series с целевой переменной.

    Example:
        >>> df = load_data()
        >>> df = create_target(df)
        >>> X, y = get_features_and_target(df)
        >>> print(f"Признаки: {X.shape}")
        >>> print(f"Целевая переменная: {y.shape}")
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
    """Разделяет данные на обучающую и тестовую выборки.

    Использует стратифицированное разделение для сохранения пропорций классов.

    Args:
        X: DataFrame с признаками.
        y: Series с целевой переменной.
        test_size: Доля тестовой выборки (0.0-1.0). Если None, используется
            значение из конфигурации (по умолчанию 0.2).
        random_state: Seed для воспроизводимости. Если None, используется
            значение из конфигурации (по умолчанию 42).

    Returns:
        tuple: Кортеж (X_train, X_test, y_train, y_test) с разделенными данными.

    Example:
        >>> X, y = get_features_and_target(df)
        >>> X_train, X_test, y_train, y_test = split_data(
        ...     X, y, test_size=0.2, random_state=42
        ... )
        >>> print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    """
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE
    return cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y),
    )


def save_features(output_path: str | None = None) -> pd.DataFrame:
    """Загружает данные, создает целевую переменную, применяет инженерию признаков и сохраняет.

    Полный пайплайн обработки данных: загрузка -> создание целевой переменной ->
    инженерия признаков -> сохранение в CSV.

    Args:
        output_path: Путь для сохранения обработанных данных. Если None,
            сохраняется в data/features.csv.

    Returns:
        DataFrame с обработанными данными.

    Example:
        >>> df = save_features("data/processed_features.csv")
        >>> print(f"Сохранено {len(df)} строк")
    """
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
