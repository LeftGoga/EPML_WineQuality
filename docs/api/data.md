# Модуль data

Модуль для загрузки и подготовки данных.

## Функции

### `load_data`

Загружает данные о качестве вина.

```python
def load_data(url: str | None = None) -> pd.DataFrame
```

**Параметры:**
- `url` (str | None): URL для загрузки данных. Если None, используется URL из конфигурации.

**Возвращает:**
- `pd.DataFrame`: DataFrame с данными о вине.

**Пример:**

```python
from wine_quality.data import load_data

df = load_data()
print(df.head())
```

### `create_target`

Создает бинарную целевую переменную на основе порога качества.

```python
def create_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame
```

**Параметры:**
- `df` (pd.DataFrame): Исходный DataFrame.
- `threshold` (int): Порог качества (по умолчанию 7).

**Возвращает:**
- `pd.DataFrame`: DataFrame с добавленной колонкой `good_quality`.

**Пример:**

```python
from wine_quality.data import create_target

df = create_target(df, threshold=7)
print(df['good_quality'].value_counts())
```

### `get_features_and_target`

Разделяет данные на признаки и целевую переменную.

```python
def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]
```

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными.

**Возвращает:**
- `tuple[pd.DataFrame, pd.Series]`: Кортеж (признаки, целевая переменная).

**Пример:**

```python
from wine_quality.data import get_features_and_target

X, y = get_features_and_target(df)
print(f"Признаки: {X.shape}")
print(f"Целевая переменная: {y.shape}")
```

### `split_data`

Разделяет данные на обучающую и тестовую выборки.

```python
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
```

**Параметры:**
- `X` (pd.DataFrame): Признаки.
- `y` (pd.Series): Целевая переменная.
- `test_size` (float | None): Доля тестовой выборки (по умолчанию из конфигурации).
- `random_state` (int | None): Seed для воспроизводимости (по умолчанию из конфигурации).

**Возвращает:**
- `tuple`: (X_train, X_test, y_train, y_test).

**Пример:**

```python
from wine_quality.data import split_data

X_train, X_test, y_train, y_test = split_data(
    X, y, test_size=0.2, random_state=42
)
```
