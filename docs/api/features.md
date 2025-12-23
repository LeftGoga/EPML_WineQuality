# Модуль features

Модуль для инженерии признаков.

## Функции

### `engineer_features`

Создает новые признаки из исходных данных.

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame
```

**Параметры:**
- `df` (pd.DataFrame): Исходный DataFrame.

**Возвращает:**
- `pd.DataFrame`: DataFrame с новыми признаками.

**Пример:**

```python
from wine_quality.features import engineer_features

df = engineer_features(df)
print(df.columns)
```

## Создаваемые признаки

Модуль создает различные производные признаки на основе исходных физико-химических свойств вина:

- Соотношения между различными компонентами
- Полиномиальные признаки
- Логарифмические преобразования
- И другие инженерные признаки

## Использование

```python
from wine_quality.data import load_data, create_target
from wine_quality.features import engineer_features

# Загрузка данных
df = load_data()
df = create_target(df, threshold=7)

# Инженерия признаков
df = engineer_features(df)

print(f"Количество признаков: {len(df.columns)}")
```
