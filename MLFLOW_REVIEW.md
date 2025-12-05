# Анализ реализации MLflow

## Общая оценка: ✅ Хорошо реализовано с несколькими улучшениями

---

## ✅ Сильные стороны

### 1. **Архитектура и структура**
- ✅ Четкое разделение ответственности: `mlflow_registry.py` для работы с Registry, `model.py` для обучения
- ✅ Хорошая обработка опциональной зависимости MLflow через try/except
- ✅ Использование контекстного менеджера `mlflow.start_run()` для автоматического закрытия

### 2. **Model Registry**
- ✅ Правильное использование `registered_model_name` в `log_model()` для автоматической регистрации
- ✅ Polling до READY статуса в `register_model_from_run()`
- ✅ Обработка случая, когда RegisteredModel не существует
- ✅ Функции для управления версиями: переходы стадий, теги, сравнение

### 3. **Логирование метаданных**
- ✅ Логирование параметров модели (n_estimators, max_depth, random_state)
- ✅ Логирование метрик (accuracy, f1_weighted)
- ✅ Логирование дополнительных метаданных (feature_columns, n_train, n_test)
- ✅ Попытка создания сигнатуры модели для валидации входных данных

### 4. **Обработка ошибок**
- ✅ Graceful fallback при отсутствии MLflow
- ✅ Обработка исключений при создании сигнатуры
- ✅ Обработка случая, когда версия не найдена после регистрации

---

## ⚠️ Проблемы и улучшения

### 1. **ВАЖНОЕ: Неэффективное использование параметра `name` в `log_model()`**

**Проблема в `model.py:125-136`:**
```python
try:
    mlflow.sklearn.log_model(
        sk_model=sk_model,
        name=artifact_path,  # ⚠️ Этот параметр не существует в MLflow 3.6.0
        registered_model_name=registered_model_name,
        **kwargs,
    )
    used = "name"
except TypeError:  # Всегда будет падать здесь
    # fallback на artifact_path
    mlflow.sklearn.log_model(
        artifact_path=artifact_path,  # ✅ Правильный параметр
        ...
    )
```

**Проблема:**
- В MLflow 3.6.0 параметр `name` в `log_model()` НЕ существует
- Правильный параметр - `artifact_path`
- Код всегда будет падать с `TypeError` на первой попытке, что неэффективно
- Fallback работает, но это лишний overhead

**Текущее поведение:** Код работает благодаря fallback, но неэффективно.

**Рекомендация:** Упростить код, убрав попытку с `name`:
```python
# Упрощенный вариант:
if registered_model_name is not None:
    mlflow.sklearn.log_model(
        sk_model=sk_model,
        artifact_path=artifact_path,  # ✅ Только правильный параметр
        registered_model_name=registered_model_name,
        **kwargs,
    )
else:
    mlflow.sklearn.log_model(
        sk_model=sk_model,
        artifact_path=artifact_path,
        **kwargs,
    )
```

### 2. **Потенциальная проблема с дублированием версий**

**Проблема в `model.py:224-259`:**
После вызова `mlflow.sklearn.log_model(..., registered_model_name=...)` модель уже зарегистрирована. Затем код пытается найти версию и дополнительно вызывает `register_model_from_run()` в fallback случае.

**Риск:** Если `get_latest_versions()` не находит версию сразу после регистрации (race condition), код может создать дубликат через `register_model_from_run()`.

**Рекомендация:**
- Увеличить задержку или добавить retry перед поиском версии
- Или использовать `search_model_versions()` с фильтром по run_id вместо `get_latest_versions()`

### 3. **Неэффективное создание MlflowClient**

**Проблема:** В `mlflow_registry.py` каждая функция создает новый `MlflowClient()`:
- `transition_model_stage()` - строка 99
- `set_model_version_tags()` - строка 111
- `list_model_versions()` - строка 117
- `compare_versions()` - строка 126
- `find_duplicate_versions()` - строка 150
- `cleanup_duplicate_versions()` - строка 178

**Рекомендация:**
- Принимать `client: MlflowClient | None = None` во всех функциях (как в `register_model_from_run()`)
- Или создать модульный уровень client для переиспользования

### 4. **Хардкод tracking URI**

**Проблема в `main.py:10`:**
```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

**Проблема:**
- Хардкод URI делает код негибким
- Не учитывает переменные окружения
- Может конфликтовать с другими настройками

**Рекомендация:**
```python
import os
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
```

### 5. **Отсутствие валидации входных данных**

**Проблема:**
- `transition_model_stage()` не проверяет валидность stage (должен быть "Staging", "Production", "Archived", "None")
- `compare_versions()` не проверяет существование версий перед сравнением

**Рекомендация:** Добавить валидацию входных параметров.

### 6. **Неоптимальный polling**

**Проблема в `mlflow_registry.py:80-88`:**
```python
while waited < timeout_sec:
    mv = client.get_model_version(name=model_name, version=version)
    status = getattr(mv, "status", None)
    if status == "READY" or mv.current_stage is not None:
        break
    time.sleep(2)
    waited += 2
```

**Проблемы:**
- Фиксированная задержка 2 секунды может быть слишком частой
- Нет обработки случая, когда статус становится "FAILED"
- `getattr(mv, "status", None)` - нестандартный способ доступа к атрибуту

**Рекомендация:**
```python
while waited < timeout_sec:
    mv = client.get_model_version(name=model_name, version=version)
    status = mv.status if hasattr(mv, "status") else None
    if status == "READY" or mv.current_stage is not None:
        break
    if status == "FAILED":
        raise RuntimeError(f"Model version {version} failed to become READY")
    time.sleep(min(2, timeout_sec - waited))  # Не спать дольше timeout
    waited += 2
```

### 7. **Неиспользуемая переменная `used`**

**Проблема в `model.py:124, 136, 148`:**
Переменная `used` возвращается, но никогда не используется в `run_experiment()`.

**Рекомендация:** Удалить возврат значения или использовать его для логирования/отладки.

### 8. **Отсутствие логирования важных событий**

**Рекомендация:** Добавить логирование:
- Успешной регистрации модели
- Переходов стадий
- Ошибок при работе с Registry

---

## 🔧 Конкретные исправления

### Исправление 1: Убрать неправильное использование `name`

```python
# В model.py, функция _log_model_compat:
def _log_model_compat(...) -> str | None:
    kwargs = {}
    # ... signature logic ...

    # Убрать попытку с name, использовать только artifact_path
    try:
        if registered_model_name is not None:
            mlflow.sklearn.log_model(
                sk_model=sk_model,
                artifact_path=artifact_path,  # ✅ Только artifact_path
                registered_model_name=registered_model_name,
                **kwargs,
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=sk_model,
                artifact_path=artifact_path,
                **kwargs,
            )
        return "artifact_path"
    except Exception as e:
        raise RuntimeError(f"Failed to log model: {e}") from e
```

### Исправление 2: Улучшить поиск версии после регистрации

```python
# В model.py, функция run_experiment:
if register_model_name:
    client = MlflowClient()
    # Подождать немного для завершения регистрации
    import time
    time.sleep(1)

    # Использовать search_model_versions с фильтром
    versions = client.search_model_versions(f"name='{register_model_name}' AND run_id='{run.info.run_id}'")

    if versions:
        version = str(versions[0].version)
        # ... остальной код ...
    else:
        # Fallback с retry
        for attempt in range(3):
            time.sleep(2)
            versions = client.search_model_versions(f"name='{register_model_name}' AND run_id='{run.info.run_id}'")
            if versions:
                version = str(versions[0].version)
                break
        else:
            # Последняя попытка через register_model_from_run
            try:
                created_version = register_model_from_run(...)
                # ...
```

### Исправление 3: Добавить параметр client во все функции

```python
# В mlflow_registry.py:
def transition_model_stage(
    model_name: str,
    version: str,
    stage: str,
    archive_existing_versions: bool = True,
    client: MlflowClient | None = None,  # ✅ Добавить
) -> None:
    client = client or MlflowClient()
    # ...
```

---

## 📊 Итоговая оценка

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Архитектура | ✅ 9/10 | Хорошая структура, четкое разделение |
| Правильность API | ⚠️ 6/10 | **КРИТИЧЕСКАЯ ошибка с параметром `name`** |
| Обработка ошибок | ✅ 8/10 | Хорошая, но можно улучшить |
| Производительность | ⚠️ 7/10 | Множественное создание клиентов |
| Best practices | ✅ 8/10 | В целом следует best practices |
| Документация | ✅ 9/10 | Хорошие docstrings |

**Общая оценка: 7.8/10**

---

## 🎯 Приоритетные действия

1. **КРИТИЧНО:** Исправить использование параметра `name` в `log_model()`
2. **ВАЖНО:** Улучшить поиск версии после регистрации (race condition)
3. **ЖЕЛАТЕЛЬНО:** Добавить параметр `client` во все функции Registry
4. **ЖЕЛАТЕЛЬНО:** Вынести tracking URI в переменные окружения
5. **ОПЦИОНАЛЬНО:** Улучшить polling логику

---

## ✅ Что работает хорошо

1. ✅ Правильная структура модулей
2. ✅ Обработка опциональной зависимости MLflow
3. ✅ Использование контекстных менеджеров
4. ✅ Логирование метаданных
5. ✅ Функции для сравнения и управления версиями
6. ✅ Обработка дубликатов версий
