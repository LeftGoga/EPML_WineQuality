from __future__ import annotations

import os
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import MAX_DEPTH, MODEL_PATH, N_ESTIMATORS, RANDOM_STATE


def train_model(
    X_train,
    y_train,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    random_state: int | None = None,
) -> RandomForestClassifier:
    """
    Тренирует RandomForestClassifier с параметрами из конфига по умолчанию.
    Возвращает обученную модель.
    """
    if n_estimators is None:
        n_estimators = N_ESTIMATORS
    if max_depth is None:
        max_depth = MAX_DEPTH
    if random_state is None:
        random_state = RANDOM_STATE

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test, y_test) -> dict[str, Any]:
    """
    Оценивает модель на X_test/y_test, печатает и возвращает словарь метрик.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def save_model(model: Any, path: str | None = None) -> None:
    """
    Сохраняет модель через joblib. Путь по умолчанию — MODEL_PATH.
    Создаёт директорию, если нужно.
    """
    if path is None:
        path = str(MODEL_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Модель сохранена: {path}")


def load_model(path: str | None = None) -> Any:
    """
    Загружает модель из файла.
    """
    if path is None:
        path = str(MODEL_PATH)
    return joblib.load(path)
