"""
Схемы валидации конфигураций с использованием Pydantic.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Схема конфигурации модели."""

    n_estimators: int | None = Field(
        default=None, ge=1, le=10000, description="Количество деревьев"
    )
    max_depth: int | None = Field(
        default=None, ge=1, le=100, description="Максимальная глубина дерева"
    )
    learning_rate: float | None = Field(
        default=None, gt=0.0, le=1.0, description="Скорость обучения"
    )
    hidden_layer_sizes: list[int] | None = Field(
        default=None, description="Размеры скрытых слоев для MLP"
    )
    max_iter: int | None = Field(
        default=None, ge=1, le=10000, description="Максимальное количество итераций"
    )

    @field_validator("hidden_layer_sizes")
    @classmethod
    def validate_hidden_layers(cls, v: list[int] | None) -> list[int] | None:
        """Валидация размеров скрытых слоев."""
        if v is not None:
            if not all(size > 0 for size in v):
                raise ValueError("Все размеры скрытых слоев должны быть положительными")
            if len(v) > 10:
                raise ValueError("Слишком много скрытых слоев (максимум 10)")
        return v


class DataConfig(BaseModel):
    """Схема конфигурации данных."""

    data_url: str = Field(description="URL для загрузки данных")
    test_size: float = Field(ge=0.0, lt=1.0, description="Доля тестовой выборки")
    quality_threshold: int = Field(
        ge=1, le=10, description="Порог качества для бинарной классификации"
    )

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        """Валидация размера тестовой выборки."""
        if not 0.0 < v < 1.0:
            raise ValueError("test_size должен быть между 0 и 1")
        return v


class MLflowConfig(BaseModel):
    """Схема конфигурации MLflow."""

    experiment_name: str = Field(min_length=1, description="Имя эксперимента в MLflow")
    model_artifact_path: str = Field(default="model", description="Путь для сохранения модели")
    log_artifacts: bool = Field(default=True, description="Логировать ли артефакты")


class PathsConfig(BaseModel):
    """Схема конфигурации путей."""

    plots_dir: str = Field(description="Директория для графиков")
    models_dir: str = Field(description="Директория для моделей")
    data_dir: str = Field(description="Директория для данных")
    plot_quality_filename: str = Field(description="Имя файла графика распределения качества")
    plot_corr_filename: str = Field(description="Имя файла графика корреляций")
    plot_features_filename: str = Field(description="Имя файла графика важности признаков")
    model_filename: str = Field(description="Имя файла модели")


class AppConfig(BaseModel):
    """Главная схема конфигурации приложения."""

    model_type: Literal["random_forest", "boosting", "mlp"] = Field(description="Тип модели")
    random_state: int = Field(ge=0, description="Случайное состояние для воспроизводимости")
    no_mlflow: bool = Field(default=False, description="Отключить логирование в MLflow")
    no_save: bool = Field(default=False, description="Не сохранять модель локально")
    analyze_experiments: bool = Field(default=False, description="Показать анализ экспериментов")
    experiment_name: str | None = Field(
        default=None, description="Имя эксперимента (переопределяет mlflow.experiment_name)"
    )
    register_model_name: str | None = Field(default=None, description="Имя модели для регистрации")
    export_comparison: str | None = Field(default=None, description="Путь для экспорта сравнения")

    # Вложенные конфигурации
    model: ModelConfig
    data: DataConfig
    mlflow: MLflowConfig
    paths: PathsConfig

    # Опциональное окружение (может отсутствовать)
    env: dict[str, Any] | None = Field(default=None, description="Настройки окружения")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Валидация типа модели."""
        allowed = {"random_forest", "boosting", "mlp"}
        if v not in allowed:
            raise ValueError(f"model_type должен быть одним из: {allowed}")
        return v

    def model_completeness(self) -> bool:
        """Проверка полноты конфигурации модели в зависимости от типа."""
        if self.model_type == "random_forest":
            return self.model.n_estimators is not None
        elif self.model_type == "boosting":
            return (
                self.model.n_estimators is not None
                and self.model.max_depth is not None
                and self.model.learning_rate is not None
            )
        elif self.model_type == "mlp":
            return self.model.hidden_layer_sizes is not None and self.model.max_iter is not None
        return False
