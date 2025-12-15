"""
Luigi пайплайн для оркестрации задач машинного обучения.
"""

from __future__ import annotations

import ast
from typing import cast

import luigi
from dotenv import load_dotenv

from .config import BASE_DIR
from .data import create_target, get_features_and_target, load_data, split_data
from .features import engineer_features, scale_features
from .model import ModelType, run_experiment, save_model

load_dotenv()


class PrepareData(luigi.Task):
    """
    Задача для подготовки данных - загрузка датасета.
    """

    output_path = luigi.Parameter(default=str(BASE_DIR / "data" / "winequality-red.csv"))

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.output_path)

    def run(self) -> None:
        """Загружает данные из URL и сохраняет локально."""
        print(f"Загрузка данных в {self.output_path}...")
        df = load_data()
        df.to_csv(self.output_path, index=False)
        print(f"✓ Данные сохранены в {self.output_path}")


class GenerateFeatures(luigi.Task):
    """
    Задача для генерации фичей из исходных данных.
    """

    input_path = luigi.Parameter(default=str(BASE_DIR / "data" / "winequality-red.csv"))
    output_path = luigi.Parameter(default=str(BASE_DIR / "data" / "features.csv"))
    quality_threshold = luigi.IntParameter(default=7)

    def requires(self) -> PrepareData:
        return PrepareData(output_path=self.input_path)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.output_path)

    def run(self) -> None:
        """Генерирует фичи из исходных данных."""
        print(f"Генерация фичей из {self.input_path}...")
        df = load_data()
        df = create_target(df, threshold=self.quality_threshold)
        df = engineer_features(df)
        df.to_csv(self.output_path, index=False)
        print(f"✓ Фичи сохранены в {self.output_path}")


class TrainModel(luigi.Task):
    """
    Задача для обучения модели машинного обучения.
    """

    features_path = luigi.Parameter(default=str(BASE_DIR / "data" / "features.csv"))
    model_path = luigi.Parameter(default=str(BASE_DIR / "models" / "wine_rf.pkl"))
    model_type = luigi.Parameter(default="random_forest")
    test_size = luigi.FloatParameter(default=0.2)
    random_state = luigi.IntParameter(default=42)
    use_mlflow = luigi.BoolParameter(default=True)
    experiment_name = luigi.Parameter(default="wine_quality_experiments")
    no_mlflow = luigi.BoolParameter(default=False)

    # Параметры для Random Forest
    rf_n_estimators = luigi.OptionalParameter(default=None)
    rf_max_depth = luigi.OptionalParameter(default=None)

    # Параметры для Boosting
    boosting_n_estimators = luigi.OptionalParameter(default=None)
    boosting_max_depth = luigi.OptionalParameter(default=None)
    boosting_learning_rate = luigi.OptionalParameter(default=None)

    # Параметры для MLP
    mlp_hidden_layer_sizes = luigi.OptionalParameter(default=None)
    mlp_max_iter = luigi.OptionalParameter(default=None)

    def requires(self) -> GenerateFeatures:
        return GenerateFeatures(output_path=self.features_path)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.model_path)

    def _parse_int_parameter(self, value: str | None) -> int | None:
        """Парсит строковый параметр в целое число."""
        if not value:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _parse_float_parameter(self, value: str | None) -> float | None:
        """Парсит строковый параметр в число с плавающей точкой."""
        if not value:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_mlp_hidden_layers(self, value: str | None) -> tuple[int, ...] | None:
        """Парсит строковый параметр в кортеж для MLP hidden_layer_sizes."""
        if not value:
            return None
        try:
            result = ast.literal_eval(value)
            if isinstance(result, tuple) and all(isinstance(x, int) for x in result):
                return cast(tuple[int, ...], result)
            return (100, 50)
        except (ValueError, SyntaxError):
            return (100, 50)

    def _prepare_data(self) -> tuple:
        """Подготавливает данные для обучения."""
        df = load_data()
        df = create_target(df, threshold=7)
        df = engineer_features(df)

        X, y = get_features_and_target(df)
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        X_train_sc, X_test_sc, _scaler = scale_features(X_train, X_test)
        return X_train_sc, X_test_sc, y_train, y_test

    def _parse_model_parameters(self) -> dict:
        """Парсит все параметры модели."""
        return {
            "rf_n_estimators": self._parse_int_parameter(self.rf_n_estimators),
            "rf_max_depth": self._parse_int_parameter(self.rf_max_depth),
            "boosting_n_estimators": self._parse_int_parameter(self.boosting_n_estimators),
            "boosting_max_depth": self._parse_int_parameter(self.boosting_max_depth),
            "boosting_learning_rate": self._parse_float_parameter(self.boosting_learning_rate),
            "mlp_hidden_layer_sizes": self._parse_mlp_hidden_layers(self.mlp_hidden_layer_sizes),
            "mlp_max_iter": self._parse_int_parameter(self.mlp_max_iter),
        }

    def run(self) -> None:
        """Обучает модель машинного обучения."""
        print(f"Обучение модели {self.model_type}...")
        print(f"  Использование фичей из {self.features_path}")

        X_train_sc, X_test_sc, y_train, y_test = self._prepare_data()
        model_type = ModelType(self.model_type.lower())
        params = self._parse_model_parameters()

        result = run_experiment(
            X_train_sc,
            y_train,
            X_test=X_test_sc,
            y_test=y_test,
            model_type=model_type,
            rf_n_estimators=params["rf_n_estimators"],
            rf_max_depth=params["rf_max_depth"],
            boosting_n_estimators=params["boosting_n_estimators"],
            boosting_max_depth=params["boosting_max_depth"],
            boosting_learning_rate=params["boosting_learning_rate"],
            mlp_hidden_layer_sizes=params["mlp_hidden_layer_sizes"],
            mlp_max_iter=params["mlp_max_iter"],
            random_state=self.random_state,
            use_mlflow=self.use_mlflow and not self.no_mlflow,
            experiment_name=self.experiment_name,
            save_local=True,
            register_model_name=f"Wine{model_type.value.title()}",
        )

        model = result["model"]
        save_model(model, self.model_path)
        print(f"✓ Модель сохранена в {self.model_path}")
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  F1-score: {result['metrics']['f1_weighted']:.4f}")


class WineQualityPipeline(luigi.WrapperTask):
    """
    Главная задача пайплайна, которая объединяет все этапы.
    """

    features_path = luigi.Parameter(default=str(BASE_DIR / "data" / "features.csv"))
    model_path = luigi.Parameter(default=str(BASE_DIR / "models" / "wine_rf.pkl"))
    model_type = luigi.Parameter(default="random_forest")
    quality_threshold = luigi.IntParameter(default=7)
    test_size = luigi.FloatParameter(default=0.2)
    random_state = luigi.IntParameter(default=42)
    use_mlflow = luigi.BoolParameter(default=True)
    experiment_name = luigi.Parameter(default="wine_quality_experiments")
    no_mlflow = luigi.BoolParameter(default=False)

    # Параметры для Random Forest
    rf_n_estimators = luigi.OptionalParameter(default=None)
    rf_max_depth = luigi.OptionalParameter(default=None)

    # Параметры для Boosting
    boosting_n_estimators = luigi.OptionalParameter(default=None)
    boosting_max_depth = luigi.OptionalParameter(default=None)
    boosting_learning_rate = luigi.OptionalParameter(default=None)

    # Параметры для MLP
    mlp_hidden_layer_sizes = luigi.OptionalParameter(default=None)
    mlp_max_iter = luigi.OptionalParameter(default=None)

    def requires(self) -> TrainModel:
        return TrainModel(
            features_path=self.features_path,
            model_path=self.model_path,
            model_type=self.model_type,
            test_size=self.test_size,
            random_state=self.random_state,
            use_mlflow=self.use_mlflow,
            experiment_name=self.experiment_name,
            no_mlflow=self.no_mlflow,
            rf_n_estimators=self.rf_n_estimators,
            rf_max_depth=self.rf_max_depth,
            boosting_n_estimators=self.boosting_n_estimators,
            boosting_max_depth=self.boosting_max_depth,
            boosting_learning_rate=self.boosting_learning_rate,
            mlp_hidden_layer_sizes=self.mlp_hidden_layer_sizes,
            mlp_max_iter=self.mlp_max_iter,
        )


if __name__ == "__main__":
    luigi.run()
