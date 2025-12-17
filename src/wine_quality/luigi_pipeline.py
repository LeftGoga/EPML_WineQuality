from __future__ import annotations

import ast
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import luigi
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from .config import BASE_DIR
from .data import create_target, get_features_and_target, load_data, split_data
from .features import engineer_features, scale_features
from .model import ModelType, run_experiment, save_model

load_dotenv()

logger = logging.getLogger(__name__)


class PrepareData(luigi.Task):
    """Задача для подготовки исходных данных."""

    output_path = luigi.Parameter(default=str(BASE_DIR / "data" / "winequality-red.csv"))
    config_path = luigi.OptionalParameter(default=None)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.output_path)

    def run(self) -> None:
        start_time = time.time()
        logger.info(f"Начало подготовки данных: {self.output_path}")
        try:
            df = load_data()
            df.to_csv(self.output_path, index=False)
            elapsed = time.time() - start_time
            logger.info(f"Данные подготовлены за {elapsed:.2f} секунд")
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных: {e}")
            raise


class GenerateFeatures(luigi.Task):
    """Задача для генерации признаков из исходных данных."""

    input_path = luigi.Parameter(default=str(BASE_DIR / "data" / "winequality-red.csv"))
    output_path = luigi.Parameter(default=str(BASE_DIR / "data" / "features.csv"))
    quality_threshold = luigi.IntParameter(default=7)
    config_path = luigi.OptionalParameter(default=None)

    def requires(self) -> PrepareData:
        return PrepareData(output_path=self.input_path, config_path=self.config_path)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.output_path)

    def run(self) -> None:
        start_time = time.time()
        logger.info(f"Начало генерации признаков: {self.output_path}")
        try:
            df = load_data()
            df = create_target(df, threshold=self.quality_threshold)
            df = engineer_features(df)
            df.to_csv(self.output_path, index=False)
            elapsed = time.time() - start_time
            logger.info(f"Признаки сгенерированы за {elapsed:.2f} секунд")
        except Exception as e:
            logger.error(f"Ошибка при генерации признаков: {e}")
            raise


class TrainModel(luigi.Task):
    """Задача для обучения модели с поддержкой конфигураций Hydra."""

    features_path = luigi.Parameter(default=str(BASE_DIR / "data" / "features.csv"))
    model_path = luigi.Parameter(default=str(BASE_DIR / "models" / "wine_rf.pkl"))
    model_type = luigi.Parameter(default="random_forest")
    test_size = luigi.FloatParameter(default=0.2)
    random_state = luigi.IntParameter(default=42)
    use_mlflow = luigi.BoolParameter(default=True)
    experiment_name = luigi.Parameter(default="wine_quality_experiments")
    no_mlflow = luigi.BoolParameter(default=False)
    rf_n_estimators = luigi.OptionalParameter(default=None)
    rf_max_depth = luigi.OptionalParameter(default=None)
    boosting_n_estimators = luigi.OptionalParameter(default=None)
    boosting_max_depth = luigi.OptionalParameter(default=None)
    boosting_learning_rate = luigi.OptionalParameter(default=None)
    mlp_hidden_layer_sizes = luigi.OptionalParameter(default=None)
    mlp_max_iter = luigi.OptionalParameter(default=None)
    config_path = luigi.OptionalParameter(default=None)
    hydra_config = luigi.OptionalParameter(default=None)

    def requires(self) -> GenerateFeatures:
        return GenerateFeatures(output_path=self.features_path, config_path=self.config_path)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.model_path)

    def _load_hydra_config(self) -> DictConfig | None:
        """Загружает конфигурацию Hydra, если она предоставлена."""
        if not self.hydra_config:
            return None
        try:
            if isinstance(self.hydra_config, str):
                config_dict = json.loads(self.hydra_config)
            else:
                config_dict = self.hydra_config
            return OmegaConf.create(config_dict)
        except Exception as e:
            logger.warning(f"Не удалось загрузить конфигурацию Hydra: {e}")
            return None

    def _parse_int_parameter(self, value: str | None) -> int | None:
        if not value:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _parse_float_parameter(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_mlp_hidden_layers(self, value: str | None) -> tuple[int, ...] | None:
        if not value:
            return None
        try:
            result = ast.literal_eval(value)
            if isinstance(result, tuple) and all(isinstance(x, int) for x in result):
                return cast(tuple[int, ...], result)
            if isinstance(result, list) and all(isinstance(x, int) for x in result):
                return cast(tuple[int, ...], tuple(result))
            return (100, 50)
        except (ValueError, SyntaxError):
            return (100, 50)

    def _prepare_data(self) -> tuple:
        df = load_data()
        df = create_target(df, threshold=7)
        df = engineer_features(df)

        X, y = get_features_and_target(df)
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        X_train_sc, X_test_sc, _scaler = scale_features(X_train, X_test)
        return X_train_sc, X_test_sc, y_train, y_test

    def _parse_model_parameters(self, hydra_cfg: DictConfig | None = None) -> dict:
        """Парсит параметры модели, приоритет у Hydra конфигурации."""
        params = {
            "rf_n_estimators": self._parse_int_parameter(self.rf_n_estimators),
            "rf_max_depth": self._parse_int_parameter(self.rf_max_depth),
            "boosting_n_estimators": self._parse_int_parameter(self.boosting_n_estimators),
            "boosting_max_depth": self._parse_int_parameter(self.boosting_max_depth),
            "boosting_learning_rate": self._parse_float_parameter(self.boosting_learning_rate),
            "mlp_hidden_layer_sizes": self._parse_mlp_hidden_layers(self.mlp_hidden_layer_sizes),
            "mlp_max_iter": self._parse_int_parameter(self.mlp_max_iter),
        }

        # Если есть Hydra конфигурация, используем её значения
        if hydra_cfg:
            model_cfg = OmegaConf.select(hydra_cfg, "model", default=None)
            if model_cfg:
                if model_cfg.get("n_estimators") is not None:
                    if self.model_type == "random_forest":
                        params["rf_n_estimators"] = model_cfg.get("n_estimators")
                    elif self.model_type == "boosting":
                        params["boosting_n_estimators"] = model_cfg.get("n_estimators")

                if model_cfg.get("max_depth") is not None:
                    if self.model_type == "random_forest":
                        params["rf_max_depth"] = model_cfg.get("max_depth")
                    elif self.model_type == "boosting":
                        params["boosting_max_depth"] = model_cfg.get("max_depth")

                if model_cfg.get("learning_rate") is not None:
                    params["boosting_learning_rate"] = model_cfg.get("learning_rate")

                if model_cfg.get("hidden_layer_sizes") is not None:
                    hidden_sizes = model_cfg.get("hidden_layer_sizes")
                    if isinstance(hidden_sizes, list):
                        params["mlp_hidden_layer_sizes"] = tuple(hidden_sizes)

                if model_cfg.get("max_iter") is not None:
                    params["mlp_max_iter"] = model_cfg.get("max_iter")

        return params

    def run(self) -> None:
        start_time = time.time()
        logger.info(f"Начало обучения модели: {self.model_type}")
        try:
            hydra_cfg = self._load_hydra_config()
            X_train_sc, X_test_sc, y_train, y_test = self._prepare_data()
            model_type = ModelType(self.model_type.lower())
            params = self._parse_model_parameters(hydra_cfg)

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

            # Сохранение метрик для мониторинга
            metrics = result.get("metrics", {})
            metrics_path = (
                Path(self.model_path).parent / f"{Path(self.model_path).stem}_metrics.json"
            )
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "accuracy": float(metrics.get("accuracy", 0.0)),
                        "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
                        "model_type": self.model_type,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            elapsed = time.time() - start_time
            logger.info(
                f"Модель обучена за {elapsed:.2f} секунд. "
                f"Accuracy: {metrics.get('accuracy', 0.0):.4f}"
            )
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            raise


class EvaluateModel(luigi.Task):
    """Задача для оценки модели и генерации отчетов."""

    model_path = luigi.Parameter(default=str(BASE_DIR / "models" / "wine_rf.pkl"))
    report_path = luigi.Parameter(default=str(BASE_DIR / "outputs" / "evaluation_report.json"))

    def requires(self) -> TrainModel:
        return TrainModel(model_path=self.model_path)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.report_path)

    def run(self) -> None:
        start_time = time.time()
        logger.info(f"Начало оценки модели: {self.model_path}")
        try:
            # Загрузка метрик из предыдущего этапа
            metrics_path = (
                Path(self.model_path).parent / f"{Path(self.model_path).stem}_metrics.json"
            )
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
            else:
                metrics = {}

            # Создание отчета
            report = {
                "model_path": self.model_path,
                "metrics": metrics,
                "evaluation_timestamp": datetime.now().isoformat(),
                "status": "completed",
            }

            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=2)

            elapsed = time.time() - start_time
            logger.info(f"Оценка завершена за {elapsed:.2f} секунд")
        except Exception as e:
            logger.error(f"Ошибка при оценке модели: {e}")
            raise


class WineQualityPipeline(luigi.WrapperTask):
    """Главный пайплайн, объединяющий все задачи."""

    features_path = luigi.Parameter(default=str(BASE_DIR / "data" / "features.csv"))
    model_path = luigi.Parameter(default=str(BASE_DIR / "models" / "wine_rf.pkl"))
    model_type = luigi.Parameter(default="random_forest")
    quality_threshold = luigi.IntParameter(default=7)
    test_size = luigi.FloatParameter(default=0.2)
    random_state = luigi.IntParameter(default=42)
    use_mlflow = luigi.BoolParameter(default=True)
    experiment_name = luigi.Parameter(default="wine_quality_experiments")
    no_mlflow = luigi.BoolParameter(default=False)
    rf_n_estimators = luigi.OptionalParameter(default=None)
    rf_max_depth = luigi.OptionalParameter(default=None)
    boosting_n_estimators = luigi.OptionalParameter(default=None)
    boosting_max_depth = luigi.OptionalParameter(default=None)
    boosting_learning_rate = luigi.OptionalParameter(default=None)
    mlp_hidden_layer_sizes = luigi.OptionalParameter(default=None)
    mlp_max_iter = luigi.OptionalParameter(default=None)
    config_path = luigi.OptionalParameter(default=None)
    hydra_config = luigi.OptionalParameter(default=None)
    run_evaluation = luigi.BoolParameter(default=True)

    def requires(self) -> list[luigi.Task]:
        tasks = [
            TrainModel(
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
                config_path=self.config_path,
                hydra_config=self.hydra_config,
            )
        ]
        if self.run_evaluation:
            report_path = str(
                Path(self.model_path).parent / f"{Path(self.model_path).stem}_report.json"
            )
            tasks.append(
                EvaluateModel(
                    model_path=self.model_path,
                    report_path=report_path,
                )
            )
        return tasks


if __name__ == "__main__":
    luigi.run()
