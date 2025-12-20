#!/usr/bin/env python
"""Скрипт для запуска ClearML пайплайна."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from clearml import Task

from wine_quality.clearml_pipeline import PipelineDecorator, wine_quality_pipeline

if __name__ == "__main__":
    # Инициализируем Task с проектом "Wine Quality" для установки проекта по умолчанию
    # Используем тип 'controller', так как пайплайны управляются контроллерами
    # Эта задача будет закрыта после запуска пайплайна
    controller_task = Task.init(
        project_name="Wine Quality",
        task_name="WineQuality",
        task_type="controller",
    )

    PipelineDecorator.run_locally()

    wine_quality_pipeline(
        quality_threshold=7,
        test_size=0.2,
        random_state=42,
        model_type="boosting",
        boosting_n_estimators=100,
        boosting_max_depth=2,
        boosting_learning_rate=0.1,
        experiment_name="Wine Quality - Boosting",
    )

    # Закрываем контроллер-задачу после запуска пайплайна
    if controller_task:
        controller_task.close()
