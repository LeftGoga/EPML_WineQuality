#!/usr/bin/env python
"""Скрипт для запуска Luigi пайплайна с Hydra конфигурациями."""

from __future__ import annotations

import sys
from pathlib import Path

# Добавление пути к модулям
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wine_quality.orchestrator import run_pipeline_with_hydra

if __name__ == "__main__":
    run_pipeline_with_hydra()
