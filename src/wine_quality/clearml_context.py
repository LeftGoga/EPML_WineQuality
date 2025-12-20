from __future__ import annotations

import logging
import traceback
from typing import Any

logger = logging.getLogger(__name__)

try:
    from clearml import Task

    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    Task = None


def is_clearml_available() -> bool:
    """Проверяет, доступен ли ClearML."""
    return CLEARML_AVAILABLE


class ClearMLTaskContext:
    """Контекстный менеджер для работы с задачами ClearML."""

    def __init__(
        self,
        project_name: str = "Wine Quality",
        task_name: str | None = None,
        task_type: str = "training",
        tags: list[str] | None = None,
        auto_connect_frameworks: bool = True,
        auto_connect_streams: bool = True,
        reuse_last_task_id: str | None = None,
        continue_last_task: bool = False,
    ):
        if not CLEARML_AVAILABLE:
            logger.warning("ClearML не установлен. Установите: pip install clearml")
            self.task = None
            return

        self.project_name = project_name
        self.task_name = task_name
        self.task_type = task_type
        self.tags = tags or []
        self.auto_connect_frameworks = auto_connect_frameworks
        self.auto_connect_streams = auto_connect_streams
        self.reuse_last_task_id = reuse_last_task_id
        self.continue_last_task = continue_last_task
        self.task: Task | None = None
        self._task_id: str | None = None

    def __enter__(self) -> ClearMLTaskContext:
        if not CLEARML_AVAILABLE:
            return self

        try:
            if self.continue_last_task and self.reuse_last_task_id:
                self.task = Task.init(
                    project_name=self.project_name,
                    task_name=self.task_name,
                    task_id=self.reuse_last_task_id,
                    reuse_last_task_id=self.reuse_last_task_id,
                    auto_connect_frameworks=self.auto_connect_frameworks,
                    auto_connect_streams=self.auto_connect_streams,
                )
            elif self.reuse_last_task_id:
                self.task = Task.init(
                    project_name=self.project_name,
                    task_name=self.task_name,
                    task_id=self.reuse_last_task_id,
                    reuse_last_task_id=None,
                    auto_connect_frameworks=self.auto_connect_frameworks,
                    auto_connect_streams=self.auto_connect_streams,
                )
            else:
                self.task = Task.init(
                    project_name=self.project_name,
                    task_name=self.task_name,
                    task_type=self.task_type,
                    auto_connect_frameworks=self.auto_connect_frameworks,
                    auto_connect_streams=self.auto_connect_streams,
                )

            if self.task:
                if self.tags:
                    for tag in self.tags:
                        self.task.add_tags([tag])

                self._task_id = self.task.id
                logger.info(
                    f"✓ ClearML задача создана: {self._task_id} в проекте '{self.project_name}'"
                )
            else:
                logger.warning("ClearML задача не была создана (Task.init вернул None)")

        except Exception as e:
            logger.error(f"Ошибка при создании задачи ClearML: {e}")
            logger.debug(traceback.format_exc())
            self.task = None

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.task:
            try:
                if exc_type is not None:
                    self.task.mark_failed(
                        status_message=str(exc_val) if exc_val else "Unknown error"
                    )
                    logger.warning(f"ClearML задача помечена как failed: {self._task_id}")
                else:
                    self.task.close()
                    logger.info(f"ClearML задача закрыта: {self._task_id}")
            except Exception as e:
                logger.warning(f"Ошибка при закрытии задачи ClearML: {e}")

    @property
    def id(self) -> str | None:
        """Возвращает ID задачи."""
        return self._task_id if self.task else None

    def connect(self, configuration: dict[str, Any]) -> None:
        """Подключает конфигурацию к задаче."""
        if self.task:
            try:
                self.task.connect(configuration)
            except Exception as e:
                logger.warning(f"Ошибка при подключении конфигурации: {e}")

    def set_parameters(self, parameters: dict[str, Any]) -> None:
        """Устанавливает параметры задачи."""
        if self.task:
            try:
                self.task.set_parameters(parameters)
            except Exception as e:
                logger.warning(f"Ошибка при установке параметров: {e}")

    def add_tags(self, tags: list[str]) -> None:
        """Добавляет теги к задаче."""
        if self.task:
            try:
                self.task.add_tags(tags)
            except Exception as e:
                logger.warning(f"Ошибка при добавлении тегов: {e}")
