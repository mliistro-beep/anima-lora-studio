"""Runs check_model_file() on a background thread: hashing a DiT takes several seconds."""

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject

from anima_trainer.models.catalog import ExpectedFile
from anima_trainer.models.file_check import (
    CHUNK_SIZE,
    CheckCancelled,
    ModelFileRejected,
    check_model_file,
)
from anima_trainer.models.linked_models import LinkedModel
from anima_trainer.system.background_task import BackgroundTask, Report


class ModelFileChecker(BackgroundTask):
    """progress(checked, size), then finished(LinkedModel), failed(reason) or cancelled."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, parent: QObject | None = None) -> None:
        super().__init__((ModelFileRejected,), CheckCancelled, parent)
        self._chunk_size = chunk_size

    def start(self, path: Path, expected: ExpectedFile) -> None:
        def job(report: Report, is_cancelled: Callable[[], bool]) -> LinkedModel:
            return check_model_file(path, expected, report, is_cancelled, self._chunk_size)

        self._run(job)
