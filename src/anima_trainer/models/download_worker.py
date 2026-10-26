"""Runs download_file() on a background thread and reports through Qt signals."""

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject

from anima_trainer.models.downloader import (
    CHUNK_SIZE,
    DownloadCancelled,
    DownloadError,
    DownloadProgress,
    download_file,
)
from anima_trainer.system.background_task import BackgroundTask, Report


class ModelDownloader(BackgroundTask):
    """progress(downloaded, total or None), then finished(path as str), failed or cancelled."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, parent: QObject | None = None) -> None:
        super().__init__((DownloadError,), DownloadCancelled, parent)
        self._chunk_size = chunk_size

    def start(self, url: str, destination: Path) -> None:
        def job(report: Report, is_cancelled: Callable[[], bool]) -> str:
            def on_progress(progress: DownloadProgress) -> None:
                report(progress.downloaded, progress.total)

            path = download_file(
                url,
                destination,
                on_progress=on_progress,
                is_cancelled=is_cancelled,
                chunk_size=self._chunk_size,
            )
            return str(path)

        self._run(job)
