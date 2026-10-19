"""Runs download_file() on a background QThread and reports through Qt signals."""

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from anima_trainer.models.downloader import (
    CHUNK_SIZE,
    DownloadCancelled,
    DownloadError,
    DownloadProgress,
    download_file,
)


class _DownloadThread(QThread):
    # object instead of int: model files exceed the 32-bit range of int signals.
    progress = Signal(object, object)
    succeeded = Signal(str)
    errored = Signal(str)
    was_cancelled = Signal()

    def __init__(self, url: str, destination: Path, chunk_size: int) -> None:
        super().__init__()
        self._url = url
        self._destination = destination
        self._chunk_size = chunk_size
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            path = download_file(
                self._url,
                self._destination,
                on_progress=self._report,
                is_cancelled=self._cancel.is_set,
                chunk_size=self._chunk_size,
            )
        except DownloadCancelled:
            self.was_cancelled.emit()
        except DownloadError as error:
            self.errored.emit(str(error))
        else:
            self.succeeded.emit(str(path))

    def _report(self, progress: DownloadProgress) -> None:
        self.progress.emit(progress.downloaded, progress.total)


class ModelDownloader(QObject):
    """progress(downloaded, total or None), then exactly one of finished/failed/cancelled."""

    progress = Signal(object, object)
    finished = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, chunk_size: int = CHUNK_SIZE, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._chunk_size = chunk_size
        self._thread: _DownloadThread | None = None

    def is_running(self) -> bool:
        return self._thread is not None

    def start(self, url: str, destination: Path) -> None:
        if self._thread is not None:
            raise RuntimeError("A download is already running")
        thread = _DownloadThread(url, destination, self._chunk_size)
        thread.progress.connect(self.progress)
        thread.succeeded.connect(self.finished)
        thread.errored.connect(self.failed)
        thread.was_cancelled.connect(self.cancelled)
        thread.finished.connect(self._on_thread_done)
        self._thread = thread
        thread.start()

    def cancel(self) -> None:
        if self._thread is not None:
            self._thread.cancel()

    def _on_thread_done(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
