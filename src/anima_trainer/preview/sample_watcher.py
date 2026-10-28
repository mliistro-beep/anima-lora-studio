"""Follows <output_dir>/sample/ during a training and reports when its previews change.

Polling, not QFileSystemWatcher: sd-scripts creates the folder only at the first preview,
and a watcher cannot watch a folder that does not exist yet.
"""

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from anima_trainer.preview.sample_files import SampleImage, scan_samples

POLL_INTERVAL_MS = 2000
SAMPLE_DIR = "sample"


class SampleWatcher(QObject):
    samples_changed = Signal(object)  # tuple[SampleImage, ...]

    def __init__(self, interval_ms: int = POLL_INTERVAL_MS, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._poll)
        self._folder: Path | None = None
        self._project = ""
        self._since: datetime | None = None
        self._last: tuple[SampleImage, ...] = ()
        self._last_stamps: tuple[tuple[int, int], ...] = ()

    def watch(self, output_dir: Path, project: str, since: datetime | None) -> None:
        """Follow the previews of `project` newer than `since`; reports the current ones first."""
        self._folder, self._project, self._since = output_dir / SAMPLE_DIR, project, since
        self._last, self._last_stamps = (), ()
        self._timer.start()
        self._poll()

    def stop(self) -> None:
        self._timer.stop()

    def is_watching(self) -> bool:
        return self._timer.isActive()

    def refresh(self) -> tuple[SampleImage, ...]:
        """Read the folder now (also when not polling) and report it if it changed."""
        self._poll()
        return self._last

    def _poll(self) -> None:
        if self._folder is None:
            return
        samples = scan_samples(self._folder, self._project, self._since)
        stamps = tuple(_file_stamp(sample.path) for sample in samples)
        # The stamps catch a preview seen half written: sd-scripts saves the PNG in place.
        if samples != self._last or stamps != self._last_stamps:
            self._last, self._last_stamps = samples, stamps
            self.samples_changed.emit(samples)


def _file_stamp(path: Path) -> tuple[int, int]:
    """Size and modification time; (-1, -1) when the file is gone."""
    try:
        status = path.stat()
    except OSError:
        return -1, -1
    return status.st_size, status.st_mtime_ns
