"""Stand-ins for TrainingRunner and ModelDownloader with the same signals."""

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from anima_trainer.training.command_builder import LaunchSpec


class FakeRunner(QObject):
    line_received = Signal(str)
    event_received = Signal(object)
    finished = Signal(object)
    failed_to_start = Signal(str)
    warning = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.started: list[tuple[LaunchSpec, Path]] = []
        self.stopped = 0
        self.shut_down = 0
        self.tail: tuple[str, ...] = ()
        self.start_error: Exception | None = None

    def start(self, spec: LaunchSpec, log_path: Path) -> None:
        if self.start_error:
            raise self.start_error
        self.started.append((spec, log_path))

    def stop(self) -> None:
        self.stopped += 1

    def shutdown(self) -> None:
        self.shut_down += 1

    def log_tail(self) -> tuple[str, ...]:
        return self.tail


class FakeDownloader(QObject):
    progress = Signal(object, object)
    finished = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.started: list[tuple[str, Path]] = []
        self.cancels = 0
        self.shut_down = 0

    def start(self, url: str, destination: Path) -> None:
        self.started.append((url, destination))

    def cancel(self) -> None:
        self.cancels += 1

    def shutdown(self) -> None:
        self.shut_down += 1
