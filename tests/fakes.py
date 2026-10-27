"""Stand-ins for TrainingRunner, ModelDownloader and RuntimeInstaller with the same signals."""

from collections.abc import Iterable
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
        self.expected: list[object] = []
        self.cancels = 0
        self.shut_down = 0

    def start(self, url: str, destination: Path, expected: object = None) -> None:
        self.started.append((url, destination))
        self.expected.append(expected)

    def cancel(self) -> None:
        self.cancels += 1

    def shutdown(self) -> None:
        self.shut_down += 1


class FakeInstaller(QObject):
    step_started = Signal(int, int, object)
    line_received = Signal(str)
    finished = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self.started: list[tuple[object, ...]] = []
        self.cancels = 0
        self.shut_down = 0
        self.running = False

    def start(self, steps: Iterable[object]) -> None:
        self.started.append(tuple(steps))
        self.running = True

    def cancel(self) -> None:
        self.cancels += 1

    def shutdown(self) -> None:
        """Like RuntimeInstaller: a running install reports its end inside shutdown()."""
        self.shut_down += 1
        if self.running:
            self.finish(False, "Installation cancelled.")

    def is_running(self) -> bool:
        return self.running

    def finish(self, ok: bool, message: str) -> None:
        self.running = False
        self.finished.emit(ok, message)
