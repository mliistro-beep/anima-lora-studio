"""Links a model file the user already has: checks it in the background, then saves it.

A file is linked only when its size and SHA-256 match the catalog; the caller then rebuilds
its RuntimePaths from links() so the TOML, Configure and Start use the file in place.
"""

from collections.abc import Mapping
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Signal

from anima_trainer.models.catalog import EXPECTED_FILES, ExpectedFile, model_label
from anima_trainer.models.check_worker import ModelFileChecker
from anima_trainer.models.linked_models import LinkedModel
from anima_trainer.models.linked_store import load_links, remove_link, save_link


class LinkSession(QObject):
    progress = Signal(object, object)  # checked bytes, file size
    finished = Signal(bool, str)  # linked, message for the user

    def __init__(
        self,
        store: QSettings,
        checker: ModelFileChecker,
        expected: Mapping[str, ExpectedFile] = EXPECTED_FILES,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._checker = checker
        self._expected = expected
        self._checking: tuple[str, Path] | None = None
        checker.progress.connect(self.progress.emit)
        checker.finished.connect(self._on_checked)
        checker.failed.connect(self._on_rejected)
        checker.cancelled.connect(self._on_cancelled)

    def is_running(self) -> bool:
        return self._checker.is_running()

    def links(self) -> Mapping[str, LinkedModel]:
        return load_links(self._store)

    def check(self, key: str, path: Path) -> str | None:
        """Start checking `path` for model `key`; returns why it cannot start, else None."""
        if key not in self._expected:
            return f"Unknown model: {key}"
        if self._checker.is_running():
            return "A file check is already running"
        self._checking = (key, path)
        self._checker.start(path, self._expected[key])
        return None

    def unlink(self, key: str) -> str | None:
        """Forget the file linked for `key`; returns the error when it cannot be saved."""
        try:
            remove_link(self._store, key)
        except OSError as error:
            return str(error)
        return None

    def cancel(self) -> None:
        self._checker.cancel()

    def shutdown(self) -> None:
        self._checker.shutdown()

    def _on_checked(self, link: object) -> None:
        key, path = self._take()
        if not isinstance(link, LinkedModel):
            self.finished.emit(False, f"ERROR: Unexpected check result for {path}")
            return
        try:
            save_link(self._store, key, link)
        except OSError as error:
            self.finished.emit(False, f"ERROR: {error}")
            return
        message = f"OK: {model_label(key)} file accepted (SHA-256 matches): {link.path}"
        self.finished.emit(True, message)

    def _on_rejected(self, reason: str) -> None:
        key, path = self._take()
        self.finished.emit(
            False, f"ERROR: {path} is not a compatible {model_label(key)} file: {reason}"
        )

    def _on_cancelled(self) -> None:
        self._take()
        self.finished.emit(False, "INFO: File check cancelled.")

    def _take(self) -> tuple[str, Path]:
        checking, self._checking = self._checking, None
        return checking or ("", Path())
