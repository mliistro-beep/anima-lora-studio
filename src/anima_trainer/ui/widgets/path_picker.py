"""A path text field with a Browse button for a folder or a file."""

from typing import Literal

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget

PickerMode = Literal["folder", "file"]


class PathPicker(QWidget):
    path_changed = Signal(str)  # after Browse or when editing finishes

    def __init__(
        self,
        label: str,
        mode: PickerMode,
        placeholder: str = "",
        file_filter: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._mode = mode
        self._file_filter = file_filter
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setAccessibleName(label)
        self.edit.editingFinished.connect(lambda: self.path_changed.emit(self.path()))
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setAccessibleName(f"Browse for {label}")
        self.browse_button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.browse_button)

    def path(self) -> str:
        return self.edit.text()

    def set_path(self, path: str) -> None:
        self.edit.setText(path)

    def _browse(self) -> None:
        start = self.path().strip()
        if self._mode == "folder":
            chosen = QFileDialog.getExistingDirectory(self, self._label, start)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, self._label, start, self._file_filter)
        if chosen:
            self.set_path(chosen)
            self.path_changed.emit(chosen)
