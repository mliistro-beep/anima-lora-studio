"""Shows one image scaled to the window, with its path."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

DEFAULT_SIZE = (900, 900)


class ImageViewer(QDialog):
    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(path.name)
        self.resize(*DEFAULT_SIZE)
        self.pixmap = QPixmap(str(path))
        self.image = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(1, 1)
        caption = QLabel(str(path))
        caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.image, 1)
        layout.addWidget(caption)
        layout.addWidget(buttons)
        if self.pixmap.isNull():
            self.image.setText(f"Cannot open {path}")

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if not self.pixmap.isNull():
            self.image.setPixmap(
                self.pixmap.scaled(
                    self.image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
