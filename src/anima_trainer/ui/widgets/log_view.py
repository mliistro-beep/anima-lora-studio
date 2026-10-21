"""Read-only log text with a title, a Copy button and autoscroll.

Consecutive tqdm "steps:" redraws replace each other, so a run shows one live progress
line instead of thousands; the log file on disk still has every line.
"""

from collections.abc import Iterable

from PySide6.QtGui import QFontDatabase, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

MAX_LOG_LINES = 5000
PROGRESS_PREFIX = "steps:"


def is_redraw(previous: str | None, line: str) -> bool:
    return (
        previous is not None
        and previous.startswith(PROGRESS_PREFIX)
        and line.startswith(PROGRESS_PREFIX)
    )


class LogView(QWidget):
    def __init__(
        self, title: str, max_lines: int = MAX_LOG_LINES, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._last_line: str | None = None
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumBlockCount(max_lines)
        self.text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text_edit.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.text_edit.setAccessibleName(title)
        self.copy_button = QPushButton("Copy")
        self.copy_button.setAccessibleName(f"Copy {title}")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>{title}</b>"))
        header.addStretch(1)
        header.addWidget(self.copy_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.text_edit, 1)

    def text(self) -> str:
        return self.text_edit.toPlainText()

    def clear(self) -> None:
        self.text_edit.clear()
        self._last_line = None

    def set_lines(self, lines: Iterable[str]) -> None:
        self.clear()
        for line in lines:
            self.append_line(line)

    def append_line(self, line: str) -> None:
        scrollbar = self.text_edit.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum()
        if is_redraw(self._last_line, line):
            self._replace_last_line(line)
        else:
            self.text_edit.appendPlainText(line)
        self._last_line = line
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def copy_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.text())

    def _replace_last_line(self, line: str) -> None:
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(line)
