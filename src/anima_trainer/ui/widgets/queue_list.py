"""The Generator's queue: numbered entries (preview.queue_text), at most MAX_QUEUE.

Add current asks the tab for the form as an entry (add_requested), since only the tab can read
it; Remove takes the selected entry, Clear all of them.
"""

from collections.abc import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QListWidget, QPushButton, QWidget

from anima_trainer.preview.generator_settings import MAX_QUEUE, GenerationEntry
from anima_trainer.preview.queue_text import entry_summary


class QueueList(QWidget):
    add_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.list = QListWidget()
        self.list.setAccessibleName("Queue")
        self.add_button = QPushButton("Add current")
        self.add_button.setToolTip(f"Add the form as it is now (at most {MAX_QUEUE} entries).")
        self.remove_button = QPushButton("Remove")
        self.clear_button = QPushButton("Clear")
        self._entries: tuple[GenerationEntry, ...] = ()
        self._editable = True
        self.add_button.clicked.connect(self.add_requested.emit)
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button.clicked.connect(lambda: self.set_entries(()))
        self.list.currentRowChanged.connect(lambda _row: self._update_buttons())
        self._update_buttons()

    def entries(self) -> tuple[GenerationEntry, ...]:
        return self._entries

    def set_entries(self, entries: Iterable[GenerationEntry]) -> None:
        self._entries = tuple(entries)[:MAX_QUEUE]
        self.list.clear()
        self.list.addItems(
            [entry_summary(number, entry) for number, entry in enumerate(self._entries, 1)]
        )
        self._update_buttons()

    def add(self, entry: GenerationEntry) -> bool:
        """False when the queue is full."""
        if len(self._entries) >= MAX_QUEUE:
            return False
        self.set_entries((*self._entries, entry))
        return True

    def set_editable(self, editable: bool) -> None:
        self._editable = editable
        self._update_buttons()

    def _remove_selected(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self._entries):
            self.set_entries(self._entries[:row] + self._entries[row + 1 :])

    def _update_buttons(self) -> None:
        self.add_button.setEnabled(self._editable and len(self._entries) < MAX_QUEUE)
        self.remove_button.setEnabled(self._editable and self.list.currentRow() >= 0)
        self.clear_button.setEnabled(self._editable and bool(self._entries))
