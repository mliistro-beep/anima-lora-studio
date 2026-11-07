"""The Generator's list of LoRA files: on/off, file name, trigger and strength, at most
MAX_LORAS. Only the trigger can be typed in; an added file gets the trigger found in it.

The Add and Remove buttons belong to the table but are laid out by the tab.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

from PySide6.QtCore import QLocale, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from anima_trainer.preview.generator_settings import (
    MAX_LORAS,
    MAX_STRENGTH,
    MIN_STRENGTH,
    LoraEntry,
)

ON_COLUMN, FILE_COLUMN, TRIGGER_COLUMN, STRENGTH_COLUMN = 0, 1, 2, 3
HEADERS = ("On", "File", "Trigger", "Strength")
STRENGTH_STEP = 0.05
STRENGTH_DECIMALS = 2
LORA_FILE_FILTER = "LoRA files (*.safetensors);;All files (*)"


def _no_trigger(_path: str) -> str:
    return ""


class LoraTable(QTableWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        trigger_for: Callable[[str], str] = _no_trigger,
    ) -> None:
        super().__init__(0, len(HEADERS), parent)
        self._trigger_for = trigger_for
        self.setHorizontalHeaderLabels(HEADERS)
        self.setAccessibleName("LoRAs")
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # Only items with ItemIsEditable (the trigger) can be edited.
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(FILE_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(TRIGGER_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(ON_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(STRENGTH_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        self.add_button = QPushButton("Add...")
        self.add_button.setToolTip(f"Add LoRA files (at most {MAX_LORAS}).")
        self.remove_button = QPushButton("Remove")
        self.add_button.clicked.connect(self._browse)
        self.remove_button.clicked.connect(self._remove_selected)
        self._editable = True

    def entries(self) -> tuple[LoraEntry, ...]:
        return tuple(self._entry(row) for row in range(self.rowCount()))

    def set_entries(self, entries: Iterable[LoraEntry]) -> None:
        self.setRowCount(0)
        for entry in entries:
            self._append(entry)
        self._update_buttons()

    def add_files(self, files: Iterable[str]) -> int:
        """Adds new files with strength 1; returns how many did not fit."""
        known = {entry.path for entry in self.entries()}
        skipped = 0
        for file in files:
            if file in known:
                continue
            if self.rowCount() >= MAX_LORAS:
                skipped += 1
                continue
            self._append(LoraEntry(path=file, trigger=self._trigger_for(file)))
            known.add(file)
        self._update_buttons()
        return skipped

    def strength_box(self, row: int) -> QDoubleSpinBox:
        box = self.cellWidget(row, STRENGTH_COLUMN)
        if not isinstance(box, QDoubleSpinBox):
            raise LookupError(f"row {row} has no strength box")
        return box

    def set_editable(self, editable: bool) -> None:
        self._editable = editable
        for row in range(self.rowCount()):
            self.strength_box(row).setEnabled(editable)
            switch = self.item(row, ON_COLUMN)
            if switch is not None:
                switch.setFlags(_switch_flags(editable))
            trigger = self.item(row, TRIGGER_COLUMN)
            if trigger is not None:
                trigger.setFlags(_trigger_flags(editable))
        self._update_buttons()

    def _append(self, entry: LoraEntry) -> None:
        row = self.rowCount()
        self.insertRow(row)
        switch = QTableWidgetItem()
        switch.setFlags(_switch_flags(self._editable))
        switch.setCheckState(Qt.CheckState.Checked if entry.enabled else Qt.CheckState.Unchecked)
        self.setItem(row, ON_COLUMN, switch)
        name = QTableWidgetItem(Path(entry.path).name)
        name.setToolTip(entry.path)
        name.setData(Qt.ItemDataRole.UserRole, entry.path)
        name.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        self.setItem(row, FILE_COLUMN, name)
        trigger = QTableWidgetItem(entry.trigger)
        trigger.setFlags(_trigger_flags(self._editable))
        trigger.setToolTip("Added to the prompt while this LoRA is on.")
        self.setItem(row, TRIGGER_COLUMN, trigger)
        box = QDoubleSpinBox()
        box.setLocale(QLocale.c())  # a dot, like the other decimal fields of the app
        box.setRange(MIN_STRENGTH, MAX_STRENGTH)
        box.setSingleStep(STRENGTH_STEP)
        box.setDecimals(STRENGTH_DECIMALS)
        box.setValue(entry.strength)
        box.setAccessibleName(f"Strength of {Path(entry.path).name}")
        box.setEnabled(self._editable)
        self.setCellWidget(row, STRENGTH_COLUMN, box)

    def _entry(self, row: int) -> LoraEntry:
        switch, name = self.item(row, ON_COLUMN), self.item(row, FILE_COLUMN)
        trigger = self.item(row, TRIGGER_COLUMN)
        if switch is None or name is None or trigger is None:
            raise LookupError(f"row {row} is incomplete")
        return LoraEntry(
            path=str(name.data(Qt.ItemDataRole.UserRole)),
            strength=round(self.strength_box(row).value(), STRENGTH_DECIMALS),
            enabled=switch.checkState() == Qt.CheckState.Checked,
            trigger=trigger.text(),
        )

    def _browse(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add LoRA files", "", LORA_FILE_FILTER)
        self.add_files(files)

    def _remove_selected(self) -> None:
        for row in sorted({index.row() for index in self.selectedIndexes()}, reverse=True):
            self.removeRow(row)
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.add_button.setEnabled(self._editable and self.rowCount() < MAX_LORAS)
        self.remove_button.setEnabled(self._editable and self.rowCount() > 0)


def _trigger_flags(editable: bool) -> Qt.ItemFlag:
    flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
    return flags | Qt.ItemFlag.ItemIsEditable if editable else flags


def _switch_flags(editable: bool) -> Qt.ItemFlag:
    flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
    return flags | Qt.ItemFlag.ItemIsUserCheckable if editable else flags
