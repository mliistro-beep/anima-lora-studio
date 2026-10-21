"""Settings tab: the Citron folder to use and the state of its model files."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from anima_trainer.models.status import ModelStatus
from anima_trainer.ui.widgets.path_picker import PathPicker
from anima_trainer.ui.widgets.progress_panel import download_text
from anima_trainer.workflow.training_session import SessionState

ROOT_HELP = (
    "Folder of an installed Citron's Anima LoRA Trainer (with its .venv, sd-scripts and "
    "models). Its config.json, configs and logs are shared with Citron."
)
MODEL_COLUMNS = ("Model", "Status", "Path", "")
PATH_COLUMN, ACTION_COLUMN = 2, 3


class SettingsTab(QWidget):
    root_applied = Signal(str)
    download_requested = Signal(str)  # base model name
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.download_buttons: dict[str, QPushButton] = {}
        self._present: frozenset[str] = frozenset()  # base models already downloaded
        self._idle = True
        self._can_download = False
        self.root_picker = PathPicker("Citron folder", "folder")
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(lambda: self.root_applied.emit(self.root_picker.path()))
        self.problems_label = QLabel()
        self.problems_label.setWordWrap(True)
        self.models_table = QTableWidget(0, len(MODEL_COLUMNS))
        self.models_table.setHorizontalHeaderLabels(list(MODEL_COLUMNS))
        self.models_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.models_table.horizontalHeader().setSectionResizeMode(
            PATH_COLUMN, QHeaderView.ResizeMode.Stretch
        )
        self.models_table.verticalHeader().setVisible(False)
        self.download_label = QLabel()
        self.cancel_button = QPushButton("Cancel download")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)

        layout = QVBoxLayout(self)
        layout.addWidget(self._root_group())
        layout.addWidget(self._models_group(), 1)
        self.set_state(SessionState.IDLE, can_download=False)

    def set_root(self, root: str) -> None:
        self.root_picker.set_path(root)

    def set_problems(self, problems: tuple[str, ...]) -> None:
        if problems:
            self.problems_label.setText("\n".join(f"ERROR: {p}" for p in problems))
        else:
            self.problems_label.setText("OK: Citron installation found.")

    def set_models(self, statuses: tuple[ModelStatus, ...]) -> None:
        self.models_table.setRowCount(len(statuses))
        buttons: dict[str, QPushButton] = {}
        for row, status in enumerate(statuses):
            cells = (status.label, "OK" if status.present else "Missing", str(status.path))
            for column, text in enumerate(cells):
                self.models_table.setItem(row, column, QTableWidgetItem(text))
            if status.base_model is not None:
                button = self._download_button(status.label, status.base_model)
                self.models_table.setCellWidget(row, ACTION_COLUMN, button)
                buttons[status.base_model] = button
        self.download_buttons = buttons
        self._present = frozenset(
            s.base_model for s in statuses if s.base_model is not None and s.present
        )
        self._update_buttons()

    def set_state(self, state: SessionState, can_download: bool) -> None:
        """`can_download` is False when the Citron folder does not exist."""
        self._idle = state is SessionState.IDLE
        self._can_download = can_download
        self.root_picker.setEnabled(self._idle)
        self.apply_button.setEnabled(self._idle)
        self.cancel_button.setEnabled(state is SessionState.DOWNLOADING)
        if self._idle:
            self.download_label.clear()
        self._update_buttons()

    def show_download(self, downloaded: int, total: int | None) -> None:
        self.download_label.setText(download_text(downloaded, total))

    def _download_button(self, label: str, base_model: str) -> QPushButton:
        button = QPushButton("Download")
        button.setAccessibleName(f"Download {label}")
        button.clicked.connect(lambda: self.download_requested.emit(base_model))
        return button

    def _update_buttons(self) -> None:
        allowed = self._idle and self._can_download
        for base_model, button in self.download_buttons.items():
            button.setEnabled(allowed and base_model not in self._present)

    def _root_group(self) -> QGroupBox:
        group = QGroupBox("Citron installation")
        layout = QVBoxLayout(group)
        help_label = QLabel(ROOT_HELP)
        help_label.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.root_picker, 1)
        row.addWidget(self.apply_button)
        layout.addWidget(help_label)
        layout.addLayout(row)
        layout.addWidget(self.problems_label)
        return group

    def _models_group(self) -> QGroupBox:
        group = QGroupBox("Models")
        layout = QVBoxLayout(group)
        layout.addWidget(self.models_table, 1)
        row = QHBoxLayout()
        row.addWidget(self.download_label, 1)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        return group
