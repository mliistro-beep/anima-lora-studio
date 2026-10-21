"""Citron's Training tab: basic settings, Configure / Start / Stop, status, progress and log."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from anima_trainer.settings.field_specs import BASIC_FIELDS
from anima_trainer.ui.widgets.log_view import LogView
from anima_trainer.ui.widgets.path_picker import PathPicker
from anima_trainer.ui.widgets.progress_panel import ProgressPanel
from anima_trainer.ui.widgets.settings_form import SettingsForm
from anima_trainer.workflow.training_session import SessionState

CUSTOM_CONFIG_LABEL = "Override Training Config Path (optional, blank uses the last generated)"
FORM_STRETCH, OUTPUT_STRETCH = 2, 3
STATUS_STRETCH, LOG_STRETCH = 1, 2


class TrainingTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.form = SettingsForm(BASIC_FIELDS)
        self.configure_button = QPushButton("Configure Training")
        self.start_button = QPushButton("Start Training")
        self.start_button.setDefault(True)
        self.stop_button = QPushButton("Stop")
        self.custom_config = PathPicker(
            CUSTOM_CONFIG_LABEL,
            "file",
            placeholder="/path/to/custom_training_config.toml",
            file_filter="TOML files (*.toml);;All files (*)",
        )
        self.status = LogView("Configuration Status")
        self.progress = ProgressPanel()
        self.log = LogView("Training Log")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._settings_column())
        splitter.addWidget(self._output_column())
        splitter.setStretchFactor(0, FORM_STRETCH)
        splitter.setStretchFactor(1, OUTPUT_STRETCH)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.set_state(SessionState.IDLE, ready=False)

    def custom_config_path(self) -> str:
        return self.custom_config.path()

    def set_state(self, state: SessionState, ready: bool) -> None:
        """`ready` is False when the Citron install cannot train (see the Settings tab)."""
        idle = state is SessionState.IDLE
        self.configure_button.setEnabled(idle and ready)
        self.start_button.setEnabled(idle and ready)
        self.stop_button.setEnabled(state in (SessionState.DOWNLOADING, SessionState.TRAINING))
        self.form.setEnabled(idle)
        self.custom_config.setEnabled(idle)

    def _settings_column(self) -> QWidget:
        buttons = QHBoxLayout()
        for button in (self.configure_button, self.start_button, self.stop_button):
            buttons.addWidget(button)
        override = QFormLayout()
        override.addRow(CUSTOM_CONFIG_LABEL, self.custom_config)
        override.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        content = QWidget()
        column = QVBoxLayout(content)
        column.addWidget(self.form)
        column.addLayout(override)
        column.addLayout(buttons)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _output_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.status, STATUS_STRETCH)
        layout.addWidget(self.progress)
        layout.addWidget(self.log, LOG_STRETCH)
        return column
