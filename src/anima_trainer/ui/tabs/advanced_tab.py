"""Citron's Advanced Settings tab, applied when Configure Training is clicked."""

from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from anima_trainer.settings.field_specs import ADVANCED_FIELDS
from anima_trainer.ui.widgets.settings_form import SettingsForm

NOTE = (
    "These settings are applied when you click <b>Configure Training</b>. "
    "Defaults match the original notebook."
)


class AdvancedTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.form = SettingsForm(ADVANCED_FIELDS)
        content = QWidget()
        column = QVBoxLayout(content)
        note = QLabel(NOTE)
        note.setWordWrap(True)
        column.addWidget(note)
        column.addWidget(self.form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
