"""Previews tab: the preview settings used by Configure, and the gallery of the images
sd-scripts makes during training (one row per epoch, one column per prompt)."""

from pydantic import ValidationError
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from anima_trainer.preview.form_fields import PREVIEW_FIELDS, PREVIEW_LABELS
from anima_trainer.preview.gallery_layout import preview_layout
from anima_trainer.preview.sample_files import SampleImage
from anima_trainer.preview.settings import MAX_PROMPTS, PreviewSettings
from anima_trainer.settings.field_specs import describe_validation_error
from anima_trainer.ui.widgets.image_grid import ImageGrid
from anima_trainer.ui.widgets.settings_form import SettingsForm

PROMPTS_HINT = f"One prompt per line, at most {MAX_PROMPTS}. Used by the next Configure Training."
EMPTY_GALLERY = "No previews yet. Turn previews on, Configure and Start the training."
SETTINGS_STRETCH, GALLERY_STRETCH = 2, 3
SETTINGS_WIDTH, GALLERY_WIDTH = 440, 760  # starting split; the user can drag it


class PreviewsTab(QWidget):
    image_activated = Signal(object)  # Path
    refresh_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.form = SettingsForm(PREVIEW_FIELDS)
        self.prompts = QPlainTextEdit()
        self.prompts.setPlaceholderText("masterpiece, best quality, 1girl, ...")
        self.prompts.setAccessibleName("Preview prompts")
        self.gallery = ImageGrid()
        self.gallery_note = QLabel(EMPTY_GALLERY)
        self.refresh_button = QPushButton("Refresh")
        self.gallery.image_activated.connect(self.image_activated.emit)
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        self._prompts: tuple[str, ...] = ()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._settings_column())
        splitter.addWidget(self._gallery_column())
        splitter.setStretchFactor(0, SETTINGS_STRETCH)
        splitter.setStretchFactor(1, GALLERY_STRETCH)
        splitter.setSizes([SETTINGS_WIDTH, GALLERY_WIDTH])
        QVBoxLayout(self).addWidget(splitter)

    def settings(self) -> PreviewSettings | tuple[str, ...]:
        """The settings in the form, or the error lines explaining what is wrong."""
        lines = tuple(self.prompts.toPlainText().splitlines())
        try:
            return PreviewSettings.model_validate({**self.form.values(), "prompts": lines})
        except ValidationError as error:
            return describe_validation_error(error, PREVIEW_LABELS)

    def set_settings(self, settings: PreviewSettings) -> None:
        self.form.set_values(settings)
        self.prompts.setPlainText("\n".join(settings.prompts))
        self._prompts = settings.prompts

    def set_prompt_labels(self, prompts: tuple[str, ...]) -> None:
        """Name the gallery columns after the prompts of the configured run."""
        self._prompts = prompts

    def set_editable(self, editable: bool) -> None:
        """Read-only while training: changes apply at the next Configure anyway."""
        self.form.setEnabled(editable)
        self.prompts.setReadOnly(not editable)

    def show_samples(self, samples: tuple[SampleImage, ...]) -> None:
        layout = preview_layout(samples, self._prompts)
        self.gallery.set_grid(layout.rows, layout.columns, layout.cells)
        self.gallery_note.setVisible(not samples)

    def _settings_column(self) -> QWidget:
        prompts_box = QGroupBox("Preview Prompts")
        prompts_layout = QVBoxLayout(prompts_box)
        hint = QLabel(PROMPTS_HINT)
        hint.setWordWrap(True)
        prompts_layout.addWidget(hint)
        prompts_layout.addWidget(self.prompts)
        content = QWidget()
        column = QVBoxLayout(content)
        column.addWidget(prompts_box)
        column.addWidget(self.form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _gallery_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Previews of the current project (double-click to enlarge)"))
        layout.addWidget(self.gallery_note)
        layout.addWidget(self.gallery, 1)
        layout.addWidget(self.refresh_button, 0, Qt.AlignmentFlag.AlignRight)
        return column
