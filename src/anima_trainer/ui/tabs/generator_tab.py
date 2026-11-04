"""Generator tab: one image with a checkpoint of the current project (or the base model
alone), or the same prompt and seed on every checkpoint to compare the epochs."""

from pathlib import Path

from pydantic import ValidationError
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from anima_trainer.preview.checkpoints import Checkpoint
from anima_trainer.preview.form_fields import GENERATOR_FIELDS, GENERATOR_LABELS
from anima_trainer.preview.generator_settings import GeneratorSettings
from anima_trainer.settings.field_specs import describe_validation_error
from anima_trainer.ui.widgets.log_view import LogView
from anima_trainer.ui.widgets.settings_form import SettingsForm

NO_LORA = "No LoRA (base model only)"
RESULT_SIDE = 512
SETTINGS_STRETCH, OUTPUT_STRETCH = 2, 3


class GeneratorTab(QWidget):
    generate_requested = Signal()
    compare_requested = Signal()
    stop_requested = Signal()
    refresh_requested = Signal()
    image_activated = Signal(object)  # Path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.checkpoint = QComboBox()
        self.checkpoint.setAccessibleName("Checkpoint")
        self.refresh_button = QPushButton("Refresh")
        self.prompt = QPlainTextEdit()
        self.prompt.setAccessibleName("Prompt")
        self.prompt.setPlaceholderText("masterpiece, best quality, 1girl, ...")
        self.form = SettingsForm(GENERATOR_FIELDS)
        self.generate_button = QPushButton("Generate")
        self.generate_button.setDefault(True)
        self.compare_button = QPushButton("Compare Epochs")
        self.compare_button.setToolTip("Same prompt and seed on every checkpoint of the project.")
        self.stop_button = QPushButton("Stop")
        self.state_note = QLabel()
        self.state_note.setWordWrap(True)
        self.progress = QProgressBar()
        self.result = QLabel("No image yet.", alignment=Qt.AlignmentFlag.AlignCenter)
        self.result.setMinimumSize(RESULT_SIDE // 2, RESULT_SIDE // 2)
        self.open_button = QPushButton("Enlarge")
        self.log = LogView("Generator Log")
        self._checkpoints: tuple[Checkpoint, ...] = ()
        self._image: Path | None = None
        self._connect()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._settings_column())
        splitter.addWidget(self._output_column())
        splitter.setStretchFactor(0, SETTINGS_STRETCH)
        splitter.setStretchFactor(1, OUTPUT_STRETCH)
        QVBoxLayout(self).addWidget(splitter)
        self.set_running(False, can_generate=False, note="")

    def settings(self) -> GeneratorSettings | tuple[str, ...]:
        values = {**self.form.values(), "prompt": self.prompt.toPlainText()}
        try:
            return GeneratorSettings.model_validate(values)
        except ValidationError as error:
            return describe_validation_error(error, GENERATOR_LABELS)

    def set_settings(self, settings: GeneratorSettings) -> None:
        self.form.set_values(settings)
        self.prompt.setPlainText(settings.prompt)

    def set_checkpoints(self, checkpoints: tuple[Checkpoint, ...]) -> None:
        """Keeps the chosen checkpoint when it is still there (by default the newest)."""
        current = self.selected_checkpoint()
        self._checkpoints = checkpoints
        self.checkpoint.clear()
        self.checkpoint.addItems([NO_LORA, *(checkpoint.label for checkpoint in checkpoints)])
        index = checkpoints.index(current) + 1 if current in checkpoints else len(checkpoints)
        self.checkpoint.setCurrentIndex(index)
        self.compare_button.setEnabled(self.compare_button.isEnabled() and bool(checkpoints))

    def checkpoints(self) -> tuple[Checkpoint, ...]:
        return self._checkpoints

    def selected_checkpoint(self) -> Checkpoint | None:
        index = self.checkpoint.currentIndex()
        return self._checkpoints[index - 1] if 0 < index <= len(self._checkpoints) else None

    def set_running(self, running: bool, can_generate: bool, note: str) -> None:
        """`can_generate` is False when models or the runtime are missing; `note` says why
        or, during a training, that the generator shares the GPU."""
        idle_ok = not running and can_generate
        self.generate_button.setEnabled(idle_ok)
        self.compare_button.setEnabled(idle_ok and bool(self._checkpoints))
        self.stop_button.setEnabled(running)
        self.form.setEnabled(not running)
        self.prompt.setReadOnly(running)
        self.checkpoint.setEnabled(not running)
        self.state_note.setText(note)
        self.state_note.setVisible(bool(note))
        if running:
            self.progress.setValue(0)

    def show_step(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)

    def show_image(self, path: Path) -> None:
        self._image = path
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.result.setText(f"Cannot open {path}")
            return
        side = RESULT_SIDE
        self.result.setPixmap(
            pixmap.scaled(
                side,
                side,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.result.setToolTip(str(path))
        self.open_button.setEnabled(True)

    def _connect(self) -> None:
        self.generate_button.clicked.connect(self.generate_requested.emit)
        self.compare_button.clicked.connect(self.compare_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        self.open_button.clicked.connect(self._open_image)
        self.open_button.setEnabled(False)

    def _open_image(self) -> None:
        if self._image is not None:
            self.image_activated.emit(self._image)

    def _settings_column(self) -> QWidget:
        lora = QGroupBox("LoRA")
        lora_row = QHBoxLayout()
        lora_row.addWidget(self.checkpoint, 1)
        lora_row.addWidget(self.refresh_button)
        lora_form = QFormLayout(lora)
        lora_form.addRow("Checkpoint", lora_row)
        prompt_box = QGroupBox("Prompt")
        QVBoxLayout(prompt_box).addWidget(self.prompt)
        buttons = QHBoxLayout()
        for button in (self.generate_button, self.compare_button, self.stop_button):
            buttons.addWidget(button)
        content = QWidget()
        column = QVBoxLayout(content)
        for widget in (lora, prompt_box, self.form):
            column.addWidget(widget)
        column.addLayout(buttons)
        column.addWidget(self.state_note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _output_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.progress)
        layout.addWidget(self.result, 2)
        layout.addWidget(self.open_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.log, 1)
        return column
