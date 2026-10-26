"""Epoch, step bar, loss, elapsed/remaining time and last checkpoint of a run.

Also shows model download progress, in permille: byte counts overflow QProgressBar's int.
"""

from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QWidget

from anima_trainer.training.progress import ProgressState, format_duration

PERMILLE = 1000
BYTES_PER_GB = 1024**3
UNKNOWN = "-"


def _format_gb(size: int) -> str:
    return f"{size / BYTES_PER_GB:.2f} GB"


def download_text(downloaded: int, total: int | None) -> str:
    if total:
        return f"Downloading model: {_format_gb(downloaded)} / {_format_gb(total)}"
    return f"Downloading model: {_format_gb(downloaded)}"


def check_text(checked: int, total: int) -> str:
    return f"Checking model file (SHA-256): {_format_gb(checked)} / {_format_gb(total)}"


class ProgressPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.epoch_label = QLabel()
        self.bar = QProgressBar()
        self.bar.setAccessibleName("Training progress")
        self.loss_label = QLabel()
        self.time_label = QLabel()
        self.checkpoint_label = QLabel()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.epoch_label, 0, 0)
        layout.addWidget(self.loss_label, 0, 1)
        layout.addWidget(self.bar, 1, 0, 1, 2)
        layout.addWidget(self.time_label, 2, 0)
        layout.addWidget(self.checkpoint_label, 2, 1)
        self.reset()

    def reset(self) -> None:
        self.show_progress(ProgressState())

    def show_progress(self, state: ProgressState) -> None:
        epoch = UNKNOWN if state.epoch is None else str(state.epoch)
        total_epochs = UNKNOWN if state.total_epochs is None else str(state.total_epochs)
        self.epoch_label.setText(f"Epoch {epoch} / {total_epochs}")
        self.bar.setRange(0, max(state.total_steps or 0, 1))
        self.bar.setValue(min(state.step, self.bar.maximum()))
        self.bar.setFormat("%v / %m steps" if state.total_steps else "")
        loss = UNKNOWN if state.loss is None else f"{state.loss:.4f}"
        self.loss_label.setText(f"Loss: {loss}")
        elapsed, remaining = format_duration(state.elapsed_s), format_duration(state.remaining_s)
        self.time_label.setText(f"Elapsed {elapsed}  |  Remaining {remaining}")
        checkpoint = state.last_checkpoint.name if state.last_checkpoint else UNKNOWN
        self.checkpoint_label.setText(f"Last checkpoint: {checkpoint}")

    def show_download(self, downloaded: int, total: int | None) -> None:
        if total:
            self.bar.setRange(0, PERMILLE)
            self.bar.setValue(min(downloaded * PERMILLE // total, PERMILLE))
            self.bar.setFormat("%p%")
        else:
            self.bar.setRange(0, 0)
        self.epoch_label.setText(download_text(downloaded, total))
