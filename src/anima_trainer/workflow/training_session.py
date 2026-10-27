"""Coordinates Configure, model download, Start and Stop for the window.

Holds the settings saved by the last Configure (what Citron re-read from config.json on
Start) and turns runner and downloader signals into log lines, progress and state changes.
"""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QObject, Signal, SignalInstance

from anima_trainer.models.catalog import ExpectedFile
from anima_trainer.models.model_downloads import ModelDownload, dit_model_download, model_download
from anima_trainer.preview.preview_plan import PreviewPlan, plan_previews
from anima_trainer.preview.settings import PreviewSettings, save_preview_settings
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.system.gpu_memory import GpuMemory, memory_of
from anima_trainer.training.command_builder import LaunchSpec
from anima_trainer.training.configure_service import ConfigureResult, configure_training
from anima_trainer.training.log_parser import TrainingEvent
from anima_trainer.training.oom_hint import oom_hint, read_kernel_log
from anima_trainer.training.outcome_report import outcome_messages, run_failed
from anima_trainer.training.progress import ProgressState, apply_event
from anima_trainer.training.runner import RunOutcome
from anima_trainer.training.start_service import StartRequest, plan_start


class SessionState(Enum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    TRAINING = "training"
    STOPPING = "stopping"


class Runner(Protocol):
    """The part of TrainingRunner the session uses."""

    line_received: SignalInstance
    event_received: SignalInstance
    finished: SignalInstance
    failed_to_start: SignalInstance
    warning: SignalInstance

    def start(self, spec: LaunchSpec, log_path: Path) -> None: ...
    def stop(self) -> None: ...
    def shutdown(self) -> None: ...
    def log_tail(self) -> tuple[str, ...]: ...


class Downloader(Protocol):
    """The part of ModelDownloader the session uses."""

    progress: SignalInstance
    finished: SignalInstance
    failed: SignalInstance
    cancelled: SignalInstance

    def start(self, url: str, destination: Path, expected: ExpectedFile | None = None) -> None: ...
    def cancel(self) -> None: ...
    def shutdown(self) -> None: ...


class TrainingSession(QObject):
    state_changed = Signal(object)  # SessionState
    log_line = Signal(str)
    log_cleared = Signal()
    progress_changed = Signal(object)  # ProgressState
    download_progress = Signal(object, object)  # downloaded, total or None
    settings_changed = Signal(object)  # TrainingSettings saved by Configure
    models_changed = Signal()

    def __init__(
        self,
        paths: RuntimePaths,
        settings: TrainingSettings,
        runner: Runner,
        downloader: Downloader,
        clock: Callable[[], datetime] = datetime.now,
        kernel_log: Callable[[], str | None] = read_kernel_log,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._settings = settings
        self._runner = runner
        self._downloader = downloader
        self._clock = clock
        self._kernel_log = kernel_log
        self._gpu_memory: tuple[GpuMemory, ...] = ()  # from the GPU detection thread
        self._state = SessionState.IDLE
        self._progress = ProgressState()
        self._pending: StartRequest | None = None
        self._connect()

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def paths(self) -> RuntimePaths:
        return self._paths

    @property
    def settings(self) -> TrainingSettings:
        return self._settings

    def set_gpu_memory(self, gpus: tuple[GpuMemory, ...]) -> None:
        """Memory of the GPUs, read off the UI thread; Configure sizes the previews with it."""
        self._gpu_memory = gpus

    def set_runtime(self, paths: RuntimePaths, settings: TrainingSettings) -> None:
        self._require_idle()
        self._paths = paths
        self._settings = settings

    def configure(
        self, settings: TrainingSettings, previews: PreviewSettings | None = None
    ) -> ConfigureResult:
        """Write the training configs; with previews on, their size is fitted to the GPU.

        The preview settings are saved to preview.json after a successful Configure.
        """
        self._require_idle()
        plan = PreviewPlan()
        if previews is not None and previews.enabled:
            memory = memory_of(self._gpu_memory, settings.gpu_index)
            plan = plan_previews(previews, memory.total_mib if memory else None)
        result = configure_training(settings, self._paths, self._clock(), plan)
        if not result.success:
            return result
        self._settings = result.settings
        self.settings_changed.emit(result.settings)
        if previews is None:
            return result
        try:
            save_preview_settings(self._paths.preview_settings_file, previews)
        except OSError as error:
            file = self._paths.preview_settings_file
            warning = f"WARN: Preview settings could not be saved to {file}: {error}"
            return replace(result, messages=(*result.messages, warning))
        return result

    def start(self, request: StartRequest) -> None:
        self._require_idle()
        self.log_cleared.emit()
        download = dit_model_download(self._paths, request.base_model)
        if download is None:
            self._launch(request)
            return
        self._pending = request
        self._begin_download(download)

    def download_model(self, key: str) -> None:
        """Download a missing model by its key (base model name, or Qwen3 / VAE file name)."""
        self._require_idle()
        download = model_download(self._paths, key)
        if download is not None:
            self._begin_download(download)

    def stop(self) -> None:
        if self._state is SessionState.DOWNLOADING:
            self._pending = None
            self._downloader.cancel()
        elif self._state is SessionState.TRAINING:
            self._set_state(SessionState.STOPPING)
            self._runner.stop()

    def shutdown(self) -> None:
        """Block until the training process and any download have stopped."""
        self._pending = None
        self._runner.shutdown()
        self._downloader.shutdown()

    def _connect(self) -> None:
        self._runner.line_received.connect(self.log_line.emit)
        self._runner.event_received.connect(self._on_event)
        self._runner.finished.connect(self._on_run_finished)
        self._runner.failed_to_start.connect(self._on_failed_to_start)
        self._runner.warning.connect(lambda message: self.log_line.emit(f"WARN: {message}"))
        self._downloader.progress.connect(self.download_progress.emit)
        self._downloader.finished.connect(self._on_download_finished)
        self._downloader.failed.connect(self._on_download_failed)
        self._downloader.cancelled.connect(self._on_download_cancelled)

    def _begin_download(self, download: ModelDownload) -> None:
        self._log(
            f"INFO: Downloading {download.description}...",
            "   This may take a few minutes. Future runs will skip this step.",
            f"   Destination: {download.destination}",
            *(
                ("   The file is checked against its SHA-256 when complete.",)
                if download.expected
                else ()
            ),
            "",
        )
        self._set_state(SessionState.DOWNLOADING)
        self._downloader.start(download.url, download.destination, download.expected)

    def _launch(self, request: StartRequest) -> None:
        self._reset_progress()
        plan = plan_start(self._settings, self._paths, request, self._clock())
        self._log(*plan.messages)
        if plan.spec is None or plan.log_path is None:
            self._set_state(SessionState.IDLE)
            return
        try:
            self._runner.start(plan.spec, plan.log_path)
        except OSError as error:  # the log file could not be opened
            self._log(f"ERROR: Could not start training: {error}")
            self._set_state(SessionState.IDLE)
            return
        self._set_state(SessionState.TRAINING)

    def _reset_progress(self) -> None:
        self._progress = ProgressState()
        self.progress_changed.emit(self._progress)

    def _on_event(self, event: TrainingEvent) -> None:
        self._progress = apply_event(self._progress, event)
        self.progress_changed.emit(self._progress)

    def _on_run_finished(self, outcome: RunOutcome) -> None:
        hint = None
        if run_failed(outcome):
            hint = oom_hint(
                outcome.exit_code, self._runner.log_tail(), self._kernel_log(), outcome.crashed
            )
        self._log(*outcome_messages(outcome, self._settings.output_directory, hint))
        self._set_state(SessionState.IDLE)

    def _on_failed_to_start(self, message: str) -> None:
        self._log(f"ERROR: {message}")
        self._set_state(SessionState.IDLE)

    def _on_download_finished(self, _path: str) -> None:
        self._log("OK: Base model downloaded successfully.", "")
        self.models_changed.emit()
        pending, self._pending = self._pending, None
        if pending is None:
            self._end_download()
            return
        self._launch(pending)

    def _on_download_failed(self, message: str) -> None:
        self._pending = None
        self._log(f"ERROR: Download failed: {message}")
        self._end_download()

    def _on_download_cancelled(self) -> None:
        self._pending = None
        self._log("INFO: Download cancelled.")
        self._end_download()

    def _end_download(self) -> None:
        # The progress panel showed the download; clear it when no run follows.
        self._reset_progress()
        self._set_state(SessionState.IDLE)

    def _log(self, *lines: str) -> None:
        for line in lines:
            self.log_line.emit(line)

    def _set_state(self, state: SessionState) -> None:
        if state is not self._state:
            self._state = state
            self.state_changed.emit(state)

    def _require_idle(self) -> None:
        if self._state is not SessionState.IDLE:
            raise RuntimeError(f"The session is busy ({self._state.value})")
