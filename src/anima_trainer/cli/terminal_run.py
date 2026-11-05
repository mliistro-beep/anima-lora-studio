"""One training started from the terminal: the lock, Configure and Start, then its end.

Used by `train` and by the TUI, each with its own way of showing lines and running Qt's events.
"""

from collections.abc import Callable

from anima_trainer.cli.train_plan import exit_code_for
from anima_trainer.models.download_worker import ModelDownloader
from anima_trainer.preview.settings import PreviewSettings
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.system.gpu_lock import TERMINAL_JOB, GpuLock
from anima_trainer.system.gpu_memory import read_all_gpu_memory
from anima_trainer.system.resource_monitor import ResourceMonitor
from anima_trainer.training.progress import ProgressState
from anima_trainer.training.runner import RunOutcome, TrainingRunner
from anima_trainer.training.start_service import StartRequest
from anima_trainer.workflow.training_session import SessionState, TrainingSession


class TerminalRun:
    def __init__(
        self,
        paths: RuntimePaths,
        settings: TrainingSettings,
        previews: PreviewSettings,
        on_line: Callable[[str], None],
        on_progress: Callable[[ProgressState], None],
        on_ended: Callable[[], None],
    ) -> None:
        self._paths = paths
        self._settings = settings
        self._previews = previews
        self._on_line = on_line
        self._on_ended = on_ended
        self._lock = GpuLock()
        self._outcome: RunOutcome | None = None
        self._active = False
        runner = TrainingRunner(monitor=ResourceMonitor())
        self._session = TrainingSession(paths, settings, runner, ModelDownloader(), lock=self._lock)
        self._session.log_line.connect(on_line)
        self._session.progress_changed.connect(on_progress)
        # After the session's own slots, so that it is already idle and its lock given back.
        runner.finished.connect(self._on_finished)
        runner.failed_to_start.connect(lambda _message: self._end())

    @property
    def active(self) -> bool:
        return self._active

    @property
    def exit_code(self) -> int:
        return exit_code_for(self._outcome)

    def begin(self) -> bool:
        """Take the lock, write the configs and start; False when no training is running."""
        # Before Configure: a refused run leaves the configs as they were.
        refusal = self._lock.acquire(TERMINAL_JOB, self._paths.gpu_lock_file)
        if refusal is not None:
            self._on_line(f"ERROR: {refusal}")
            return False
        # Read now: time can pass at the confirmation, and previews fit what is free at the start.
        self._session.set_gpu_memory(read_all_gpu_memory())
        # Not persisted: the window finds its settings as the user left them.
        result = self._session.configure(self._settings, self._previews, persist=False)
        for line in (*result.messages, ""):
            self._on_line(line)
        if not result.success:
            self._lock.release(TERMINAL_JOB)
            return False
        saved = self._session.settings
        self._active = True
        self._session.start(
            StartRequest(saved.gpu_index, saved.num_cpu_threads_per_process, saved.base_model)
        )
        if self._session.state is SessionState.IDLE:
            self._active = False
            self._lock.release(TERMINAL_JOB)
            return False
        return self._active

    def stop(self) -> None:
        self._session.stop()

    def shutdown(self) -> None:
        """Block until no training process is left; safe to call at any time."""
        self._session.shutdown()
        self._lock.release(TERMINAL_JOB)
        self._active = False

    def _on_finished(self, outcome: RunOutcome) -> None:
        self._outcome = outcome
        self._end()

    def _end(self) -> None:
        if self._active:
            self._active = False
            self._lock.release(TERMINAL_JOB)
            self._on_ended()
