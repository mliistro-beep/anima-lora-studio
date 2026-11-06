"""Runs the image generator for the Generator tab: a queue of images (one for Generate, one
per checkpoint for Compare epochs). generate.py fits each image to the free GPU memory it
reads itself, so nothing here blocks the window on nvidia-smi.

The process is run by a TrainingRunner (process group stop, log file); its lines are turned
into progress, images and log lines.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from anima_trainer.preview.inference_command import GenerationRequest, build_generation_launch
from anima_trainer.preview.vram_profile import PROFILES
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.system.gpu_lock import GENERATOR_JOB, GpuLock
from anima_trainer.training.runner import RunOutcome
from anima_trainer.training.toml_builder import FILE_TIMESTAMP_FORMAT
from anima_trainer.workflow.training_session import Runner

EXIT_OUT_OF_MEMORY = 3  # generate.py: no profile fitted the free GPU memory and RAM, or all ran out
STEP_PREFIX = "STEP: "
IMAGE_PREFIX = "IMAGE: "
PROFILE_PREFIX = "PROFILE: "


class GeneratorSession(QObject):
    running_changed = Signal(bool)
    log_line = Signal(str)
    step_progress = Signal(int, int)  # done, total
    image_ready = Signal(object)  # Path
    finished = Signal(bool, str)  # success, message

    def __init__(
        self,
        runner: Runner,
        clock: Callable[[], datetime] = datetime.now,
        parent: QObject | None = None,
        lock: GpuLock | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._clock = clock
        self._lock = lock
        self._queue: tuple[GenerationRequest, ...] = ()
        self._paths: RuntimePaths | None = None
        self._gpu_index = ""
        self._training = False
        self._done = 0
        self._image_seen = False
        self._running = False
        runner.line_received.connect(self._on_line)
        runner.finished.connect(self._on_finished)
        runner.failed_to_start.connect(lambda message: self._finish(False, f"ERROR: {message}"))
        runner.warning.connect(lambda message: self.log_line.emit(f"WARN: {message}"))

    def is_running(self) -> bool:
        return self._running

    def generate(
        self,
        paths: RuntimePaths,
        requests: tuple[GenerationRequest, ...],
        gpu_index: str,
        training: bool,
    ) -> str | None:
        """Start the queue; returns why it cannot start, or None."""
        if self._running:
            return "A generation is already running."
        if not requests:
            return "Nothing to generate."
        problem = _model_problem(paths, requests)
        if problem is not None:
            return problem
        if self._lock is not None:
            refusal = self._lock.acquire(GENERATOR_JOB, paths.gpu_lock_file)
            if refusal is not None:
                return refusal
        self._queue, self._paths, self._gpu_index = requests, paths, gpu_index
        self._training, self._done = training, 0
        self._set_running(True)
        self._start_next()
        return None

    def stop(self) -> None:
        if self._running:
            self._queue = self._queue[:1]  # only the image being generated is left
            self._runner.stop()

    def shutdown(self) -> None:
        self._queue = ()
        self._runner.shutdown()
        self._release_lock()

    def _start_next(self) -> None:
        if self._paths is None or not self._queue:
            return
        request = self._queue[0]
        spec = build_generation_launch(
            self._paths, request, PROFILES, self._gpu_index, self._training
        )
        stamp = self._clock().strftime(FILE_TIMESTAMP_FORMAT)
        log_path = self._paths.logs_dir / f"generate_{stamp}.log"
        position = self._done + 1
        total = self._done + len(self._queue)
        self.log_line.emit(f"INFO: Generating image {position}/{total}: {request.output.name}")
        self._image_seen = False
        try:
            self._runner.start(spec, log_path)
        except OSError as error:  # the log file could not be opened
            self._finish(False, f"ERROR: Could not start generation: {error}")

    def _on_line(self, line: str) -> None:
        if not self._running:
            return
        if line.startswith(STEP_PREFIX):
            done, _, total = line.removeprefix(STEP_PREFIX).partition("/")
            if done.isdecimal() and total.isdecimal():
                self.step_progress.emit(int(done), int(total))
                return
        if line.startswith(IMAGE_PREFIX):
            self._image_seen = True
            self.image_ready.emit(Path(line.removeprefix(IMAGE_PREFIX)))
            return
        if line.startswith(PROFILE_PREFIX):
            self.log_line.emit(f"INFO: Memory profile: {line.removeprefix(PROFILE_PREFIX)}")
            return
        self.log_line.emit(line)

    def _on_finished(self, outcome: RunOutcome) -> None:
        if not self._running:
            return
        if outcome.stopped:
            self._finish(False, "INFO: Generation stopped.")
        elif outcome.exit_code == 0 and self._image_seen:
            self._done += 1
            self._queue = self._queue[1:]
            if self._queue:
                self._start_next()
            else:
                images = "image" if self._done == 1 else "images"
                self._finish(True, f"OK: {self._done} {images} generated.")
        elif outcome.exit_code == EXIT_OUT_OF_MEMORY:
            self._finish(
                False,
                "ERROR: Not enough GPU memory or system RAM for any profile, see the log above.",
            )
        else:
            where = outcome.log_path if outcome.log_path is not None else "the log above"
            self._finish(
                False, f"ERROR: Generation failed (exit code {outcome.exit_code}), see {where}"
            )

    def _finish(self, ok: bool, message: str) -> None:
        self._queue = ()
        self._release_lock()
        self._set_running(False)
        self.finished.emit(ok, message)

    def _release_lock(self) -> None:
        if self._lock is not None:
            self._lock.release(GENERATOR_JOB)

    def _set_running(self, running: bool) -> None:
        if running != self._running:
            self._running = running
            self.running_changed.emit(running)


def _model_problem(paths: RuntimePaths, requests: tuple[GenerationRequest, ...]) -> str | None:
    missing = paths.missing_required_models()
    if missing:
        return f"Missing models: {', '.join(missing)}"
    for request in requests:
        if not request.dit.is_file():
            return f"Model file not found: {request.dit}"
        for lora in request.loras:
            if not lora.path.is_file():
                return f"LoRA file not found: {lora.path}"
    return None
