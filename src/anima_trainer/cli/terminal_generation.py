"""One generation started from the terminal: settings saved, the plan, the lock, then its end.

The TUI's counterpart of the Generator tab without a project: the base model or a model file
of the user's, the LoRAs with their triggers, and the window's GPU. The settings are saved to
the window's generator.json first, as the window does, so a wrong prompt is not lost.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from anima_trainer.cli.generator_values import gpu_index_for
from anima_trainer.preview.generation_plan import GenerationTarget, plan_generation, plan_queue
from anima_trainer.preview.generator_settings import GeneratorSettings, save_generator_settings
from anima_trainer.preview.inference_command import GenerationRequest
from anima_trainer.preview.progress_text import ImageProgress
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.system.gpu_lock import GpuLock
from anima_trainer.training.runner import TrainingRunner
from anima_trainer.workflow.generator_session import GeneratorSession
from anima_trainer.workflow.training_session import Runner

WITHOUT_PROJECT = GenerationTarget(project="", checkpoints=(None,))


class TerminalGeneration:
    def __init__(
        self,
        paths: RuntimePaths,
        on_line: Callable[[str], None],
        on_step: Callable[[int, int], None],
        on_image: Callable[[Path], None],
        on_ended: Callable[[bool, str], None],
        runner: Runner | None = None,
        clock: Callable[[], datetime] = datetime.now,
        on_image_started: Callable[[ImageProgress], None] | None = None,
    ) -> None:
        self._paths = paths
        self._on_line = on_line
        self._clock = clock
        self._session = GeneratorSession(
            runner if runner is not None else TrainingRunner(), clock=clock, lock=GpuLock()
        )
        self._session.log_line.connect(on_line)
        self._session.step_progress.connect(on_step)
        self._session.image_ready.connect(on_image)
        self._session.finished.connect(on_ended)
        if on_image_started is not None:
            self._session.image_started.connect(on_image_started)

    @property
    def active(self) -> bool:
        return self._session.is_running()

    def begin(self, settings: GeneratorSettings, queue: bool = False) -> bool:
        """Save the settings, plan the form (or with `queue` the queue's entries) and start;
        False when nothing is being generated."""
        try:
            save_generator_settings(self._paths.generator_settings_file, settings)
        except OSError as error:
            self._on_line(f"WARN: Generator settings could not be saved: {error}")
        now = self._clock()
        if queue:
            plan = plan_queue(settings, self._paths, now)
        else:
            plan = plan_generation(settings, self._paths, WITHOUT_PROJECT, now)
        requests = tuple(item for item in plan if isinstance(item, GenerationRequest))
        if not requests:
            for line in plan:
                self._on_line(str(line))
            return False
        if queue:
            entries = len(settings.queue)
            self._on_line(f"INFO: Queue: {entries} entries, {len(requests)} images.")
        self._on_line(f"INFO: Prompt: {requests[0].prompt}")
        gpu_index = gpu_index_for(self._paths.config_file)
        refused = self._session.generate(self._paths, requests, gpu_index, training=False)
        if refused is not None:
            self._on_line(f"ERROR: {refused}")
            return False
        return True

    def stop(self) -> None:
        self._session.stop()

    def shutdown(self) -> None:
        """Block until no generation process is left; safe to call at any time."""
        self._session.shutdown()
