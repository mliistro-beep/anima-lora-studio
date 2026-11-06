"""Wires the Previews and Generator tabs to the sample watcher, the generator session and the
settings files, for the main window."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QWidget

from anima_trainer.preview.checkpoints import Checkpoint, list_checkpoints
from anima_trainer.preview.generator_settings import (
    GeneratorSettings,
    load_generator_settings,
    save_generator_settings,
)
from anima_trainer.preview.inference_command import (
    GenerationRequest,
    LoraUse,
    output_image_path,
    project_generated_folder,
)
from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.sample_watcher import SampleWatcher
from anima_trainer.preview.settings import PreviewSettings, load_preview_settings
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.ui.dialogs.image_viewer import ImageViewer
from anima_trainer.ui.tabs.generator_tab import GeneratorTab
from anima_trainer.ui.tabs.previews_tab import PreviewsTab
from anima_trainer.workflow.generator_session import GeneratorSession

TRAINING_NOTE = (
    "A training is running: images use a lighter memory profile so the training keeps its "
    "GPU memory, and both run slower."
)
NO_RUNTIME_NOTE = "Install the training runtime and the models in the Settings tab first."
NO_OUTPUT_DIR = "Set the Output Directory in the Training tab and Configure first."


class PreviewController(QObject):
    def __init__(
        self,
        previews_tab: PreviewsTab,
        generator_tab: GeneratorTab,
        generator: GeneratorSession,
        watcher: SampleWatcher,
        window: QWidget,
        gpu_index: Callable[[], str],
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        super().__init__(window)
        self.previews_tab = previews_tab
        self.generator_tab = generator_tab
        self.generator = generator
        self.watcher = watcher
        self.viewer: ImageViewer | None = None
        self._window = window
        self._clock = clock
        self._paths: RuntimePaths | None = None
        self._settings = TrainingSettings()
        self._gpu_index = gpu_index
        self._training = False
        self._ready = False
        self._connect()

    def load(self, paths: RuntimePaths, settings: TrainingSettings) -> tuple[str, ...]:
        """Show the preview and generator settings of a data folder; returns their warnings."""
        self._paths, self._settings = paths, settings
        previews = load_preview_settings(paths.preview_settings_file)
        generator = load_generator_settings(paths.generator_settings_file)
        self.previews_tab.set_settings(previews.settings)
        self.generator_tab.set_settings(generator.settings)
        self._show_project()
        return (*previews.warnings, *generator.warnings)

    def set_paths(self, paths: RuntimePaths) -> None:
        """After a model file was linked or unlinked: the generator uses the new model paths."""
        self._paths = paths

    def preview_settings(self) -> PreviewSettings | tuple[str, ...]:
        return self.previews_tab.settings()

    def previews_enabled(self) -> bool:
        return self.previews_tab.previews_enabled()

    def saved_settings_turned_off(self) -> PreviewSettings:
        """The saved settings with previews off, for a form that cannot be read: a later
        Configure must not turn them back on."""
        if self._paths is None:
            return PreviewSettings()
        saved = load_preview_settings(self._paths.preview_settings_file).settings
        return saved.with_changes(enabled=False)

    def previews_configured(self, previews: PreviewSettings) -> None:
        """After a successful Configure: the gallery columns follow its prompts."""
        self.previews_tab.set_prompt_labels(previews.prompts)

    def set_training_settings(self, settings: TrainingSettings) -> None:
        """After a Configure: the gallery and checkpoints follow the configured project."""
        self._settings = settings
        self._show_project()

    def set_ready(self, ready: bool) -> None:
        """False while the runtime or models are missing, or the app is busy installing."""
        self._ready = ready
        self._update_generator_state()

    def training_started(self) -> None:
        self._training = True
        output_dir = self._output_dir()
        if output_dir is not None:
            since = self._clock().replace(microsecond=0)
            self.watcher.watch(output_dir, self._settings.project_name, since)
        self.previews_tab.set_editable(False)
        self._update_generator_state()

    def training_finished(self) -> None:
        self._training = False
        self.watcher.stop()
        self.watcher.refresh()
        self.previews_tab.set_editable(True)
        self.refresh_checkpoints()

    def is_generating(self) -> bool:
        return self.generator.is_running()

    def refresh_checkpoints(self) -> None:
        output_dir = self._output_dir()
        checkpoints = (
            () if output_dir is None else list_checkpoints(output_dir, self._settings.project_name)
        )
        self.generator_tab.set_checkpoints(checkpoints)
        self._update_generator_state()

    def shutdown(self) -> None:
        self.watcher.stop()
        self.generator.shutdown()

    def _connect(self) -> None:
        self.watcher.samples_changed.connect(self.previews_tab.show_samples)
        self.previews_tab.refresh_requested.connect(self.watcher.refresh)
        self.previews_tab.image_activated.connect(self._show_image)
        tab = self.generator_tab
        tab.generate_requested.connect(lambda: self._generate(compare=False))
        tab.compare_requested.connect(lambda: self._generate(compare=True))
        tab.stop_requested.connect(self.generator.stop)
        tab.refresh_requested.connect(self.refresh_checkpoints)
        tab.image_activated.connect(self._show_image)
        self.generator.log_line.connect(tab.log.append_line)
        self.generator.step_progress.connect(tab.show_step)
        self.generator.image_ready.connect(tab.show_image)
        self.generator.running_changed.connect(lambda _running: self._update_generator_state())
        self.generator.finished.connect(lambda _ok, message: tab.log.append_line(message))

    def _show_project(self) -> None:
        """Previews of the last run of the project and its checkpoints."""
        output_dir = self._output_dir()
        if output_dir is None:
            self.watcher.stop()
            self.previews_tab.show_samples(())
        else:
            self.watcher.watch(output_dir, self._settings.project_name, None)
            self.watcher.stop()
        self.refresh_checkpoints()

    def _generate(self, compare: bool) -> None:
        tab = self.generator_tab
        settings = tab.settings()
        if isinstance(settings, tuple):
            tab.log.set_lines(settings)
            return
        output_dir = self._output_dir()
        problems = (*settings.problems(), *(() if output_dir is not None else (NO_OUTPUT_DIR,)))
        if problems or self._paths is None or output_dir is None:
            tab.log.set_lines(tuple(f"ERROR: {problem}" for problem in problems))
            return
        try:
            save_generator_settings(self._paths.generator_settings_file, settings)
        except OSError as error:
            tab.log.append_line(f"WARN: Generator settings could not be saved: {error}")
        checkpoints = tab.checkpoints() if compare else (tab.selected_checkpoint(),)
        requests = tuple(
            self._request(self._paths, settings, checkpoint, output_dir)
            for checkpoint in checkpoints
        )
        refused = self.generator.generate(self._paths, requests, self._gpu_index(), self._training)
        if refused is not None:
            tab.log.append_line(f"ERROR: {refused}")

    def _request(
        self,
        paths: RuntimePaths,
        settings: GeneratorSettings,
        checkpoint: Checkpoint | None,
        output_dir: Path,
    ) -> GenerationRequest:
        label = checkpoint.label if checkpoint is not None else "No LoRA"
        project = self._settings.project_name or "image"
        folder = project_generated_folder(output_dir)
        output = output_image_path(folder, project, label, settings.seed, self._clock())
        loras = (
            (LoraUse(checkpoint.path, settings.lora_multiplier),) if checkpoint is not None else ()
        )
        return GenerationRequest(
            dit=paths.dit_model(self._settings.base_model),
            loras=loras,
            prompt=settings.prompt,
            negative_prompt=settings.negative_prompt,
            size=SampleSize(settings.width, settings.height),
            steps=settings.steps,
            cfg_scale=settings.cfg_scale,
            flow_shift=settings.flow_shift,
            seed=settings.seed,
            sampler=settings.sampler,
            output=output,
        )

    def _update_generator_state(self) -> None:
        note = TRAINING_NOTE if self._training else "" if self._ready else NO_RUNTIME_NOTE
        self.generator_tab.set_running(self.generator.is_running(), self._ready, note)

    def _output_dir(self) -> Path | None:
        """The configured output folder; None when there is none yet (a blank value would
        otherwise mean the folder the app was started from)."""
        if self._paths is None or not self._settings.output_directory.strip():
            return None
        return self._paths.resolve(self._settings.output_directory)

    def _show_image(self, path: Path) -> None:
        if self.viewer is not None:
            self.viewer.deleteLater()
        self.viewer = ImageViewer(path, self._window)
        self.viewer.open()
