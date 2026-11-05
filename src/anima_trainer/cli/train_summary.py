"""From the options to the settings of the training, and what the user reads before approving."""

import os
from pathlib import Path

from anima_trainer.cli.train_options import TrainOptions
from anima_trainer.cli.train_plan import GRAPHICAL_APP_NOTE
from anima_trainer.dataset.step_estimator import estimate_steps
from anima_trainer.dataset.validator import DatasetReport
from anima_trainer.presets.apply import apply_preset
from anima_trainer.presets.preset_file import Preset
from anima_trainer.preview.settings import PreviewSettings
from anima_trainer.settings.defaults import DEFAULTS
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.system.gpu_memory import GpuMemory, memory_of

TERMINAL_BASE_MODEL = DEFAULTS["base_model"]
# The user's RTX 3060 (12 GB): 768 is the size known to be safe there.
SAFE_RESOLUTION = 768
SMALL_GPU_MIB = 12288


def settings_for(
    options: TrainOptions, preset: Preset, saved: TrainingSettings
) -> TrainingSettings:
    """The settings of this training; raises pydantic's ValidationError on a value they refuse.

    Folders are made absolute here: Configure would read a relative one from the data folder,
    while on a command line it means from the current folder.
    """
    return apply_preset(preset, saved).with_changes(
        project_name=options.project,
        image_directory=_absolute(options.dataset),
        output_directory=_absolute(options.output),
        max_train_epochs=options.epochs,
        repeats=options.repeats,
        resolution=options.resolution,
    )


def _absolute(folder: str | None) -> str:
    # Not resolve(): a link stays as the user wrote it, in the summary and in the configs.
    return os.path.abspath(Path(folder or "").expanduser())


def base_model_problem(settings: TrainingSettings) -> str | None:
    if settings.base_model == TERMINAL_BASE_MODEL:
        return None
    return (
        f"ERROR: The terminal trains only {TERMINAL_BASE_MODEL} for now, not "
        f"{settings.base_model}. For another model, {GRAPHICAL_APP_NOTE}"
    )


def resolution_warning(settings: TrainingSettings, gpus: tuple[GpuMemory, ...]) -> str | None:
    memory = memory_of(gpus, settings.gpu_index)
    if memory is None or settings.resolution <= SAFE_RESOLUTION:
        return None
    if memory.total_mib > SMALL_GPU_MIB:
        return None
    return (
        f"WARN: Resolution {settings.resolution} may not fit in {memory.total_mib} MiB of VRAM; "
        f"{SAFE_RESOLUTION} is the safe size for this GPU."
    )


def summary_lines(
    preset: Preset,
    settings: TrainingSettings,
    report: DatasetReport,
    previews: PreviewSettings,
) -> tuple[str, ...]:
    images, repeats, epochs = report.image_count, settings.repeats, settings.max_train_epochs
    estimate = estimate_steps(
        images,
        repeats,
        settings.train_batch_size,
        settings.gradient_accumulation_steps,
        epochs,
    )
    samples = images * repeats * epochs
    about = f" ({preset.description})" if preset.description else ""
    prompts = _count(len(previews.prompts), "prompt")
    shown_previews = f"on ({prompts})" if previews.enabled else "off"
    return (
        f"Preset:    {preset.name}{about}",
        f"Project:   {settings.project_name}",
        f"Dataset:   {settings.image_directory}",
        f"Output:    {settings.output_directory}",
        f"Model:     {settings.base_model}",
        f"Network:   dim {settings.network_dim}, alpha {settings.network_alpha}",
        f"Size:      {settings.resolution}",
        f"Samples:   {_count(images, 'image')} x {_count(repeats, 'repeat')} x "
        f"{_count(epochs, 'epoch')} = {_count(samples, 'sample')}",
        f"Total:     {_count(estimate.total_steps, 'step')}",
        f"Previews:  {shown_previews}",
        *(f"WARN: {warning}" for warning in report.warnings),
    )


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def preset_list_lines(presets: tuple[Preset, ...]) -> tuple[str, ...]:
    return tuple(_preset_line(preset) for preset in presets)


def _preset_line(preset: Preset) -> str:
    suggested = preset.values.get("max_train_epochs")
    epochs = f"{suggested} epochs suggested" if suggested else "no epochs suggested"
    kind = "built-in" if preset.builtin else "yours"
    about = f": {preset.description}" if preset.description else ""
    return f"{preset.name} ({kind}, {epochs}){about}"
