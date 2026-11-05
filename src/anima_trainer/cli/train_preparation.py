"""What `train` and the TUI share before a training: reading the data folder, checking the
choices and describing the result. Nothing here writes to disk."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from pydantic_core import ErrorDetails
from PySide6.QtCore import QSettings

from anima_trainer.cli.train_options import TrainOptions, missing_option_lines
from anima_trainer.cli.train_plan import (
    missing_model_problems,
    runtime_problem_lines,
    settings_problem_lines,
)
from anima_trainer.cli.train_summary import (
    base_model_problem,
    resolution_warning,
    settings_for,
    summary_lines,
)
from anima_trainer.dataset.validator import DatasetError, DatasetReport, validate_dataset
from anima_trainer.models.linked_models import LinkedModel
from anima_trainer.models.linked_store import load_links
from anima_trainer.presets.catalog import LoadedPresets, find_preset, load_presets
from anima_trainer.presets.preset_file import Preset
from anima_trainer.preview.settings import PreviewLoadResult, load_preview_settings
from anima_trainer.runtime.data_dir import (
    OpenedRuntime,
    default_data_dir,
    load_data_dir,
    open_data_dir,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.system.gpu_memory import read_all_gpu_memory
from anima_trainer.system.memory_safety import read_memory_safety, warnings_only


@dataclass(frozen=True)
class TerminalContext:
    opened: OpenedRuntime
    paths: RuntimePaths  # with the linked models
    presets: LoadedPresets
    previews: PreviewLoadResult


@dataclass(frozen=True)
class TrainPlan:
    preset: Preset
    settings: TrainingSettings
    report: DatasetReport


def load_terminal_context() -> TerminalContext:
    """Needs the application names set: QSettings finds the graphical app's file through them."""
    store = QSettings()
    return terminal_context(
        open_data_dir(load_data_dir(store, default_data_dir())), load_links(store)
    )


def terminal_context(opened: OpenedRuntime, links: Mapping[str, LinkedModel]) -> TerminalContext:
    paths = opened.paths.with_links(links)
    return TerminalContext(
        opened,
        paths,
        load_presets(paths.presets_dir),
        load_preview_settings(paths.preview_settings_file),
    )


def prepare_training(
    options: TrainOptions, context: TerminalContext
) -> TrainPlan | tuple[str, ...]:
    """What this training would be, or the lines saying why it cannot start."""
    if options.preset is None:
        return ("ERROR: --preset is required (see --list-presets)",)
    preset = find_preset(context.presets, options.preset)
    if preset is None:
        names = ", ".join(item.name for item in context.presets.presets)
        return (f"ERROR: Unknown preset '{options.preset}'. Available: {names}",)
    missing = missing_option_lines(options, preset)
    if missing:
        return missing
    try:
        settings = settings_for(options, preset, context.opened.settings)
    except ValidationError as error:
        return tuple(f"ERROR: {_field_error(detail)}" for detail in error.errors())
    problem = base_model_problem(settings)
    if problem is not None:
        return (problem,)
    blockers = (
        runtime_problem_lines(context.opened.problems)  # without the runtime the rest says little
        or settings_problem_lines((*context.opened.warnings, *context.previews.warnings))
        or missing_model_problems(context.paths, settings.base_model)
    )
    if blockers:
        return blockers
    try:
        report = validate_dataset(Path(settings.image_directory))
    except DatasetError as error:
        return (f"ERROR: {error}",)
    if report.image_count == 0:
        return (f"ERROR: No images found in {settings.image_directory}",)
    return TrainPlan(preset, settings, report)


def plan_lines(plan: TrainPlan, context: TerminalContext) -> tuple[str, ...]:
    """The summary to approve, with what the system and the GPU suggest watching out for."""
    warning = resolution_warning(plan.settings, read_all_gpu_memory())
    return (
        *warnings_only(read_memory_safety()),
        *summary_lines(plan.preset, plan.settings, plan.report, context.previews.settings),
        *((warning,) if warning else ()),
    )


def _field_error(detail: ErrorDetails) -> str:
    return f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
