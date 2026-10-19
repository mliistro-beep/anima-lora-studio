"""The "Configure Training" flow from Citron: validate, estimate, check models, write TOMLs, save.

Returns a structured result plus the same status report Citron printed, with text markers
(OK / WARN / ERROR / INFO) instead of emoji.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from anima_trainer.dataset.step_estimator import StepEstimate, estimate_steps
from anima_trainer.dataset.validator import DatasetError, DatasetReport, validate_dataset
from anima_trainer.models.paths import CitronPaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.settings.store import save_settings
from anima_trainer.training.toml_builder import GeneratedConfigs, write_configs

MAX_LISTED_MISSING = 20
RULE = "-" * 54


@dataclass(frozen=True)
class ConfigureResult:
    success: bool
    messages: tuple[str, ...]
    settings: TrainingSettings
    configs: GeneratedConfigs | None = None
    dataset: DatasetReport | None = None
    estimate: StepEstimate | None = None


def configure_training(
    settings: TrainingSettings, paths: CitronPaths, now: datetime
) -> ConfigureResult:
    missing = settings.missing_required_fields()
    if missing:
        errors = tuple(f"ERROR: {label} cannot be empty." for label in missing)
        return ConfigureResult(False, errors, settings)

    messages = _header(settings)
    try:
        report = validate_dataset(Path(settings.image_directory))
    except DatasetError as error:
        return ConfigureResult(False, (*messages, f"ERROR: {error}"), settings)

    messages = (*messages, *_dataset_lines(report))
    if report.image_count == 0:
        failure = (*messages, "", "ERROR: Cannot configure, no images found.")
        return ConfigureResult(False, failure, settings, dataset=report)

    estimate = estimate_steps(
        report.image_count,
        settings.repeats,
        settings.train_batch_size,
        settings.gradient_accumulation_steps,
        settings.max_train_epochs,
    )
    messages = (
        *messages,
        *_estimate_lines(settings, report, estimate),
        *_model_lines(settings, paths),
    )
    missing_models = paths.missing_required_models()
    if missing_models:
        failure = (*messages, "", f"ERROR: Missing models: {', '.join(missing_models)}")
        return ConfigureResult(False, failure, settings, dataset=report, estimate=estimate)

    return _generate(settings, paths, now, messages, report, estimate)


def _generate(
    settings: TrainingSettings,
    paths: CitronPaths,
    now: datetime,
    messages: tuple[str, ...],
    report: DatasetReport,
    estimate: StepEstimate,
) -> ConfigureResult:
    messages = (*messages, "", "Generating TOML configs...")
    try:
        configs = write_configs(settings, paths, now)
    except OSError as error:
        failure = (*messages, f"ERROR: Failed to generate configs: {error}")
        return ConfigureResult(False, failure, settings, dataset=report, estimate=estimate)

    updated = settings.with_changes(
        last_train_config=str(configs.training), last_dataset_config=str(configs.dataset)
    )
    messages = (
        *messages,
        f"  OK: Training config: {configs.training}",
        f"  OK: Dataset  config: {configs.dataset}",
    )
    try:
        save_settings(paths.config_file, updated)
    except OSError as error:
        messages = (*messages, f"WARN: Settings could not be saved to {paths.config_file}: {error}")
    messages = (*messages, "", "OK: Configuration complete, ready to train.")
    return ConfigureResult(True, messages, updated, configs, report, estimate)


def _header(settings: TrainingSettings) -> tuple[str, ...]:
    return (
        f"Project:          {settings.project_name}",
        f"Image directory:  {settings.image_directory}",
        f"Output directory: {settings.output_directory}",
        "",
    )


def _dataset_lines(report: DatasetReport) -> tuple[str, ...]:
    lines: tuple[str, ...] = (f"Images found:     {report.image_count}",)
    missing = report.missing_captions
    if missing:
        listed = tuple(f"    - {name}" for name in missing[:MAX_LISTED_MISSING])
        lines = (*lines, f"WARN: Missing captions ({len(missing)}):", *listed)
        if len(missing) > MAX_LISTED_MISSING:
            lines = (*lines, f"    ... and {len(missing) - MAX_LISTED_MISSING} more")
    elif report.image_count:
        lines = (*lines, "OK: All images have caption files.")
    return (*lines, *(f"WARN: {warning}" for warning in report.warnings))


def _estimate_lines(
    settings: TrainingSettings, report: DatasetReport, estimate: StepEstimate
) -> tuple[str, ...]:
    per_epoch = estimate.steps_per_epoch
    images = f"{report.image_count} imgs x {settings.repeats} repeats"
    epochs = f"{per_epoch} x {settings.max_train_epochs} epochs"
    return (
        "",
        f"-- Step Estimate {RULE[:37]}",
        f"  Steps per epoch: {per_epoch}  ({images})",
        f"  Total steps:     {estimate.total_steps}  ({epochs})",
        RULE,
    )


def _model_lines(settings: TrainingSettings, paths: CitronPaths) -> tuple[str, ...]:
    dit = paths.dit_model(settings.base_model)
    dit_lines = (
        (f"  OK: DiT: {dit}",)
        if dit.exists()
        else (f"  INFO: DiT: {dit}", "      (will auto-download when training starts)")
    )
    others = tuple(
        f"  OK: {label}: {path}" if path.exists() else f"  ERROR: {label} missing: {path}"
        for label, path in (("Qwen3", paths.qwen3_model), ("VAE", paths.vae_model))
    )
    return ("", "Checking models...", *dit_lines, *others)
