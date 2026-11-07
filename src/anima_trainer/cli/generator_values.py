"""The TUI generator's fields: from typed text to GeneratorSettings, and the LoRA list edits.

The settings are the window's (<data>/generator.json), so both find the same LoRAs, triggers
and prompts. Nothing here writes to disk.
"""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from pydantic import ValidationError

from anima_trainer.preview.form_fields import GENERATOR_RESET_FIELDS, GENERATOR_SETTINGS_LABELS
from anima_trainer.preview.generator_settings import MAX_LORAS, GeneratorSettings, LoraEntry
from anima_trainer.settings.field_specs import describe_validation_error
from anima_trainer.settings.store import load_settings


def settings_from_fields(values: Mapping[str, object]) -> GeneratorSettings | tuple[str, ...]:
    """Valid settings, or one ERROR: line per wrong field."""
    try:
        return GeneratorSettings.model_validate(dict(values))
    except ValidationError as error:
        return describe_validation_error(error, GENERATOR_SETTINGS_LABELS)


def add_loras(
    entries: tuple[LoraEntry, ...], files: Iterable[str], trigger_for: Callable[[str], str]
) -> tuple[tuple[LoraEntry, ...], tuple[str, ...]]:
    """The list with the files added (each with the trigger found in it), and a line for every
    file left out: INFO: when already in the list (as the window, no doubles), WARN: when full."""
    added = list(entries)
    warnings: list[str] = []
    for file in files:
        if any(entry.path == file for entry in added):
            warnings.append(f"INFO: {file} is already in the list.")
            continue
        if len(added) >= MAX_LORAS:
            warnings.append(f"WARN: At most {MAX_LORAS} LoRAs: {file} was not added.")
            continue
        added.append(LoraEntry(path=file, trigger=trigger_for(file)))
    return tuple(added), tuple(warnings)


def image_defaults() -> dict[str, object]:
    """What Reset to defaults writes, the same fields as the window's: prompts, LoRAs, model and
    folder stay."""
    defaults = GeneratorSettings().model_dump()
    return {name: defaults[name] for name in GENERATOR_RESET_FIELDS}


def gpu_index_for(config_file: Path) -> str:
    """The GPU chosen in the window; the default one when nothing valid is saved."""
    return load_settings(config_file).settings.gpu_index
