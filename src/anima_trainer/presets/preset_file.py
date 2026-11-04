"""The preset file format: TOML with `name`, `description` and a `[settings]` table.

A preset holds only the values that describe a kind of training. What is chosen for each training
or belongs to one machine is refused, so a preset means the same thing to whoever receives it.
"""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError

from anima_trainer.settings.schema import TrainingSettings

PRESET_FIELDS = frozenset(
    {
        "base_model",
        "network_dim",
        "network_alpha",
        "learning_rate",
        "max_train_epochs",
        "caption_dropout",
        "optimizer_type",
        "lr_scheduler",
        "lr_scheduler_num_cycles",
        "lr_warmup_steps",
        "train_batch_size",
        "gradient_accumulation_steps",
        "max_grad_norm",
        "save_every_n_epochs",
        "save_last_n_epochs",
        "mixed_precision",
        "gradient_checkpointing",
        "seed",
        "noise_offset",
        "multires_noise_discount",
        "timestep_sampling",
        "discrete_flow_shift",
    }
)
PER_TRAINING_FIELDS = frozenset(
    {"project_name", "image_directory", "output_directory", "repeats", "resolution"}
)
MACHINE_FIELDS = frozenset(
    {
        "gpu_index",
        "num_cpu_threads_per_process",
        "cache_latents",
        "cache_text_encoder_outputs",
        "vae_chunk_size",
        "vae_disable_cache",
    }
)
APP_FIELDS = frozenset({"last_train_config", "last_dataset_config"})
KEPT_FIELDS = PER_TRAINING_FIELDS | MACHINE_FIELDS | APP_FIELDS


class PresetError(Exception):
    """A preset file that cannot be used; the message says why."""


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    values: Mapping[str, Any]
    builtin: bool = False


def _showable(value: str) -> str:
    # Shown in a terminal: a control character could move the cursor or fake other output.
    if not value.isprintable():
        raise ValueError("contains control characters")
    return value.strip()


def _not_blank(value: str) -> str:
    if not value:
        raise ValueError("cannot be empty")
    return value


class _PresetFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, AfterValidator(_showable), AfterValidator(_not_blank)]
    description: Annotated[str, AfterValidator(_showable)] = ""
    settings: dict[str, Any] = {}


def parse_preset(text: str, builtin: bool = False) -> Preset:
    """The preset in `text`; raises PresetError naming what is wrong."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise PresetError(f"not valid TOML: {error}") from error
    except RecursionError as error:
        raise PresetError("not valid TOML: nested too deeply") from error
    try:
        parsed = _PresetFile.model_validate(raw)
    except ValidationError as error:
        raise PresetError(_describe(error)) from error
    values = _checked_values(parsed.settings)
    return Preset(parsed.name, parsed.description, MappingProxyType(values), builtin)


def _checked_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """The values as the settings understand them (a "32" written for a whole number becomes 32)."""
    problems = [problem for key in values if (problem := _field_problem(key)) is not None]
    if problems:
        raise PresetError("; ".join(problems))
    try:
        checked = TrainingSettings().with_changes(**values)
    except ValidationError as error:
        raise PresetError(_describe(error)) from error
    return {key: getattr(checked, key) for key in values}


def _field_problem(key: str) -> str | None:
    if key in PRESET_FIELDS:
        return None
    if key in PER_TRAINING_FIELDS:
        return f"{key} is chosen for each training"
    if key in MACHINE_FIELDS:
        return f"{key} belongs to this machine"
    if key in APP_FIELDS:
        return f"{key} is set by the app"
    return f"{key} is not a setting"


def _describe(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
        for detail in error.errors()
    )
