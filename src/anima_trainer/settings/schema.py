"""Validated, immutable training settings (the desktop equivalent of Citron's settings dict).

Field order matches Citron's DEFAULTS so that config.json is written with the same key order.
Constraints mirror the minimum/maximum values of Citron's Gradio inputs.
"""

import re
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field

from anima_trainer.settings.choices import (
    BaseModelName,
    LrScheduler,
    MixedPrecision,
    OptimizerType,
    TimestepSampling,
)
from anima_trainer.settings.defaults import DEFAULTS

# Becomes a file name (TOML, log) and the LoRA output name, so no path components allowed.
PROJECT_NAME_PATTERN = re.compile(r"^[\w.-]+$")
GPU_INDEX_PATTERN = r"^\d+(,\d+)*$"

REQUIRED_TEXT_FIELDS = (
    ("project_name", "Project name"),
    ("image_directory", "Image directory"),
    ("output_directory", "Output directory"),
)


def _reject_bool(value: Any) -> Any:
    # Lax mode would turn True/False into 1/0 (or 1.0/0.0); a checkbox value in a number
    # field is a bug.
    if isinstance(value, bool):
        raise ValueError("a number is required, not a boolean")
    return value


def _gpu_index_to_str(value: Any) -> Any:
    # Citron stores the GPU index as a string, but older files may hold an int.
    if isinstance(value, bool):
        raise ValueError("a GPU index is required, not a boolean")
    return str(value) if isinstance(value, int) else value


def _check_project_name(value: str) -> str:
    # Blank is allowed here and reported by missing_required_fields(), as in Citron.
    if value.strip() and (value in {".", ".."} or not PROJECT_NAME_PATTERN.fullmatch(value)):
        raise ValueError("use only letters, digits, '.', '_' and '-'")
    return value


WholeNumber = Annotated[int, BeforeValidator(_reject_bool)]
DecimalNumber = Annotated[float, BeforeValidator(_reject_bool)]
GpuIndex = Annotated[str, BeforeValidator(_gpu_index_to_str), Field(pattern=GPU_INDEX_PATTERN)]
ProjectName = Annotated[str, AfterValidator(_check_project_name)]


class TrainingSettings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="ignore", validate_default=True, allow_inf_nan=False
    )

    # Basic
    project_name: ProjectName = DEFAULTS["project_name"]
    base_model: BaseModelName = DEFAULTS["base_model"]
    image_directory: str = DEFAULTS["image_directory"]
    output_directory: str = DEFAULTS["output_directory"]
    network_dim: WholeNumber = Field(DEFAULTS["network_dim"], ge=1)
    network_alpha: WholeNumber = Field(DEFAULTS["network_alpha"], ge=1)
    learning_rate: DecimalNumber = Field(DEFAULTS["learning_rate"], gt=0)
    max_train_epochs: WholeNumber = Field(DEFAULTS["max_train_epochs"], ge=1)
    resolution: WholeNumber = Field(DEFAULTS["resolution"], ge=64)
    repeats: WholeNumber = Field(DEFAULTS["repeats"], ge=1)
    caption_dropout: DecimalNumber = Field(DEFAULTS["caption_dropout"], ge=0.0, le=1.0)
    gpu_index: GpuIndex = DEFAULTS["gpu_index"]
    # Advanced
    optimizer_type: OptimizerType = DEFAULTS["optimizer_type"]
    lr_scheduler: LrScheduler = DEFAULTS["lr_scheduler"]
    lr_scheduler_num_cycles: WholeNumber = Field(DEFAULTS["lr_scheduler_num_cycles"], ge=1)
    lr_warmup_steps: WholeNumber = Field(DEFAULTS["lr_warmup_steps"], ge=0)
    train_batch_size: WholeNumber = Field(DEFAULTS["train_batch_size"], ge=1)
    gradient_accumulation_steps: WholeNumber = Field(DEFAULTS["gradient_accumulation_steps"], ge=1)
    max_grad_norm: DecimalNumber = Field(DEFAULTS["max_grad_norm"], ge=0)
    save_every_n_epochs: WholeNumber = Field(DEFAULTS["save_every_n_epochs"], ge=1)
    save_last_n_epochs: WholeNumber = Field(DEFAULTS["save_last_n_epochs"], ge=1)
    mixed_precision: MixedPrecision = DEFAULTS["mixed_precision"]
    gradient_checkpointing: bool = DEFAULTS["gradient_checkpointing"]
    seed: WholeNumber = DEFAULTS["seed"]
    noise_offset: DecimalNumber = Field(DEFAULTS["noise_offset"], ge=0)
    multires_noise_discount: DecimalNumber = Field(DEFAULTS["multires_noise_discount"], ge=0, le=1)
    timestep_sampling: TimestepSampling = DEFAULTS["timestep_sampling"]
    discrete_flow_shift: DecimalNumber = Field(DEFAULTS["discrete_flow_shift"], gt=0)
    cache_latents: bool = DEFAULTS["cache_latents"]
    cache_text_encoder_outputs: bool = DEFAULTS["cache_text_encoder_outputs"]
    vae_chunk_size: WholeNumber = Field(DEFAULTS["vae_chunk_size"], ge=1)
    vae_disable_cache: bool = DEFAULTS["vae_disable_cache"]
    num_cpu_threads_per_process: WholeNumber = Field(DEFAULTS["num_cpu_threads_per_process"], ge=1)
    # Internal
    last_train_config: str = DEFAULTS["last_train_config"]
    last_dataset_config: str = DEFAULTS["last_dataset_config"]

    def with_changes(self, **changes: Any) -> Self:
        """Return a new, re-validated copy with the given fields replaced."""
        return type(self).model_validate({**self.model_dump(), **changes})

    def missing_required_fields(self) -> tuple[str, ...]:
        """Labels of the text fields Citron refuses to configure with when blank."""
        return tuple(
            label for name, label in REQUIRED_TEXT_FIELDS if not getattr(self, name).strip()
        )
