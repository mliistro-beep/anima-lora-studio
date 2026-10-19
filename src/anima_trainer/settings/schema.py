"""Validated, immutable training settings (the desktop equivalent of Citron's settings dict).

Field order matches Citron's DEFAULTS so that config.json is written with the same key order.
Constraints mirror the minimum/maximum values of Citron's Gradio inputs.
"""

from typing import Annotated, Any, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from anima_trainer.settings.choices import (
    BaseModelName,
    LrScheduler,
    MixedPrecision,
    OptimizerType,
    TimestepSampling,
)
from anima_trainer.settings.defaults import DEFAULTS

# Citron stores the GPU index as a string, but older files may hold an int.
GpuIndex = Annotated[str, BeforeValidator(lambda value: str(value))]

REQUIRED_TEXT_FIELDS = (
    ("project_name", "Project name"),
    ("image_directory", "Image directory"),
    ("output_directory", "Output directory"),
)


class TrainingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", validate_default=True)

    # Basic
    project_name: str = DEFAULTS["project_name"]
    base_model: BaseModelName = DEFAULTS["base_model"]
    image_directory: str = DEFAULTS["image_directory"]
    output_directory: str = DEFAULTS["output_directory"]
    network_dim: int = Field(DEFAULTS["network_dim"], ge=1)
    network_alpha: int = Field(DEFAULTS["network_alpha"], ge=1)
    learning_rate: float = Field(DEFAULTS["learning_rate"], gt=0)
    max_train_epochs: int = Field(DEFAULTS["max_train_epochs"], ge=1)
    resolution: int = Field(DEFAULTS["resolution"], ge=64)
    repeats: int = Field(DEFAULTS["repeats"], ge=1)
    caption_dropout: float = Field(DEFAULTS["caption_dropout"], ge=0.0, le=1.0)
    gpu_index: GpuIndex = DEFAULTS["gpu_index"]
    # Advanced
    optimizer_type: OptimizerType = DEFAULTS["optimizer_type"]
    lr_scheduler: LrScheduler = DEFAULTS["lr_scheduler"]
    lr_scheduler_num_cycles: int = Field(DEFAULTS["lr_scheduler_num_cycles"], ge=1)
    lr_warmup_steps: int = Field(DEFAULTS["lr_warmup_steps"], ge=0)
    train_batch_size: int = Field(DEFAULTS["train_batch_size"], ge=1)
    gradient_accumulation_steps: int = Field(DEFAULTS["gradient_accumulation_steps"], ge=1)
    max_grad_norm: float = DEFAULTS["max_grad_norm"]
    save_every_n_epochs: int = Field(DEFAULTS["save_every_n_epochs"], ge=1)
    save_last_n_epochs: int = Field(DEFAULTS["save_last_n_epochs"], ge=1)
    mixed_precision: MixedPrecision = DEFAULTS["mixed_precision"]
    gradient_checkpointing: bool = DEFAULTS["gradient_checkpointing"]
    seed: int = DEFAULTS["seed"]
    noise_offset: float = DEFAULTS["noise_offset"]
    multires_noise_discount: float = DEFAULTS["multires_noise_discount"]
    timestep_sampling: TimestepSampling = DEFAULTS["timestep_sampling"]
    discrete_flow_shift: float = DEFAULTS["discrete_flow_shift"]
    cache_latents: bool = DEFAULTS["cache_latents"]
    cache_text_encoder_outputs: bool = DEFAULTS["cache_text_encoder_outputs"]
    vae_chunk_size: int = Field(DEFAULTS["vae_chunk_size"], ge=1)
    vae_disable_cache: bool = DEFAULTS["vae_disable_cache"]
    num_cpu_threads_per_process: int = Field(DEFAULTS["num_cpu_threads_per_process"], ge=1)
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
