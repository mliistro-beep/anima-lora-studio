"""Default settings, copied verbatim from Citron's DEFAULTS (app.py) in the same key order."""

from types import MappingProxyType
from typing import ReadOnly, TypedDict, cast

from anima_trainer.settings.choices import (
    BaseModelName,
    LrScheduler,
    MixedPrecision,
    OptimizerType,
    TimestepSampling,
)


class CitronDefaults(TypedDict):
    """Typed view of DEFAULTS so each default keeps its own type for the schema."""

    project_name: ReadOnly[str]
    base_model: ReadOnly[BaseModelName]
    image_directory: ReadOnly[str]
    output_directory: ReadOnly[str]
    network_dim: ReadOnly[int]
    network_alpha: ReadOnly[int]
    learning_rate: ReadOnly[float]
    max_train_epochs: ReadOnly[int]
    resolution: ReadOnly[int]
    repeats: ReadOnly[int]
    caption_dropout: ReadOnly[float]
    gpu_index: ReadOnly[str]
    optimizer_type: ReadOnly[OptimizerType]
    lr_scheduler: ReadOnly[LrScheduler]
    lr_scheduler_num_cycles: ReadOnly[int]
    lr_warmup_steps: ReadOnly[int]
    train_batch_size: ReadOnly[int]
    gradient_accumulation_steps: ReadOnly[int]
    max_grad_norm: ReadOnly[float]
    save_every_n_epochs: ReadOnly[int]
    save_last_n_epochs: ReadOnly[int]
    mixed_precision: ReadOnly[MixedPrecision]
    gradient_checkpointing: ReadOnly[bool]
    seed: ReadOnly[int]
    noise_offset: ReadOnly[float]
    multires_noise_discount: ReadOnly[float]
    timestep_sampling: ReadOnly[TimestepSampling]
    discrete_flow_shift: ReadOnly[float]
    cache_latents: ReadOnly[bool]
    cache_text_encoder_outputs: ReadOnly[bool]
    vae_chunk_size: ReadOnly[int]
    vae_disable_cache: ReadOnly[bool]
    num_cpu_threads_per_process: ReadOnly[int]
    last_train_config: ReadOnly[str]
    last_dataset_config: ReadOnly[str]


_VALUES: CitronDefaults = {
    # Basic
    "project_name": "my_lora",
    "base_model": "anima-base-v1.0",
    "image_directory": "",
    "output_directory": "",
    "network_dim": 20,
    "network_alpha": 20,
    "learning_rate": 0.0001,
    "max_train_epochs": 10,
    "resolution": 768,
    "repeats": 10,
    "caption_dropout": 0.1,
    "gpu_index": "0",
    # Advanced
    "optimizer_type": "AdamW8bit",
    "lr_scheduler": "cosine_with_restarts",
    "lr_scheduler_num_cycles": 1,
    "lr_warmup_steps": 100,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "save_every_n_epochs": 1,
    "save_last_n_epochs": 4,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "seed": 42,
    "noise_offset": 0.03,
    "multires_noise_discount": 0.3,
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "cache_latents": True,
    "cache_text_encoder_outputs": True,
    "vae_chunk_size": 64,
    "vae_disable_cache": True,
    "num_cpu_threads_per_process": 1,
    # Internal
    "last_train_config": "",
    "last_dataset_config": "",
}

# Read-only at runtime too; the cast keeps the per-key types for static checking.
DEFAULTS = cast(CitronDefaults, MappingProxyType(_VALUES))
