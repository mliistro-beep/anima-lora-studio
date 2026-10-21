"""How each setting appears in the form: label, widget kind, limits and group.

Labels, groups and limits are Citron's (app.py, build_ui) so both UIs read the same.
Floats are edited as text: a spin box would round values such as 0.0001 and break TOML parity.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from anima_trainer.settings.choices import (
    BASE_MODELS,
    LR_SCHEDULERS,
    MIXED_PRECISIONS,
    OPTIMIZERS,
    TIMESTEP_SAMPLINGS,
)

FieldKind = Literal["text", "folder", "choice", "gpu", "int", "float", "fraction", "bool"]

# QSpinBox holds a C int.
INT_MAX = 2**31 - 1
INT_MIN = -(2**31)

INTERNAL_FIELDS = ("last_train_config", "last_dataset_config")
VALUE_ERROR_PREFIX = "Value error, "


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: FieldKind
    group: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    placeholder: str = ""
    tooltip: str = ""


def _int(name: str, label: str, group: str, minimum: int | None = 1) -> FieldSpec:
    low = INT_MIN if minimum is None else minimum
    return FieldSpec(name, label, "int", group, minimum=low, maximum=INT_MAX)


_PATHS = "Project & Paths"
_NETWORK = "Network"
_DATASET = "Dataset"

BASIC_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("project_name", "Project Name", "text", _PATHS, placeholder="my_lora"),
    FieldSpec("gpu_index", "GPU", "gpu", _PATHS),
    FieldSpec(
        "base_model",
        "Base Model",
        "choice",
        _PATHS,
        choices=BASE_MODELS,
        tooltip="The model is downloaded when you start training, if it is missing.",
    ),
    FieldSpec(
        "image_directory",
        "Image Directory (flat folder with images + .txt captions)",
        "folder",
        _PATHS,
        placeholder="/path/to/my_dataset",
    ),
    FieldSpec(
        "output_directory",
        "Output Directory (where trained LoRA is saved)",
        "folder",
        _PATHS,
        placeholder="/path/to/output",
    ),
    _int("network_dim", "Network Dim", _NETWORK),
    _int("network_alpha", "Network Alpha", _NETWORK),
    FieldSpec("learning_rate", "Learning Rate", "float", _NETWORK),
    _int("max_train_epochs", "Max Epochs", _NETWORK),
    _int("resolution", "Resolution (px)", _DATASET, minimum=64),
    _int("repeats", "Repeats", _DATASET),
    FieldSpec(
        "caption_dropout",
        "Caption Dropout",
        "fraction",
        _DATASET,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
    ),  # fmt: skip
)

_OPTIMIZER = "Optimizer & Scheduler"
_BATCH = "Batch & Gradient"
_SAVING = "Saving"
_PRECISION = "Precision & Memory"
_NOISE = "Noise & Flow"
_MISC = "Misc"

ADVANCED_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("optimizer_type", "Optimizer", "choice", _OPTIMIZER, choices=OPTIMIZERS),
    FieldSpec("lr_scheduler", "LR Scheduler", "choice", _OPTIMIZER, choices=LR_SCHEDULERS),
    _int("lr_scheduler_num_cycles", "LR Scheduler Num Cycles", _OPTIMIZER),
    _int("lr_warmup_steps", "LR Warmup Steps", _OPTIMIZER, minimum=0),
    _int("train_batch_size", "Train Batch Size", _BATCH),
    _int("gradient_accumulation_steps", "Gradient Accumulation Steps", _BATCH),
    FieldSpec("max_grad_norm", "Max Grad Norm", "float", _BATCH),
    _int("save_every_n_epochs", "Save Every N Epochs", _SAVING),
    _int("save_last_n_epochs", "Keep Last N Checkpoints", _SAVING),
    FieldSpec("mixed_precision", "Mixed Precision", "choice", _PRECISION, choices=MIXED_PRECISIONS),
    _int("vae_chunk_size", "VAE Chunk Size", _PRECISION),
    FieldSpec("gradient_checkpointing", "Gradient Checkpointing", "bool", _PRECISION),
    FieldSpec("cache_latents", "Cache Latents", "bool", _PRECISION),
    FieldSpec("cache_text_encoder_outputs", "Cache Text Encoder Outputs", "bool", _PRECISION),
    FieldSpec("vae_disable_cache", "VAE Disable Cache", "bool", _PRECISION),
    FieldSpec("noise_offset", "Noise Offset", "float", _NOISE),
    FieldSpec("multires_noise_discount", "Multires Noise Discount", "float", _NOISE),
    FieldSpec(
        "timestep_sampling", "Timestep Sampling", "choice", _NOISE, choices=TIMESTEP_SAMPLINGS
    ),
    FieldSpec("discrete_flow_shift", "Discrete Flow Shift", "float", _NOISE),
    _int("seed", "Seed", _MISC, minimum=None),
    _int("num_cpu_threads_per_process", "CPU Threads Per Process", _MISC),
)

_LABELS = {spec.name: spec.label for spec in (*BASIC_FIELDS, *ADVANCED_FIELDS)}


def field_label(name: str) -> str:
    return _LABELS.get(name, name)


def describe_validation_error(error: ValidationError) -> tuple[str, ...]:
    """One "ERROR: <label>: <reason>" line per invalid field, in the error's order."""
    return tuple(_describe(detail) for detail in error.errors())


def _describe(detail: ErrorDetails) -> str:
    location = detail["loc"]
    field = field_label(str(location[0])) if location else "Settings"
    nested = "".join(f".{part}" for part in location[1:])
    # Custom validators raise ValueError, which pydantic prefixes with "Value error, ".
    reason = detail["msg"].removeprefix(VALUE_ERROR_PREFIX)
    return f"ERROR: {field}{nested}: {reason}"
