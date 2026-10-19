"""Allowed values for the enumerated training settings, identical to Citron's dropdowns."""

from typing import Literal, get_args

BaseModelName = Literal["anima-base-v1.0", "anima-preview3-base", "anima-preview"]
OptimizerType = Literal["AdamW8bit", "AdamW", "Lion", "SGD", "Prodigy"]
LrScheduler = Literal[
    "cosine_with_restarts",
    "cosine",
    "linear",
    "constant",
    "constant_with_warmup",
    "polynomial",
]
MixedPrecision = Literal["bf16", "fp16", "no"]
TimestepSampling = Literal["sigmoid", "uniform", "logit_normal"]

BASE_MODELS: tuple[str, ...] = get_args(BaseModelName)
OPTIMIZERS: tuple[str, ...] = get_args(OptimizerType)
LR_SCHEDULERS: tuple[str, ...] = get_args(LrScheduler)
MIXED_PRECISIONS: tuple[str, ...] = get_args(MixedPrecision)
TIMESTEP_SAMPLINGS: tuple[str, ...] = get_args(TimestepSampling)
