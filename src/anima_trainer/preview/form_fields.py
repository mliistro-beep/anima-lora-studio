"""How the preview and generator settings appear in their forms (prompts have their own
multi-line boxes)."""

from anima_trainer.preview.settings import MAX_SIDE, MIN_SIDE, SIDE_STEP
from anima_trainer.preview.vram_profile import GENERATOR_SIDE_STEP
from anima_trainer.settings.field_specs import INT_MAX, FieldSpec

SAMPLERS = ("er_sde", "euler")
SAMPLER_TIP = (
    "ER-SDE: the Anima author's default, good images in fewer steps. Euler: sd-scripts' own."
)
_PREVIEWS = "Previews during training"
_IMAGE = "Image"


def _side(name: str, label: str, group: str, step: int) -> FieldSpec:
    return FieldSpec(name, label, "int", group, minimum=MIN_SIDE, maximum=MAX_SIDE, step=step)


def _sampling(group: str, side_step: int) -> tuple[FieldSpec, ...]:
    return (
        _side("width", "Width", group, side_step),
        _side("height", "Height", group, side_step),
        FieldSpec("sampler", "Sampler", "choice", group, choices=SAMPLERS, tooltip=SAMPLER_TIP),
        FieldSpec("steps", "Steps", "int", group, minimum=1, maximum=200),
        FieldSpec("cfg_scale", "CFG Scale", "float", group),
        FieldSpec("flow_shift", "Flow Shift", "float", group),
        FieldSpec("seed", "Seed", "int", group, minimum=0, maximum=INT_MAX),
    )  # fmt: skip


PREVIEW_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("enabled", "Generate previews", "bool", _PREVIEWS),
    FieldSpec("every_n_epochs", "Every N epochs", "int", _PREVIEWS, minimum=1, maximum=INT_MAX),
    FieldSpec("sample_at_first", "Also before training", "bool", _PREVIEWS),
    FieldSpec(
        "auto_size", "Size from GPU memory", "bool", _PREVIEWS,
        tooltip="Smaller previews on GPUs with less memory, so sampling fits next to the training.",
    ),
    FieldSpec("negative_prompt", "Negative Prompt", "text", _PREVIEWS),
    *_sampling(_PREVIEWS, SIDE_STEP),
)  # fmt: skip

GENERATOR_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("negative_prompt", "Negative Prompt", "text", _IMAGE),
    FieldSpec(
        "lora_multiplier", "Checkpoint Strength", "float", _IMAGE,
        tooltip="Strength of the project checkpoint, 0 to 2; 1 is the trained strength.",
    ),
    *_sampling(_IMAGE, GENERATOR_SIDE_STEP),
)  # fmt: skip

PREVIEW_LABELS = {spec.name: spec.label for spec in PREVIEW_FIELDS} | {"prompts": "Prompts"}
GENERATOR_LABELS = {spec.name: spec.label for spec in GENERATOR_FIELDS} | {"prompt": "Prompt"}
