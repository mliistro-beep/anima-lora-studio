"""LoRA key names for sd-scripts' inference: PEFT keys renamed to kohya, notes about the rest.

sd-scripts applies a LoRA module when it finds `lora_unet_<model weight name without .weight,
dots as underscores>.lora_down.weight` and `.lora_up.weight` (and uses `.alpha` when present,
else a scale of 1). PEFT files (ai-toolkit) name the same tensors `diffusion_model.<module
path>.lora_A.weight` / `.lora_B.weight` (and maybe `.alpha`), which sd-scripts drops without a
word: renaming them is enough, the scale stays the one ComfyUI and Krita apply. The `lora_te_*`
text encoder part is not applied by sd-scripts' inference.

Runs in the training venv next to generate.py and imports nothing of the app, which tests it
through anima_trainer.runtime_scripts.lora_keys.
"""

from collections.abc import Iterable

KOHYA_DIT_PREFIX = "lora_unet_"
KOHYA_TEXT_ENCODER_PREFIX = "lora_te_"
PEFT_DIT_PREFIX = "diffusion_model."
# PEFT suffix -> kohya suffix
SUFFIXES = (
    (".lora_A.weight", ".lora_down.weight"),
    (".lora_B.weight", ".lora_up.weight"),
    (".alpha", ".alpha"),
)

Note = tuple[str, str]  # (INFO or WARN, text)


def kohya_key(key: str) -> str | None:
    """The kohya name of a PEFT LoRA key, or None when `key` is not one."""
    if not key.startswith(PEFT_DIT_PREFIX):
        return None
    for peft_suffix, kohya_suffix in SUFFIXES:
        if key.endswith(peft_suffix):
            module = key[len(PEFT_DIT_PREFIX) : -len(peft_suffix)]
            return f"{KOHYA_DIT_PREFIX}{module.replace('.', '_')}{kohya_suffix}"
    return None


def kohya_names(keys: Iterable[str]) -> dict[str, str]:
    """PEFT key -> kohya key, for the PEFT keys of `keys` only."""
    return {key: name for key in keys if (name := kohya_key(key)) is not None}


def lora_notes(name: str, keys: Iterable[str]) -> tuple[Note, ...]:
    """What generate.py says about a LoRA file before applying it."""
    keys = list(keys)
    renamed = kohya_names(keys)
    applied = [key for key in keys if key in renamed or key.startswith(KOHYA_DIT_PREFIX)]
    text_encoder = [key for key in keys if key.startswith(KOHYA_TEXT_ENCODER_PREFIX)]
    unused = len(keys) - len(applied) - len(text_encoder)
    notes: list[Note] = []
    if renamed:
        notes.append(
            ("INFO", f"{name} is a PEFT LoRA: its keys are renamed for sd-scripts in memory.")
        )
    if text_encoder:
        notes.append(
            (
                "WARN",
                f"The text encoder part of {name} is ignored: only its image model part "
                "is applied.",
            )
        )
    if unused:
        verb = "matches" if unused == 1 else "match"
        noun, pronoun = ("key", "is") if unused == 1 else ("keys", "are")
        notes.append(
            ("WARN", f"{unused} {noun} of {name} {verb} nothing in Anima and {pronoun} ignored.")
        )
    return tuple(notes)
