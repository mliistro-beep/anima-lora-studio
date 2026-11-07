"""LoRA and checkpoint files for the generator, seen from their header (no weights, no torch).

Two LoRA formats can be applied: kohya (`lora_unet_<module path with underscores>.lora_down
.weight`, `.lora_up.weight`, `.alpha`), the one sd-scripts writes and loads, and PEFT
(`diffusion_model.<module path>.lora_A.weight`, `.lora_B.weight`, maybe `.alpha`), which
ai-toolkit writes and generate.py renames to kohya in memory. `lora_te_*` keys are the Anima text
encoder part, which sd-scripts' inference does not apply. Other LoRA variants (LoHa, LoKr, DoRA,
LoCon, OFT, full differences) are refused. The rules are those Anima Merger Studio found on 293
real files (its DOCS/anima-format.md). `llm_adapter.blocks` are not transformer blocks.

Only safetensors_header is imported, so the two modules can be copied together.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from anima_trainer.preview.safetensors_header import (
    SafetensorsHeaderError,
    read_metadata,
    read_tensor_names,
)

# Transformer blocks of the Anima generations: a LoRA reaching exactly one of these counts was
# probably made for that generation.
GENERATION_BLOCKS = (28, 40, 52)

KOHYA_DIT_PREFIX = "lora_unet_"
KOHYA_TEXT_ENCODER_PREFIX = "lora_te_"
PEFT_DIT_PREFIX = "diffusion_model."
KOHYA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".alpha")
PEFT_SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".alpha")
CHECKPOINT_PREFIXES = ("model.diffusion_model.", "net.")

# The modules of Anima's DiT that LoRAs change: attention and MLP of each block, its adaLN
# linears, and the LLM adapter. Other models share the prefixes (SDXL and Flux kohya files start
# with lora_unet_, Z-Image and Krea2 PEFT files with diffusion_model., Krea2 even has blocks.N),
# so the module names are what tells an Anima LoRA apart.
_ANIMA_MODULE = (
    r"blocks{sep}\d{{1,6}}{sep}(?:(?:self_attn|cross_attn){sep}(?:q_proj|k_proj|v_proj|output_proj)"
    r"|mlp{sep}layer[12]|adaln_modulation_(?:self_attn|cross_attn|mlp){sep}\d+)"
    r"|llm_adapter{sep}.+"
)
_KOHYA_ANIMA = re.compile(
    r"^lora_unet_(?:"
    + _ANIMA_MODULE.format(sep="_")
    + r")\.(?:lora_down\.weight|lora_up\.weight|alpha)$"
)
_PEFT_ANIMA = re.compile(
    r"^diffusion_model\.(?:"
    + _ANIMA_MODULE.format(sep=r"\.")
    + r")\.(?:lora_A\.weight|lora_B\.weight|alpha)$"
)
_KOHYA_BLOCK = re.compile(r"^lora_unet_blocks_(\d{1,6})_")
_PEFT_BLOCK = re.compile(r"^diffusion_model\.blocks\.(\d{1,6})\.")
_CHECKPOINT_BLOCK = re.compile(r"^blocks\.(\d{1,6})\.")
# Parts of a key that mark a variant which is not a plain low-rank pair.
_VARIANTS = (
    (".hada_", "LoHa"),
    (".lokr_", "LoKr"),
    (".dora_scale", "DoRA"),
    (".lora_magnitude_vector", "DoRA"),
    (".lora_mid.", "LoCon"),
    (".oft_", "OFT"),
)
_DIFFERENCE_SUFFIXES = (".diff", ".diff_b")


class LoraFormat(StrEnum):
    KOHYA = "kohya"
    PEFT = "PEFT"


class LoraFileError(Exception):
    """The file cannot be used as a LoRA; the message names it."""


class CheckpointFileError(Exception):
    """The file cannot be used as an Anima model; the message names it."""


@dataclass(frozen=True)
class LoraFile:
    path: Path
    format: LoraFormat
    blocks: tuple[int, ...]  # the transformer blocks it changes, ascending
    has_text_encoder: bool
    unused_keys: int  # keys that match nothing sd-scripts applies

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def highest_block(self) -> int | None:
        return self.blocks[-1] if self.blocks else None


def read_lora(path: Path) -> LoraFile:
    """Raises LoraFileError when the file cannot be read or applied."""
    try:
        keys = read_tensor_names(path)
    except SafetensorsHeaderError as error:
        raise LoraFileError(str(error)) from error
    _refuse_variants(path, keys)
    kohya = [key for key in keys if _KOHYA_ANIMA.match(key)]
    peft = [key for key in keys if _PEFT_ANIMA.match(key)]
    text_encoder = [key for key in keys if key.startswith(KOHYA_TEXT_ENCODER_PREFIX)]
    if kohya and peft:
        raise LoraFileError(f"{path.name} mixes the kohya and PEFT formats: it cannot be applied.")
    if not kohya and not peft:
        raise LoraFileError(_no_dit_part(path, keys))
    lora_format, dit_keys, pattern = (
        (LoraFormat.KOHYA, kohya, _KOHYA_BLOCK) if kohya else (LoraFormat.PEFT, peft, _PEFT_BLOCK)
    )
    return LoraFile(
        path=path,
        format=lora_format,
        blocks=_block_numbers(dit_keys, pattern),
        has_text_encoder=bool(text_encoder),
        unused_keys=len(keys) - len(dit_keys) - len(text_encoder),
    )


# sd-scripts writes the trigger given at training time here (Civitai LoRAs made with it too).
TRIGGER_METADATA_KEY = "modelspec.trigger_phrase"


def lora_trigger(path: Path) -> str:
    """The trigger saved in the LoRA's metadata; blank when there is none or the file cannot be
    read (adding a LoRA must not fail on this: the user can type the trigger)."""
    try:
        return read_metadata(path).get(TRIGGER_METADATA_KEY, "").strip()
    except SafetensorsHeaderError:
        return ""


def read_checkpoint_blocks(path: Path) -> int:
    """The number of transformer blocks of an Anima DiT file; raises CheckpointFileError."""
    try:
        keys = read_tensor_names(path)
    except SafetensorsHeaderError as error:
        raise CheckpointFileError(str(error)) from error
    if any(key.endswith(KOHYA_SUFFIXES[:2] + PEFT_SUFFIXES[:2]) for key in keys):
        raise CheckpointFileError(f"{path.name} is a LoRA, not a model: add it as a LoRA.")
    blocks = _block_numbers([_without_prefix(key) for key in keys], _CHECKPOINT_BLOCK)
    if not blocks:
        raise CheckpointFileError(
            f"{path.name} is not an Anima model: it has no transformer blocks."
        )
    missing = sorted(set(range(blocks[-1] + 1)) - set(blocks))
    if missing:
        raise CheckpointFileError(f"{path.name} is incomplete: block {missing[0]} is missing.")
    return len(blocks)


def check_lora(lora: LoraFile, model_blocks: int) -> tuple[str, ...]:
    """ERROR: / WARN: lines about applying `lora` to a model with `model_blocks` blocks."""
    highest = lora.highest_block
    if highest is not None and highest >= model_blocks:
        return (
            f"ERROR: {lora.name} changes block {highest}, but the model has {model_blocks} blocks: "
            "it was made for a larger model and cannot be applied.",
        )
    lines = []
    if highest is not None and highest + 1 in GENERATION_BLOCKS and highest + 1 != model_blocks:
        lines.append(
            f"WARN: {lora.name} seems made for a model with {highest + 1} blocks; on this model "
            f"with {model_blocks} its effect is weak or wrong: remap it to {model_blocks} blocks "
            "first."
        )
    if lora.has_text_encoder:
        lines.append(
            f"WARN: The text encoder part of {lora.name} is ignored: only its image model part "
            "is applied."
        )
    if lora.unused_keys:
        lines.append(
            f"WARN: {lora.unused_keys} keys of {lora.name} match nothing in Anima and are ignored."
        )
    return tuple(lines)


def _refuse_variants(path: Path, keys: tuple[str, ...]) -> None:
    for key in keys:
        for marker, kind in _VARIANTS:
            if marker in key:
                raise LoraFileError(
                    f"{path.name} is a {kind} file: only plain LoRAs (kohya or PEFT) can be "
                    "applied."
                )
        if key.endswith(_DIFFERENCE_SUFFIXES):
            raise LoraFileError(
                f"{path.name} holds a full weight difference (.diff): only plain LoRAs can "
                "be applied."
            )


def _no_dit_part(path: Path, keys: tuple[str, ...]) -> str:
    if not any(key.endswith(KOHYA_SUFFIXES + PEFT_SUFFIXES) for key in keys):
        return f"{path.name} holds no LoRA weights: is it a model or another kind of file?"
    if any(key.startswith(KOHYA_TEXT_ENCODER_PREFIX) for key in keys):
        return (
            f"{path.name} changes only the text encoder, which the generator does not apply: "
            "it would have no effect."
        )
    return f"{path.name} is not an Anima LoRA: none of its keys belongs to the Anima image model."


def _block_numbers(keys: list[str], pattern: re.Pattern[str]) -> tuple[int, ...]:
    numbers = {int(match.group(1)) for key in keys if (match := pattern.match(key))}
    return tuple(sorted(numbers))


def _without_prefix(key: str) -> str:
    for prefix in CHECKPOINT_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key
