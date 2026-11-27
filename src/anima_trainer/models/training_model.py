"""A DiT file of the user's (an Anima finetune or merge) as the model to train a LoRA on.

There is no known SHA-256 for such a file, so it is checked from its header instead (no
weights, no torch) against the tensors sd-scripts expects: load_anima_model builds the DiT with
a fixed 28-block config and raises on any missing or unexpected tensor, but only once the
training has started. Here the same problems become a clear error before anything runs.
"""

import re
from pathlib import Path

from anima_trainer.models.anima_dit_layout import ANIMA_BLOCKS, anima_dit_tensors
from anima_trainer.preview.lora_files import CHECKPOINT_PREFIXES
from anima_trainer.preview.safetensors_header import (
    SafetensorsHeaderError,
    TensorInfo,
    read_tensors,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings

TRAINABLE_BLOCKS = ANIMA_BLOCKS
TRAINABLE_DTYPES = frozenset({"BF16", "F16", "F32"})
MAX_LISTED = 3
_BLOCK = re.compile(r"^blocks\.(\d{1,6})\.")
_LORA_MARKERS = ("lora_", ".alpha", "lora_A.", "lora_B.")


class TrainingModelRejected(Exception):
    """The file cannot be trained on; the message names it and says why."""


def training_dit(settings: TrainingSettings, paths: RuntimePaths) -> Path:
    """The DiT the training uses: the user's file when set, else the chosen base model.

    A relative file is read from the data folder, where sd-scripts runs.
    """
    if settings.model_file:
        return paths.resolve(settings.model_file)
    return paths.dit_model(settings.base_model)


def check_training_model(path: Path) -> str:
    """A one-line description of an accepted file; raises TrainingModelRejected."""
    tensors = _read(path)
    if any(marker in name for name in tensors for marker in _LORA_MARKERS):
        raise TrainingModelRejected(f"{path.name} is a LoRA, not a model.")
    found = {_without_prefix(name): info for name, info in tensors.items()}
    blocks = sorted({int(match[1]) for name in found if (match := _BLOCK.match(name))})
    if not blocks:
        raise TrainingModelRejected(
            f"{path.name} is not an Anima model: it has no transformer blocks."
        )
    _check_blocks(path, blocks)  # first: a larger Anima also has parts of its own
    _check_foreign_parts(path, found)
    _check_prefixes(path, tuple(tensors))
    used = sorted({info.dtype for info in found.values()})
    unusable = [dtype for dtype in used if dtype not in TRAINABLE_DTYPES]
    if unusable:
        raise TrainingModelRejected(
            f"{path.name} is stored as {unusable[0]}: training needs a BF16, F16 or F32 model."
        )
    _check_tensors(path, found)
    return f"{path.name}: an Anima model with {TRAINABLE_BLOCKS} blocks ({'/'.join(used)})"


def _read(path: Path) -> dict[str, TensorInfo]:
    if not path.exists():
        raise TrainingModelRejected(f"Model file not found: {path}")
    if not path.is_file():
        raise TrainingModelRejected(f"{path} is not a file.")
    try:
        return read_tensors(path)
    except SafetensorsHeaderError as error:
        raise TrainingModelRejected(str(error)) from error


def _check_prefixes(path: Path, names: tuple[str, ...]) -> None:
    # sd-scripts strips the prefix of each key on its own; one file with two is refused anyway.
    prefixes = {_prefix(name) for name in names}
    if len(prefixes) > 1:
        shown = ", ".join(sorted(prefix or "none" for prefix in prefixes))
        raise TrainingModelRejected(
            f"{path.name} mixes key prefixes ({shown}): it cannot be loaded."
        )


def _check_foreign_parts(path: Path, found: dict[str, TensorInfo]) -> None:
    known = {name.split(".")[0] for name in anima_dit_tensors()}
    foreign = sorted({name.split(".")[0] for name in found} - known)
    if foreign:
        raise TrainingModelRejected(
            f"{path.name} has parts sd-scripts cannot load ({_listed(foreign)}): "
            "choose a file with the Anima DiT only."
        )


def _check_blocks(path: Path, blocks: list[int]) -> None:
    if blocks[-1] + 1 != TRAINABLE_BLOCKS:
        raise TrainingModelRejected(
            f"{path.name} has {blocks[-1] + 1} blocks: sd-scripts trains only the Anima model "
            f"with {TRAINABLE_BLOCKS} blocks (Anima 2B and its finetunes)."
        )
    gaps = sorted(set(range(TRAINABLE_BLOCKS)) - set(blocks))
    if gaps:
        raise TrainingModelRejected(f"{path.name} is incomplete: block {gaps[0]} is missing.")


def _check_tensors(path: Path, found: dict[str, TensorInfo]) -> None:
    expected = anima_dit_tensors()
    missing = sorted(set(expected) - set(found))
    if missing:
        raise TrainingModelRejected(
            f"{path.name} has tensors missing that sd-scripts needs: {_listed(missing)}."
        )
    unexpected = sorted(set(found) - set(expected))
    if unexpected:
        raise TrainingModelRejected(
            f"{path.name} has unexpected tensors sd-scripts refuses: {_listed(unexpected)}."
        )
    resized = sorted(name for name, shape in expected.items() if found[name].shape != shape)
    if resized:
        name = resized[0]
        raise TrainingModelRejected(
            f"{path.name} has {name} of shape {list(found[name].shape)}, "
            f"not {list(expected[name])}: it is not an Anima 2B model."
        )


def _listed(names: list[str]) -> str:
    more = len(names) - MAX_LISTED
    return ", ".join(names[:MAX_LISTED]) + (f" and {more} more" if more > 0 else "")


def _prefix(name: str) -> str:
    return next((prefix for prefix in CHECKPOINT_PREFIXES if name.startswith(prefix)), "")


def _without_prefix(name: str) -> str:
    return name.removeprefix(_prefix(name))
