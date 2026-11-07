"""The text saved in every generated PNG: `parameters` in the A1111 / Forge layout (read by
Civitai and image viewers) and a JSON of the entry, so the image can be made again later.

Only file names are saved, never folders: an image shared online must not show where the
user keeps their files. The app's name and version are a reminder, not loaded back.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from anima_trainer.preview.generator_settings import GenerationEntry, LoraEntry

if TYPE_CHECKING:
    from anima_trainer.preview.inference_command import LoraUse

APP_NAME = "Anima LoRA Studio"


@dataclass(frozen=True)
class ImageMetadata:
    parameters: str
    settings: str  # JSON: generator, version, entry


def image_metadata(
    entry: GenerationEntry, seed: int, checkpoint: "LoraUse | None", version: str
) -> ImageMetadata:
    """The metadata of one image of `entry`, made with `seed`; `checkpoint` is the project's
    checkpoint applied before the entry's LoRAs, if any."""
    loras = _used_loras(entry, checkpoint)
    return ImageMetadata(
        parameters=_parameters(entry, seed, loras, version),
        settings=_settings(entry, seed, loras, version),
    )


def _used_loras(entry: GenerationEntry, checkpoint: "LoraUse | None") -> tuple[LoraEntry, ...]:
    """The LoRAs applied to the image, by file name, the checkpoint first."""
    own = tuple(
        lora.model_copy(update={"path": Path(lora.path).name}) for lora in entry.enabled_loras()
    )
    if checkpoint is None:
        return own
    return (LoraEntry(path=checkpoint.path.name, strength=checkpoint.strength), *own)


def _model_name(entry: GenerationEntry) -> str:
    return Path(entry.model_file).stem if entry.model_file else entry.base_model


def _parameters(
    entry: GenerationEntry, seed: int, loras: tuple[LoraEntry, ...], version: str
) -> str:
    tags = "".join(f" <lora:{Path(lora.path).stem}:{lora.strength}>" for lora in loras)
    lines = [f"{entry.final_prompt()}{tags}"]
    if entry.negative_prompt:
        lines.append(f"Negative prompt: {entry.negative_prompt}")
    lines.append(
        f"Steps: {entry.steps}, Sampler: {entry.sampler}, CFG scale: {entry.cfg_scale}, "
        f"Seed: {seed}, Size: {entry.width}x{entry.height}, Model: {_model_name(entry)}, "
        f"Flow shift: {entry.flow_shift}, Version: {APP_NAME} {version}"
    )
    return "\n".join(lines)


def _settings(entry: GenerationEntry, seed: int, loras: tuple[LoraEntry, ...], version: str) -> str:
    """This very image: its seed, fixed, one image, and only the LoRAs that were used."""
    model_file = Path(entry.model_file).name if entry.model_file else ""
    this_image = entry.with_changes(
        seed=seed, random_seed=False, count=1, loras=loras, model_file=model_file
    )
    return json.dumps(
        {"generator": APP_NAME, "version": version, "entry": this_image.model_dump(mode="json")}
    )
