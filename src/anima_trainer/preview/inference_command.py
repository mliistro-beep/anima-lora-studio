"""The command that runs runtime_scripts/generate.py in the training venv for one image."""

import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from types import MappingProxyType

from anima_trainer.preview.image_metadata import ImageMetadata
from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.vram_profile import (
    GenerationProfile,
    estimate_mib,
    fallback_chain,
    margin_mib,
    ram_margin_mib,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime_scripts import GENERATE
from anima_trainer.training.command_builder import SD_SCRIPTS_ENV, LaunchSpec

GENERATED_DIR = "generated"
TIME_FORMAT = "%Y%m%d-%H%M%S"


@dataclass(frozen=True)
class LoraUse:
    path: Path
    strength: float


@dataclass(frozen=True)
class GenerationRequest:
    """One image, with no tie to the training: the model file and the LoRAs to apply."""

    dit: Path
    loras: tuple[LoraUse, ...]  # empty: the base model alone
    prompt: str
    negative_prompt: str
    size: SampleSize
    steps: int
    cfg_scale: float
    flow_shift: float
    seed: int
    sampler: str
    output: Path
    metadata: ImageMetadata | None = None  # the PNG's texts; None: a plain PNG
    batch: int = 1  # the queue entry it comes from (the form is a batch of its own)
    batches: int = 1


def default_generated_folder(paths: RuntimePaths) -> Path:
    """Where the generator saves when no folder was chosen: <data>/generated."""
    return paths.root / GENERATED_DIR


def project_generated_folder(output_dir: Path) -> Path:
    """The generated images of a training project: <output>/generated."""
    return output_dir / GENERATED_DIR


def output_image_path(
    folder: Path,
    prefix: str,
    label: str,
    seed: int,
    now: datetime,
    taken: Collection[Path] = frozenset(),
    number: int | None = None,
) -> Path:
    """<folder>/{prefix}_{label}_{time}_{seed}.png, or ..._{time}_{number}_{seed}.png for the
    images of a group (03 for the third), so they sort in the order they were made; _2, _3, ...
    is added when that file exists (two images in the same second) or is `taken` by an image
    planned before it."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    place = "" if number is None else f"_{number:02d}"
    stem = f"{prefix}_{slug}_{now.strftime(TIME_FORMAT)}{place}_{seed}"
    candidates = (folder / f"{stem}{'' if n == 1 else f'_{n}'}.png" for n in count(1))
    return next(path for path in candidates if path not in taken and not path.exists())


def _png_text_arguments(metadata: ImageMetadata | None) -> tuple[str, ...]:
    """With `=`: a text starting with a dash must not be read as an option."""
    if metadata is None:
        return ()
    return (f"--png_parameters={metadata.parameters}", f"--png_settings={metadata.settings}")


def _profiles_json(profiles: tuple[GenerationProfile, ...], size: SampleSize) -> str:
    entries = [
        {
            "name": profile.name,
            "vae_tiling": profile.vae_tiling,
            "blocks_to_swap": profile.blocks_to_swap,
            "text_encoder_cpu": profile.text_encoder_cpu,
            "width": size.width,
            "height": size.height,
            "estimate_mib": estimate_mib(profile, size),
            "ram_mib": profile.ram_mib,
        }
        for profile in profiles
    ]
    return json.dumps(entries)


def build_generation_launch(
    paths: RuntimePaths,
    request: GenerationRequest,
    profiles: tuple[GenerationProfile, ...],
    gpu_index: str,
    training: bool,
) -> LaunchSpec:
    """generate.py tries `profiles` in order, from the first that fits the free GPU memory
    with the margin (larger while a training runs), skipping those that do not fit the
    available system RAM with its margin; fallbacks that would not save GPU memory at this
    size are left out."""
    loras = tuple(
        part
        for lora in request.loras
        for part in ("--lora", str(lora.path), "--lora_multiplier", str(lora.strength))
    )
    arguments = (
        "-B",  # no __pycache__ next to the scripts or in sd-scripts
        str(GENERATE),
        "--dit", str(request.dit),
        "--qwen3", str(paths.qwen3_model),
        "--vae", str(paths.vae_model),
        *loras,
        # One item each: as two, argparse takes a prompt starting with "-" for an option.
        f"--prompt={request.prompt}",
        f"--negative_prompt={request.negative_prompt}",
        "--steps", str(request.steps),
        "--cfg", str(request.cfg_scale),
        "--flow_shift", str(request.flow_shift),
        "--seed", str(request.seed),
        "--sampler", request.sampler,
        "--profiles", _profiles_json(fallback_chain(profiles, request.size), request.size),
        "--margin_mib", str(margin_mib(training)),
        "--ram_margin_mib", str(ram_margin_mib(training)),
        "--output", str(request.output),
        *_png_text_arguments(request.metadata),
    )  # fmt: skip
    env = MappingProxyType(
        {
            "CUDA_VISIBLE_DEVICES": gpu_index,
            "PYTHONUNBUFFERED": "1",
            # less memory held between the text encoder, DiT and VAE phases
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            SD_SCRIPTS_ENV: str(paths.sd_scripts_dir),
        }
    )
    return LaunchSpec(paths.venv_python, arguments, env, paths.root)
