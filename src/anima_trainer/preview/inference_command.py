"""The command that runs runtime_scripts/generate.py in the training venv for one image."""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from types import MappingProxyType

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
class GenerationRequest:
    base_model: str
    lora: Path | None  # None: the base model alone
    lora_multiplier: float
    prompt: str
    negative_prompt: str
    size: SampleSize
    steps: int
    cfg_scale: float
    flow_shift: float
    seed: int
    sampler: str
    output: Path


def output_image_path(output_dir: Path, project: str, label: str, seed: int, now: datetime) -> Path:
    """<output_dir>/generated/{project}_{checkpoint label}_{time}_{seed}.png, with _2, _3, ...
    added when that file exists (two images in the same second)."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    stem = f"{project}_{slug}_{now.strftime(TIME_FORMAT)}_{seed}"
    folder = output_dir / GENERATED_DIR
    candidates = (folder / f"{stem}{'' if n == 1 else f'_{n}'}.png" for n in count(1))
    return next(path for path in candidates if not path.exists())


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
    lora = ("--lora", str(request.lora)) if request.lora is not None else ()
    arguments = (
        "-B",  # no __pycache__ next to the scripts or in sd-scripts
        str(GENERATE),
        "--dit", str(paths.dit_model(request.base_model)),
        "--qwen3", str(paths.qwen3_model),
        "--vae", str(paths.vae_model),
        *lora,
        "--lora_multiplier", str(request.lora_multiplier),
        "--prompt", request.prompt,
        "--negative_prompt", request.negative_prompt,
        "--steps", str(request.steps),
        "--cfg", str(request.cfg_scale),
        "--flow_shift", str(request.flow_shift),
        "--seed", str(request.seed),
        "--sampler", request.sampler,
        "--profiles", _profiles_json(fallback_chain(profiles, request.size), request.size),
        "--margin_mib", str(margin_mib(training)),
        "--ram_margin_mib", str(ram_margin_mib(training)),
        "--output", str(request.output),
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
