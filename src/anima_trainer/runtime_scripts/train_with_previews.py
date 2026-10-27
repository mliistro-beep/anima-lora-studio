"""Run sd-scripts' Anima LoRA training with ER-SDE previews and out-of-memory-safe sampling.

accelerate starts this file instead of anima_train_network.py, with the same arguments, and
only when previews are on. It replaces two functions of library.anima_train_utils, then runs
the unchanged training script:

- do_sample: ER-SDE when the preview prompt asks for it (sample_sampler = "er_sde"),
  sd-scripts' own Euler otherwise, so Euler previews are exactly sd-scripts' images;
- _sample_image_inference: the preview is decoded in VAE tiles (sd-scripts decodes the whole
  image at once, about 5 GB at 1024x1024 on top of the training; tiles keep it under 1 GB),
  and a preview that runs out of GPU memory is skipped with a WARN line instead of stopping
  the training; afterwards the VAE goes back where it was, with tiling as it was.
"""

import gc
import inspect
import os
import runpy
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from samplers import denoiser_from_velocity, er_sde, flow_sigmas

SD_SCRIPTS_ENV = "ANIMA_TRAINER_SD_SCRIPTS"
TRAIN_SCRIPT = "anima_train_network.py"
ER_SDE = "er_sde"
LATENT_CHANNELS = 16
LATENT_SCALE = 8
OOM_WARNING = "WARN: Preview skipped, the GPU ran out of memory (prompt {index}). Training goes on."


class _PromptState:
    """Sampler of the preview being generated, set around each prompt."""

    def __init__(self) -> None:
        self.sampler = ""


def er_sde_do_sample(
    height: int,
    width: int,
    seed: int | None,
    dit: Any,
    crossattn_emb: torch.Tensor,
    steps: int,
    dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 1.0,
    flow_shift: float = 3.0,
    neg_crossattn_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """sd-scripts' do_sample (same arguments, same starting noise for a seed) with ER-SDE."""
    latent_h, latent_w = height // LATENT_SCALE, width // LATENT_SCALE
    generator = torch.manual_seed(seed) if seed is not None else None
    shape = (1, LATENT_CHANNELS, 1, latent_h, latent_w)
    noise = torch.randn(shape, dtype=torch.float32, generator=generator, device="cpu")
    x = noise.to(dtype).to(device)
    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)
    use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None

    def velocity(latent: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        model_input = latent.to(dtype)
        t = sigma.to(dtype).unsqueeze(0)
        positive = dit(model_input, t, crossattn_emb, padding_mask=padding_mask).float()
        if not use_cfg:
            return positive
        negative = dit(model_input, t, neg_crossattn_emb, padding_mask=padding_mask).float()
        return negative + guidance_scale * (positive - negative)

    noise_generator = torch.Generator()
    if seed is None:
        noise_generator.seed()
    else:
        noise_generator.manual_seed(seed + 1)
    sigmas = flow_sigmas(steps, float(flow_shift), device)
    return er_sde(denoiser_from_velocity(velocity), x, sigmas, noise_generator).to(dtype)


@contextmanager
def _tiled_decode(vae: Any) -> Iterator[None]:
    """VAE tiles for the preview only: use_tiling also applies to the training's own encode
    (latents not cached). A VAE without tiling (sd-scripts' 2D one) decodes as before."""
    if not hasattr(vae, "use_tiling"):
        yield
        return
    was_tiling = vae.use_tiling
    vae.enable_tiling()
    try:
        yield
    finally:
        if not was_tiling:
            vae.disable_tiling()


def install_hooks(utils: ModuleType, report: Callable[[str], None] = print) -> None:
    """Replace do_sample and _sample_image_inference in sd-scripts' anima_train_utils."""
    state = _PromptState()
    euler_do_sample = utils.do_sample
    inference = utils._sample_image_inference
    signature = inspect.signature(inference)

    def do_sample(*args: Any, **kwargs: Any) -> torch.Tensor:
        sampler = er_sde_do_sample if state.sampler == ER_SDE else euler_do_sample
        return sampler(*args, **kwargs)

    def safe_inference(*args: Any, **kwargs: Any) -> Any:
        arguments = signature.bind(*args, **kwargs).arguments
        prompt_dict, vae = arguments["prompt_dict"], arguments["vae"]
        vae_device = vae.device
        state.sampler = str(prompt_dict.get("sample_sampler", ""))
        try:
            with _tiled_decode(vae):
                return inference(*args, **kwargs)
        except torch.OutOfMemoryError:
            report(OOM_WARNING.format(index=prompt_dict.get("enum", 0)))
        finally:
            state.sampler = ""
        # Outside the except block, so the error's frames (and their tensors) are released.
        vae.to(vae_device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    utils.do_sample = do_sample
    utils._sample_image_inference = safe_inference


def sd_scripts_dir() -> Path:
    configured = os.environ.get(SD_SCRIPTS_ENV)
    return Path(configured) if configured else Path.cwd() / "sd-scripts"


def main() -> None:
    folder = sd_scripts_dir()
    train_script = folder / TRAIN_SCRIPT
    sys.path.insert(0, str(folder))
    from library import anima_train_utils

    install_hooks(anima_train_utils, lambda line: print(line, flush=True))
    sys.argv = [str(train_script), *sys.argv[1:]]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
