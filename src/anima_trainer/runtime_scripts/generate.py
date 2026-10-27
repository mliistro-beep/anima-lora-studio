"""Generate one image with Anima (and optionally a LoRA), fitted to the free GPU memory.

Run by the training venv's Python. Models are loaded one at a time (text encoder, DiT, VAE)
with sd-scripts' anima_minimal_inference helpers; the sampler is ER-SDE or Euler. The memory
profiles are tried in the given order, from the first whose estimate plus the margin fits the
free GPU memory read here, skipping those whose RAM figure plus its margin does not fit the
available system RAM; on out-of-memory the next, lighter one is used.

Reports to the app through stdout, one line each:
  INFO: <text>            free GPU memory and available RAM
  PROFILE: <name>         a profile is being tried
  STEP: <done>/<total>    a sampling step finished
  WARN: <text>            a profile ran out of memory, the next one follows
  IMAGE: <path>           the image was saved
  ERROR: <text>           nothing was generated (exit code 1, or 3 when out of memory)
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from profile_choice import (
    MIB,
    MemoryBudget,
    ProfileNeed,
    all_out_of_gpu_memory,
    available_ram_mib,
    lighter_skipped_for_ram,
    next_profile_warning,
    not_enough_memory,
    ram_unknown_warning,
    usable_profiles,
)
from samplers import denoiser_from_velocity, er_sde, euler, flow_sigmas

SD_SCRIPTS_ENV = "ANIMA_TRAINER_SD_SCRIPTS"
EXIT_FAILED = 1
EXIT_OUT_OF_MEMORY = 3
LATENT_SCALE = 8
MAX_TOKENS = 512
MEMINFO = "/proc/meminfo"


@dataclass(frozen=True)
class Profile:
    name: str
    vae_tiling: bool
    blocks_to_swap: int
    text_encoder_cpu: bool
    width: int
    height: int
    estimate_mib: int  # GPU memory
    ram_mib: int  # system RAM


def say(kind: str, text: str) -> None:
    print(f"{kind}: {text}", flush=True)


def parse_options(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("--dit", "--qwen3", "--vae", "--output", "--prompt"):
        parser.add_argument(name, required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--lora", default="")
    parser.add_argument("--lora_multiplier", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--flow_shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", choices=("er_sde", "euler"), default="er_sde")
    parser.add_argument("--profiles", required=True, help="JSON list of memory profiles")
    parser.add_argument("--margin_mib", type=int, default=512, help="free GPU memory left aside")
    parser.add_argument("--ram_margin_mib", type=int, default=1024, help="free RAM left aside")
    return parser.parse_args(argv)


def parse_profiles(text: str) -> tuple[Profile, ...]:
    return tuple(Profile(**entry) for entry in json.loads(text))


def _inference_args(ami: Any, options: argparse.Namespace, profile: Profile) -> Any:
    """anima_minimal_inference's own argument namespace for this profile."""
    argv = [
        "--dit", options.dit, "--vae", options.vae, "--text_encoder", options.qwen3,
        "--prompt", options.prompt, "--negative_prompt", options.negative_prompt,
        "--image_size", str(profile.height), str(profile.width),
        "--infer_steps", str(options.steps), "--guidance_scale", str(options.cfg),
        "--flow_shift", str(options.flow_shift), "--seed", str(options.seed),
        "--save_path", str(Path(options.output).parent),
    ]  # fmt: skip
    if options.lora:
        argv += ["--lora_weight", options.lora, "--lora_multiplier", str(options.lora_multiplier)]
    if profile.text_encoder_cpu:
        argv.append("--text_encoder_cpu")
    saved, sys.argv = sys.argv, ["anima_minimal_inference.py", *argv]
    try:
        return ami.parse_args()
    finally:
        sys.argv = saved


def _set_strategies(options: argparse.Namespace) -> None:
    from library import strategy_anima, strategy_base

    tokenize = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=options.qwen3, t5_tokenizer_path=None, qwen3_max_length=MAX_TOKENS,
        t5_max_length=MAX_TOKENS,
    )  # fmt: skip
    strategy_base.TokenizeStrategy.set_strategy(tokenize)
    strategy_base.TextEncodingStrategy.set_strategy(strategy_anima.AnimaTextEncodingStrategy())


def _encode_prompts(ami: Any, args: Any, device: torch.device) -> list[list[torch.Tensor]]:
    """Raw text encoder outputs for the prompt and the negative prompt, kept on the CPU."""
    from library import strategy_base

    te_device = torch.device("cpu") if args.text_encoder_cpu else device
    no_lora = argparse.Namespace(**{**vars(args), "lora_weight": None})
    text_encoder = ami.load_text_encoder(no_lora, dtype=torch.bfloat16, device=te_device)
    tokenize = strategy_base.TokenizeStrategy.get_strategy()
    encode = strategy_base.TextEncodingStrategy.get_strategy()
    embeds = []
    with torch.no_grad():
        for text in (args.prompt, args.negative_prompt):
            tokens = tokenize.tokenize(ami.process_escape(text))
            embed = encode.encode_tokens(tokenize, [text_encoder], tokens)
            embeds.append([part.cpu() for part in embed])
    del text_encoder
    _free(device)
    return embeds


def _load_dit(ami: Any, args: Any, profile: Profile, device: torch.device) -> Any:
    if not profile.blocks_to_swap:
        return ami.load_dit_model(args, device, torch.bfloat16)
    dit = ami.load_dit_model(args, torch.device("cpu"), torch.bfloat16)
    dit.enable_block_swap(profile.blocks_to_swap, device)
    dit.move_to_device_except_swap_blocks(device)
    dit.switch_block_swap_for_inference()
    return dit


def _cross_attention(dit: Any, embed: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        crossattn = dit._preprocess_text_embeds(
            source_hidden_states=embed[0].to(device),
            target_input_ids=embed[2].to(device),
            target_attention_mask=embed[3].to(device),
            source_attention_mask=embed[1].to(device),
        )
    crossattn[~embed[3].to(device).bool()] = 0
    return crossattn.to(torch.bfloat16)


def _sample(
    ami: Any,
    dit: Any,
    options: argparse.Namespace,
    profile: Profile,
    conds: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    positive, negative = conds
    shape = (
        1,
        dit.LATENT_CHANNELS,
        1,
        profile.height // LATENT_SCALE,
        profile.width // LATENT_SCALE,
    )
    seed_generator = torch.Generator(device="cpu").manual_seed(options.seed)
    latents = ami.randn_tensor(shape, generator=seed_generator, device=device, dtype=torch.bfloat16)
    padding_mask = torch.zeros(1, 1, shape[-2], shape[-1], dtype=torch.bfloat16, device=device)
    use_cfg = options.cfg != 1.0
    done = [0]

    def velocity(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        model_input = x.to(torch.bfloat16)
        t = sigma.to(device, torch.bfloat16).reshape(1)
        with torch.no_grad():
            prediction = dit(model_input, t, positive, padding_mask=padding_mask).float()
            if use_cfg:
                unconditional = dit(model_input, t, negative, padding_mask=padding_mask).float()
                prediction = unconditional + options.cfg * (prediction - unconditional)
        done[0] += 1
        say("STEP", f"{done[0]}/{options.steps}")
        return prediction

    sigmas = flow_sigmas(options.steps, options.flow_shift, device)
    if options.sampler == "euler":
        return euler(velocity, latents, sigmas).to(torch.bfloat16)
    noise_generator = torch.Generator().manual_seed(options.seed + 1)
    denoise = denoiser_from_velocity(velocity)
    return er_sde(denoise, latents, sigmas, noise_generator).to(torch.bfloat16)


def _save(pixels: torch.Tensor, output: Path) -> None:
    from PIL import Image

    image = ((torch.clamp(pixels, -1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.transpose(1, 2, 0)).save(output)


def _free(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _load_vae(args: Any, profile: Profile) -> Any:
    """The VAE on the CPU; decode_latent moves it to the GPU. Tiles (256 px, sd-scripts'
    default) keep the decode memory small whatever the image size."""
    from library import anima_train_utils

    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
    vae.to(torch.bfloat16).eval()
    if profile.vae_tiling:
        vae.enable_tiling()
    return vae


def generate(ami: Any, options: argparse.Namespace, profile: Profile, device: torch.device) -> None:
    args = _inference_args(ami, options, profile)
    embeds = _encode_prompts(ami, args, device)
    dit = _load_dit(ami, args, profile, device)
    conds = (_cross_attention(dit, embeds[0], device), _cross_attention(dit, embeds[1], device))
    latents = _sample(ami, dit, options, profile, conds, device)
    del dit, conds
    _free(device)
    vae = _load_vae(args, profile)
    _save(ami.decode_latent(vae, latents, device), Path(options.output))
    del vae
    _free(device)


def _free_mib(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    free, _total = torch.cuda.mem_get_info(device)
    return free // MIB


def _available_ram_mib() -> int | None:
    try:
        return available_ram_mib(Path(MEMINFO).read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError):  # not Linux, or no /proc
        return None


def _memory_note(budget: MemoryBudget) -> str:
    gpu = (
        f"{budget.free_vram_mib} MiB of GPU memory free"
        if budget.free_vram_mib is not None
        else "No CUDA GPU, using the CPU"
    )
    if budget.available_ram_mib is None:
        return f"{gpu}, system RAM unknown"
    return f"{gpu}, {budget.available_ram_mib} MiB of RAM available"


def run(ami: Any, options: argparse.Namespace, profiles: tuple[Profile, ...]) -> int:
    """RAM is read once, before loading anything: each profile's figure already counts what
    this process takes, so after an out-of-memory error the starting value still applies."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    budget = MemoryBudget(
        free_vram_mib=_free_mib(device),
        vram_margin_mib=options.margin_mib,
        available_ram_mib=_available_ram_mib(),
        ram_margin_mib=options.ram_margin_mib,
    )
    needs = [ProfileNeed(profile.estimate_mib, profile.ram_mib) for profile in profiles]
    usable = [profiles[index] for index in usable_profiles(needs, budget)]
    if not usable:
        say("ERROR", not_enough_memory(needs, budget))
        return EXIT_OUT_OF_MEMORY
    say("INFO", _memory_note(budget))
    ram_warning = ram_unknown_warning(sys.platform) if budget.available_ram_mib is None else None
    if ram_warning is not None:
        say("WARN", ram_warning)
    _set_strategies(options)
    for position, profile in enumerate(usable):
        say("PROFILE", profile.name)
        failed = False
        try:
            generate(ami, options, profile, device)
        except torch.OutOfMemoryError:
            failed = True
        if not failed:
            say("IMAGE", options.output)
            return 0
        _free(device)  # outside the except block, so the error's tensors can be released
        if position + 1 < len(usable):
            say("WARN", next_profile_warning(profile.name, usable[position + 1].name))
    skipped = [profiles[index].name for index in lighter_skipped_for_ram(needs, budget)]
    say("ERROR", all_out_of_gpu_memory(usable[-1].name, skipped, budget.available_ram_mib))
    return EXIT_OUT_OF_MEMORY


def main() -> int:
    options = parse_options(sys.argv[1:])
    profiles = parse_profiles(options.profiles)
    if not profiles:
        say("ERROR", "No memory profile given.")
        return EXIT_FAILED
    folder = Path(os.environ.get(SD_SCRIPTS_ENV) or Path.cwd() / "sd-scripts")
    sys.path.insert(0, str(folder))
    import anima_minimal_inference as ami

    return run(ami, options, profiles)


if __name__ == "__main__":
    sys.exit(main())
