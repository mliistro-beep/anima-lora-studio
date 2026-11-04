"""Fit image generation to the GPU memory instead of refusing it.

Generation profiles go from the fastest to the lightest; the generator starts with the first
one that fits the free memory (runtime_scripts/profile_choice.py, with these estimates) and
falls back to the next ones on out-of-memory. Each image is made by its own process that loads
the text encoder, the DiT and the VAE one at a time, so the peak is the largest phase.

The figures are PyTorch's reserved memory per phase, measured with generate.py on an RTX 3060
(Anima DiT 2B with 28 blocks, Qwen3 0.6B, Qwen-Image VAE; 2026-10-27, expandable_segments on),
rounded up. The VAE decode is the largest phase unless it is tiled: its activations grow with
the whole image, while tiles keep them to 256 px squares at little cost in time. The image is
never made smaller to fit: with the DiT blocks swapped and the VAE in tiles, the floor is the
weights left on the GPU, not the image size.
"""

from dataclasses import dataclass

from anima_trainer.preview.sample_prompts import SampleSize

PREVIEW_SIDE_STEP = 16  # sd-scripts rounds training previews to a multiple of 16
GENERATOR_SIDE_STEP = 32  # anima_minimal_inference requires multiples of 32
MIB_PER_MEGAPIXEL = 1024 * 1024

CUDA_CONTEXT_MIB = 250
TEXT_ENCODER_MIB = 1190
DIT_BF16_MIB = 4050
DIT_SWAPPED_MIB = 720  # bf16 with 26 of the 28 blocks kept in system RAM
DENOISE_MIB_PER_MP = 330
VAE_DECODE_MIB = 300
VAE_DECODE_MIB_PER_MP = 4750
VAE_TILED_DECODE_MIB = 750
VAE_TILED_DECODE_MIB_PER_MP = 150

BLOCKS_TO_SWAP = 26  # sd-scripts allows at most num_blocks - 2

IDLE_MARGIN_MIB = 512
TRAINING_MARGIN_MIB = 1536  # a running training must never be the one to run out of memory

# System RAM left aside: above earlyoom's threshold (6% of 16 GB, about 950 MiB, see
# DOCS/SYSTEM_SETUP.md), and more while a training runs, whose RAM grows at previews and saves.
IDLE_RAM_MARGIN_MIB = 1024
TRAINING_RAM_MARGIN_MIB = 2048


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    label: str
    blocks_to_swap: int
    vae_tiling: bool
    text_encoder_cpu: bool
    # System RAM taken by generate.py (drop of MemAvailable), measured at 1024x1024 on
    # 2026-10-27 and rounded up: the models are loaded through the CPU, and swapped DiT
    # blocks stay there. The latents stay on the GPU, so the image size barely changes it.
    ram_mib: int


PROFILES = (
    GenerationProfile(
        "standard", "bf16, models loaded one at a time", blocks_to_swap=0, vae_tiling=False,
        text_encoder_cpu=False, ram_mib=4150,
    ),
    GenerationProfile(
        "block_swap", "DiT blocks swapped to RAM, VAE in tiles", blocks_to_swap=BLOCKS_TO_SWAP,
        vae_tiling=True, text_encoder_cpu=False, ram_mib=7250,
    ),
    GenerationProfile(
        "low_vram", "Text encoder on the CPU", blocks_to_swap=BLOCKS_TO_SWAP, vae_tiling=True,
        text_encoder_cpu=True, ram_mib=6400,
    ),
)  # fmt: skip


@dataclass(frozen=True)
class PreviewSizeChoice:
    size: SampleSize | None  # None: previews would not fit during training
    note: str


def estimate_mib(profile: GenerationProfile, size: SampleSize) -> int:
    megapixels = size.width * size.height / MIB_PER_MEGAPIXEL
    text_encoder = 0 if profile.text_encoder_cpu else TEXT_ENCODER_MIB
    dit = DIT_SWAPPED_MIB if profile.blocks_to_swap else DIT_BF16_MIB
    denoise = dit + DENOISE_MIB_PER_MP * megapixels
    if profile.vae_tiling:
        decode = VAE_TILED_DECODE_MIB + VAE_TILED_DECODE_MIB_PER_MP * megapixels
    else:
        decode = VAE_DECODE_MIB + VAE_DECODE_MIB_PER_MP * megapixels
    return round(CUDA_CONTEXT_MIB + max(text_encoder, denoise, decode))


def fallback_chain(
    profiles: tuple[GenerationProfile, ...], size: SampleSize
) -> tuple[GenerationProfile, ...]:
    """`profiles` without the ones that would not use less memory than the profile before them:
    retrying with those after an out-of-memory error would reload the models and fail again."""
    chain: list[GenerationProfile] = []
    for profile in profiles:
        if not chain or estimate_mib(profile, size) < estimate_mib(chain[-1], size):
            chain.append(profile)
    return tuple(chain)


def margin_mib(training: bool) -> int:
    """Free memory left aside: more while a training runs, so it never runs out itself."""
    return TRAINING_MARGIN_MIB if training else IDLE_MARGIN_MIB


def ram_margin_mib(training: bool) -> int:
    """System RAM left aside, more while a training runs (see TRAINING_RAM_MARGIN_MIB)."""
    return TRAINING_RAM_MARGIN_MIB if training else IDLE_RAM_MARGIN_MIB


# Longest preview side that leaves room for sampling inside a training, by total GPU memory.
_PREVIEW_SIDE_BY_TOTAL_MIB = ((11264, 1024), (7680, 768), (5632, 512))


def fit_preview_size(total_mib: int | None, requested: SampleSize, auto: bool) -> PreviewSizeChoice:
    text = f"{requested.width}x{requested.height}"
    if total_mib is None:
        return PreviewSizeChoice(requested, f"INFO: GPU memory unknown, preview size {text} kept.")
    advised = next((side for limit, side in _PREVIEW_SIDE_BY_TOTAL_MIB if total_mib >= limit), None)
    if not auto:
        if advised is None or max(requested.width, requested.height) > advised:
            note = (
                f"WARN: Preview size {text} may not fit in {total_mib} MiB of GPU memory during "
                "training; a preview that runs out of memory is skipped."
            )
            return PreviewSizeChoice(requested, note)
        return PreviewSizeChoice(requested, f"INFO: Preview size {text}.")
    if advised is None:
        note = (
            f"WARN: Previews turned off: {total_mib} MiB of GPU memory is too little to "
            "sample during training."
        )
        return PreviewSizeChoice(None, note)
    size = _fit(requested, advised, PREVIEW_SIDE_STEP)
    note = (
        f"INFO: Preview size {size.width}x{size.height} chosen for {total_mib} MiB of GPU memory."
    )
    return PreviewSizeChoice(size, note)


def _fit(size: SampleSize, max_side: int, step: int) -> SampleSize:
    longer = max(size.width, size.height)
    if longer <= max_side:
        return size
    ratio = max_side / longer

    def scaled(side: int) -> int:
        return max(step, int(side * ratio) // step * step)

    return SampleSize(scaled(size.width), scaled(size.height))
