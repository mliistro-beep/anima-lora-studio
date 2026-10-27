"""Fit image generation to the GPU memory instead of refusing it.

Generation profiles go from the fastest to the lightest; the generator starts with the first
one that fits the free memory (runtime_scripts/profile_choice.py, with these estimates) and
falls back to the next ones on out-of-memory. Each image is
made by its own process that loads the text encoder, the DiT and the VAE one at a time, so
keeping them all loaded would not be faster. The memory figures are estimates for the Anima
DiT (2B, 28 blocks), Qwen3 0.6B and the Qwen-Image VAE, to be refined with measurements.
"""

from dataclasses import dataclass

from anima_trainer.preview.sample_prompts import SampleSize

PREVIEW_SIDE_STEP = 16  # sd-scripts rounds training previews to a multiple of 16
GENERATOR_SIDE_STEP = 32  # anima_minimal_inference requires multiples of 32
MIB_PER_MEGAPIXEL = 1024 * 1024

CUDA_CONTEXT_MIB = 500
DIT_BF16_MIB = 3990
DIT_FP8_MIB = 2000
DIT_SWAPPED_MIB = 700  # bf16 with 26 of the 28 blocks kept in system RAM
TEXT_ENCODER_MIB = 1140
VAE_MIB = 250
DENOISE_MIB_PER_MP = 1000
DECODE_MIB_PER_MP = 3000
CHUNKED_DECODE_MIB_PER_MP = 800

VAE_CHUNK_SIZE = 64  # same default as the training settings
BLOCKS_TO_SWAP = 26  # sd-scripts allows at most num_blocks - 2

IDLE_MARGIN_MIB = 512
TRAINING_MARGIN_MIB = 1536  # a running training must never be the one to run out of memory


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    label: str
    fp8: bool
    vae_chunk_size: int | None
    blocks_to_swap: int
    text_encoder_cpu: bool
    max_side: int | None  # longer image side is reduced to this


PROFILES = (
    GenerationProfile(
        "standard", "bf16, models loaded one at a time", fp8=False, vae_chunk_size=None,
        blocks_to_swap=0, text_encoder_cpu=False, max_side=None,
    ),
    GenerationProfile(
        "fp8", "DiT in fp8, VAE in chunks", fp8=True, vae_chunk_size=VAE_CHUNK_SIZE,
        blocks_to_swap=0, text_encoder_cpu=False, max_side=None,
    ),
    GenerationProfile(
        "block_swap", "DiT blocks swapped to RAM", fp8=False, vae_chunk_size=VAE_CHUNK_SIZE,
        blocks_to_swap=BLOCKS_TO_SWAP, text_encoder_cpu=False, max_side=None,
    ),
    GenerationProfile(
        "low_res", "Reduced size (768)", fp8=False, vae_chunk_size=VAE_CHUNK_SIZE,
        blocks_to_swap=BLOCKS_TO_SWAP, text_encoder_cpu=True, max_side=768,
    ),
    GenerationProfile(
        "lowest", "Reduced size (512)", fp8=False, vae_chunk_size=VAE_CHUNK_SIZE,
        blocks_to_swap=BLOCKS_TO_SWAP, text_encoder_cpu=True, max_side=512,
    ),
)  # fmt: skip


@dataclass(frozen=True)
class PreviewSizeChoice:
    size: SampleSize | None  # None: previews would not fit during training
    note: str


def profile_size(profile: GenerationProfile, size: SampleSize) -> SampleSize:
    return size if profile.max_side is None else _fit(size, profile.max_side, GENERATOR_SIDE_STEP)


def estimate_mib(profile: GenerationProfile, size: SampleSize) -> int:
    fitted = profile_size(profile, size)
    megapixels = fitted.width * fitted.height / MIB_PER_MEGAPIXEL
    dit = (
        DIT_SWAPPED_MIB if profile.blocks_to_swap else DIT_FP8_MIB if profile.fp8 else DIT_BF16_MIB
    )
    per_mp = CHUNKED_DECODE_MIB_PER_MP if profile.vae_chunk_size else DECODE_MIB_PER_MP
    text_encoder = 0 if profile.text_encoder_cpu else TEXT_ENCODER_MIB
    denoise = dit + DENOISE_MIB_PER_MP * megapixels
    decode = VAE_MIB + per_mp * megapixels
    return round(CUDA_CONTEXT_MIB + max(text_encoder, denoise, decode))


def margin_mib(training: bool) -> int:
    """Free memory left aside: more while a training runs, so it never runs out itself."""
    return TRAINING_MARGIN_MIB if training else IDLE_MARGIN_MIB


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
