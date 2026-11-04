import pytest

from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.vram_profile import (
    PROFILES,
    estimate_mib,
    fallback_chain,
    fit_preview_size,
)

SIZE = SampleSize(1024, 1024)


def _profile(name):
    return next(profile for profile in PROFILES if profile.name == name)


def test_profiles_go_from_fastest_to_lightest():
    estimates = [estimate_mib(profile, SIZE) for profile in PROFILES]

    assert [profile.name for profile in PROFILES] == ["standard", "block_swap", "low_vram"]
    assert estimates == sorted(estimates, reverse=True)
    assert len(set(estimates)) == len(estimates)


@pytest.mark.parametrize(
    "size, expected",
    [(SampleSize(512, 512), ["standard", "block_swap", "low_vram"]),
     (SampleSize(1024, 1024), ["standard", "block_swap", "low_vram"]),
     (SampleSize(1536, 1024), ["standard", "block_swap"]),
     (SampleSize(1536, 1536), ["standard", "block_swap"])],
)  # fmt: skip
def test_the_fallback_chain_keeps_only_profiles_that_save_memory(size, expected):
    # Above ~1.4 megapixels the DiT sampling outgrows the text encoder, so moving the text
    # encoder to the CPU saves nothing and retrying with it would fail the same way.
    chain = fallback_chain(PROFILES, size)

    assert [profile.name for profile in chain] == expected
    estimates = [estimate_mib(profile, size) for profile in chain]
    assert estimates == sorted(set(estimates), reverse=True)  # strictly decreasing


def test_only_the_standard_profile_keeps_whole_models_on_the_gpu():
    standard, *lighter = PROFILES

    assert (standard.blocks_to_swap, standard.vae_tiling, standard.text_encoder_cpu) == (
        0, False, False,
    )  # fmt: skip
    assert all(profile.blocks_to_swap and profile.vae_tiling for profile in lighter)
    assert [profile.text_encoder_cpu for profile in PROFILES] == [False, False, True]


# GPU memory used by generate.py at 1024x1024 on an RTX 3060 (nvidia-smi peak minus idle,
# CUDA context included, expandable_segments on), measured on 2026-10-27; estimates must
# cover it, not by much. low_vram measured 1016 MiB: its estimate (1300) stays cautious
# because its DiT sampling term is the one measured for the whole bf16 DiT.
@pytest.mark.parametrize(
    "name, size, measured_mib",
    [("standard", SampleSize(1024, 1024), 5084),
     ("block_swap", SampleSize(1024, 1024), 1316)],
)  # fmt: skip
def test_estimates_cover_the_measured_peak(name, size, measured_mib):
    estimate = estimate_mib(_profile(name), size)

    assert measured_mib <= estimate <= measured_mib * 1.15


# System RAM taken by generate.py at 1024x1024 (drop of MemAvailable during the run, 4 steps,
# no LoRA), measured on 2026-10-27 on the same machine; the profile figures must cover it.
@pytest.mark.parametrize(
    "name, measured_mib",
    [("standard", 3948), ("block_swap", 6911), ("low_vram", 6107)],
)  # fmt: skip
def test_ram_figures_cover_the_measured_use(name, measured_mib):
    assert measured_mib <= _profile(name).ram_mib <= measured_mib * 1.15


def test_the_fastest_profile_needs_the_least_ram():
    standard, *lighter = PROFILES

    assert all(standard.ram_mib < profile.ram_mib for profile in lighter)


def test_estimates_grow_with_the_image_size():
    for profile in PROFILES:
        assert estimate_mib(profile, SampleSize(1536, 1024)) > estimate_mib(profile, SIZE)


@pytest.mark.parametrize(
    "total, expected",
    [(24576, SampleSize(1024, 1024)), (12288, SampleSize(1024, 1024)),
     (8192, SampleSize(768, 768)), (6144, SampleSize(512, 512)), (4096, None)],
)  # fmt: skip
def test_automatic_preview_size_follows_the_total_gpu_memory(total, expected):
    choice = fit_preview_size(total, SIZE, auto=True)

    assert choice.size == expected
    assert choice.note


def test_fixed_preview_size_is_kept_but_warns_when_above_the_advice():
    choice = fit_preview_size(6144, SIZE, auto=True)
    fixed = fit_preview_size(6144, SIZE, auto=False)

    assert fixed.size == SIZE
    assert fixed.note.startswith("WARN:")
    assert choice.note.startswith("INFO:")


def test_unknown_total_memory_keeps_the_requested_size():
    choice = fit_preview_size(None, SampleSize(768, 1024), auto=True)

    assert choice.size == SampleSize(768, 1024)
