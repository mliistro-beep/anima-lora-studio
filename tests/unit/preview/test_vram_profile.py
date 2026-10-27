import pytest

from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.vram_profile import (
    IDLE_MARGIN_MIB,
    PROFILES,
    TRAINING_MARGIN_MIB,
    estimate_mib,
    fit_preview_size,
    plan_generation,
    profile_size,
)

SIZE = SampleSize(1024, 1024)


def test_profiles_go_from_fastest_to_lightest():
    estimates = [estimate_mib(profile, SIZE) for profile in PROFILES]

    assert estimates == sorted(estimates, reverse=True)
    assert len(set(estimates)) == len(estimates)
    assert PROFILES[0].name == "full" and not PROFILES[0].fp8
    assert PROFILES[-1].text_encoder_cpu and PROFILES[-1].max_side == 512


def test_plenty_of_free_memory_starts_with_the_fastest_profile_and_keeps_fallbacks():
    plan = plan_generation(11386, SIZE, training=False)

    assert plan.blocked_reason is None
    assert plan.profiles == PROFILES


def test_less_memory_starts_with_the_first_profile_that_fits():
    free = estimate_mib(PROFILES[2], SIZE) + IDLE_MARGIN_MIB

    plan = plan_generation(free, SIZE, training=False)

    assert plan.profiles == PROFILES[2:]


def test_a_running_training_keeps_a_bigger_margin():
    free = estimate_mib(PROFILES[2], SIZE) + IDLE_MARGIN_MIB

    plan = plan_generation(free, SIZE, training=True)

    assert plan.profiles[0] != PROFILES[2]
    assert TRAINING_MARGIN_MIB > IDLE_MARGIN_MIB


def test_too_little_memory_blocks_with_the_needed_amount():
    plan = plan_generation(300, SIZE, training=False)

    assert plan.profiles == ()
    needed = estimate_mib(PROFILES[-1], SIZE) + IDLE_MARGIN_MIB
    assert plan.blocked_reason == (
        f"Not enough free GPU memory: about {needed} MiB needed, 300 MiB free."
    )


def test_unknown_free_memory_starts_from_the_sequential_profile():
    plan = plan_generation(None, SIZE, training=False)

    assert plan.profiles[0].name == "sequential"
    assert plan.profiles[-1] == PROFILES[-1]


@pytest.mark.parametrize(
    "size, cap, expected",
    [(SampleSize(1024, 1536), 768, SampleSize(512, 768)),
     (SampleSize(1024, 1024), 512, SampleSize(512, 512)),
     (SampleSize(1000, 600), 512, SampleSize(512, 304)),
     (SampleSize(512, 512), 768, SampleSize(512, 512))],
)  # fmt: skip
def test_low_resolution_profiles_scale_the_longer_side_keeping_multiples_of_16(size, cap, expected):
    profile = next(p for p in PROFILES if p.max_side == cap)

    assert profile_size(profile, size) == expected


def test_other_profiles_keep_the_size():
    assert profile_size(PROFILES[0], SampleSize(1216, 832)) == SampleSize(1216, 832)


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


def test_unknown_free_memory_during_training_starts_light_to_protect_the_training():
    plan = plan_generation(None, SIZE, training=True)

    assert plan.profiles[0].name == "block_swap"
    assert plan.profiles[-1] == PROFILES[-1]
