import pytest

from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.vram_profile import (
    PROFILES,
    estimate_mib,
    fit_preview_size,
    profile_size,
)

SIZE = SampleSize(1024, 1024)


def test_profiles_go_from_fastest_to_lightest():
    estimates = [estimate_mib(profile, SIZE) for profile in PROFILES]

    assert estimates == sorted(estimates, reverse=True)
    assert len(set(estimates)) == len(estimates)
    assert PROFILES[0].name == "standard" and not PROFILES[0].fp8
    assert PROFILES[-1].text_encoder_cpu and PROFILES[-1].max_side == 512


@pytest.mark.parametrize(
    "size, cap, expected",
    [(SampleSize(1024, 1536), 768, SampleSize(512, 768)),
     (SampleSize(1024, 1024), 512, SampleSize(512, 512)),
     (SampleSize(1000, 600), 512, SampleSize(512, 288)),
     (SampleSize(512, 512), 768, SampleSize(512, 512))],
)  # fmt: skip
def test_low_resolution_profiles_scale_the_longer_side_keeping_multiples_of_32(size, cap, expected):
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
