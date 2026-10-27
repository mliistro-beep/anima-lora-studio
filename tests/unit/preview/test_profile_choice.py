from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.vram_profile import (
    IDLE_MARGIN_MIB,
    PROFILES,
    TRAINING_MARGIN_MIB,
    estimate_mib,
    margin_mib,
)
from anima_trainer.runtime_scripts.profile_choice import first_fitting, not_enough_memory

SIZE = SampleSize(1024, 1024)
ESTIMATES = [estimate_mib(profile, SIZE) for profile in PROFILES]


def test_plenty_of_free_memory_starts_with_the_fastest_profile():
    assert first_fitting(ESTIMATES, 11386, IDLE_MARGIN_MIB) == 0


def test_less_memory_starts_with_the_first_profile_that_fits():
    assert first_fitting(ESTIMATES, ESTIMATES[1] + IDLE_MARGIN_MIB, IDLE_MARGIN_MIB) == 1


def test_a_running_training_keeps_a_bigger_margin():
    free = ESTIMATES[1] + IDLE_MARGIN_MIB

    assert first_fitting(ESTIMATES, free, TRAINING_MARGIN_MIB) == 2
    assert (margin_mib(False), margin_mib(True)) == (IDLE_MARGIN_MIB, TRAINING_MARGIN_MIB)


def test_too_little_memory_fits_nothing_and_says_how_much_is_needed():
    assert first_fitting(ESTIMATES, 300, IDLE_MARGIN_MIB) is None
    needed = ESTIMATES[-1] + IDLE_MARGIN_MIB
    assert not_enough_memory(ESTIMATES, 300, IDLE_MARGIN_MIB) == (
        f"Not enough free GPU memory: about {needed} MiB needed, 300 MiB free."
    )


def test_unknown_free_memory_starts_from_the_first_profile():
    assert first_fitting(ESTIMATES, None, TRAINING_MARGIN_MIB) == 0
