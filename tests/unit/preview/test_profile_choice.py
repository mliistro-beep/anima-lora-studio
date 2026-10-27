import pytest

from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.vram_profile import (
    IDLE_MARGIN_MIB,
    IDLE_RAM_MARGIN_MIB,
    PROFILES,
    TRAINING_MARGIN_MIB,
    TRAINING_RAM_MARGIN_MIB,
    estimate_mib,
    margin_mib,
    ram_margin_mib,
)
from anima_trainer.runtime_scripts.profile_choice import (
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

SIZE = SampleSize(1024, 1024)
NEEDS = [ProfileNeed(estimate_mib(profile, SIZE), profile.ram_mib) for profile in PROFILES]
PLENTY_OF_RAM = 12000


def _budget(free_vram, available_ram=PLENTY_OF_RAM, training=False):
    return MemoryBudget(
        free_vram_mib=free_vram,
        vram_margin_mib=margin_mib(training),
        available_ram_mib=available_ram,
        ram_margin_mib=ram_margin_mib(training),
    )


def test_plenty_of_free_memory_starts_with_the_fastest_profile():
    assert usable_profiles(NEEDS, _budget(11386)) == (0, 1, 2)


def test_less_gpu_memory_starts_with_the_first_profile_that_fits():
    free = NEEDS[1].vram_mib + IDLE_MARGIN_MIB

    assert usable_profiles(NEEDS, _budget(free)) == (1, 2)


def test_a_running_training_keeps_a_bigger_gpu_margin():
    free = NEEDS[1].vram_mib + TRAINING_MARGIN_MIB - 1

    assert usable_profiles(NEEDS, _budget(free))[0] == 1
    assert usable_profiles(NEEDS, _budget(free, training=True))[0] == 2
    assert (margin_mib(False), margin_mib(True)) == (IDLE_MARGIN_MIB, TRAINING_MARGIN_MIB)


def test_too_little_gpu_memory_fits_nothing_and_says_how_much_is_needed():
    budget = _budget(300)

    assert usable_profiles(NEEDS, budget) == ()
    needed = NEEDS[-1].vram_mib + IDLE_MARGIN_MIB
    assert not_enough_memory(NEEDS, budget) == (
        f"Not enough free GPU memory: about {needed} MiB needed, 300 MiB free."
    )


def test_unknown_free_gpu_memory_starts_from_the_first_profile():
    assert usable_profiles(NEEDS, _budget(None, training=True)) == (0, 1, 2)


def test_profiles_that_do_not_fit_the_available_ram_are_left_out():
    # Lighter GPU profiles keep DiT blocks in RAM: they are the ones that need more of it.
    standard, block_swap, low_vram = (need.ram_mib for need in NEEDS)
    available = low_vram + IDLE_RAM_MARGIN_MIB

    assert block_swap > available - IDLE_RAM_MARGIN_MIB > standard
    assert usable_profiles(NEEDS, _budget(11386, available)) == (0, 2)


def test_a_running_training_keeps_a_bigger_ram_margin():
    available = NEEDS[0].ram_mib + TRAINING_RAM_MARGIN_MIB - 1

    assert usable_profiles(NEEDS, _budget(11386, available)) == (0,)
    assert usable_profiles(NEEDS, _budget(11386, available, training=True)) == ()
    assert (ram_margin_mib(False), ram_margin_mib(True)) == (
        IDLE_RAM_MARGIN_MIB,
        TRAINING_RAM_MARGIN_MIB,
    )


def test_when_only_the_ram_is_short_the_message_says_so():
    # GPU memory for block_swap only, RAM for neither block_swap nor low_vram: generating
    # anyway would push the system into swap (on a 16 GB machine it froze, 2026-10-27).
    free = NEEDS[1].vram_mib + TRAINING_MARGIN_MIB
    budget = _budget(free, 5000, training=True)

    assert usable_profiles(NEEDS, budget) == ()
    needed = min(NEEDS[1].ram_mib, NEEDS[2].ram_mib) + TRAINING_RAM_MARGIN_MIB
    assert not_enough_memory(NEEDS, budget) == (
        f"Not enough free system RAM: about {needed} MiB needed, 5000 MiB available. "
        "Close other programs, or wait for the training to end."
    )


def test_unknown_available_ram_does_not_block():
    assert usable_profiles(NEEDS, _budget(11386, None)) == (0, 1, 2)


def test_a_profile_fits_the_ram_exactly_at_its_figure_plus_the_margin():
    exact = NEEDS[0].ram_mib + IDLE_RAM_MARGIN_MIB

    assert usable_profiles(NEEDS, _budget(11386, exact)) == (0,)
    assert usable_profiles(NEEDS, _budget(11386, exact - 1)) == ()


def test_the_starting_profile_alone_can_be_left_out_for_ram():
    needs = [ProfileNeed(5000, 8000), ProfileNeed(1300, 4000)]

    assert usable_profiles(needs, _budget(11386, 6000)) == (1,)


def test_without_profiles_nothing_is_usable_and_the_message_says_why():
    assert usable_profiles([], _budget(11386)) == ()
    assert not_enough_memory([], _budget(None)) == "No memory profile given."


def test_the_fallback_warning_names_the_profile_tried_next():
    assert next_profile_warning("standard", "low_vram") == (
        "Out of GPU memory with standard, trying low_vram."
    )


def test_lighter_profiles_left_out_for_ram_are_listed():
    budget = _budget(11386, NEEDS[0].ram_mib + IDLE_RAM_MARGIN_MIB)

    assert lighter_skipped_for_ram(NEEDS, budget) == (1, 2)
    assert lighter_skipped_for_ram(NEEDS, _budget(11386)) == ()
    assert lighter_skipped_for_ram(NEEDS, _budget(300)) == ()


def test_running_out_after_the_lightest_profile_says_so():
    assert all_out_of_gpu_memory("low_vram", (), 9000) == (
        "Out of GPU memory even with the lightest profile."
    )


def test_running_out_with_lighter_profiles_left_out_blames_the_ram():
    # The user must free RAM, not GPU memory: a lighter profile exists but did not fit it.
    assert all_out_of_gpu_memory("standard", ("block_swap", "low_vram"), 5000) == (
        "Out of GPU memory with standard; the lighter profiles (block_swap, low_vram) need "
        "more system RAM than the 5000 MiB available. Close other programs, or wait for the "
        "training to end."
    )


@pytest.mark.parametrize(
    "platform, expected",
    [("linux", "System RAM could not be read from /proc/meminfo: profiles are chosen by "
               "GPU memory only."),
     ("win32", None),
     ("darwin", None)],
)  # fmt: skip
def test_unreadable_ram_is_a_warning_only_where_meminfo_should_exist(platform, expected):
    assert ram_unknown_warning(platform) == expected


MEMINFO = """MemTotal:       16276864 kB
MemFree:         8853504 kB
MemAvailable:   12512256 kB
Buffers:           70000 kB
"""


@pytest.mark.parametrize(
    "text, expected",
    [(MEMINFO, 12219),
     ("MemTotal: 16276864 kB\n", None),
     ("MemAvailable: lots kB\n", None),
     ("", None)],
)  # fmt: skip
def test_available_ram_is_read_from_meminfo(text, expected):
    assert available_ram_mib(text) == expected
