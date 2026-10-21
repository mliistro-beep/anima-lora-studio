import pytest

from anima_trainer.system.gpu_choice import choice_for_index, gpu_index_from_choice


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("0: NVIDIA GeForce RTX 3060", "0"),
        ("1: Some GPU", "1"),
        ("2", "2"),
        ("", "0"),
        (None, "0"),
        (" 3 : spaced", "3"),
    ],
)
def test_gpu_index_from_choice(choice, expected):
    assert gpu_index_from_choice(choice) == expected


@pytest.mark.parametrize("choice", ["CPU (no CUDA detected)", "abc: x", ":"])
def test_non_numeric_choice_falls_back_to_default(choice):
    assert gpu_index_from_choice(choice) == "0"


CHOICES = ("0: NVIDIA A", "1: NVIDIA B")


@pytest.mark.parametrize(
    ("choices", "saved", "expected"),
    [
        (CHOICES, "1", "1: NVIDIA B"),
        (CHOICES, "0", "0: NVIDIA A"),
        (CHOICES, "7", "0: NVIDIA A"),
        (("0", "1"), "1", "1"),
        ((), "2", "2"),
    ],
)
def test_choice_for_saved_index(choices, saved, expected):
    # Like Citron ("<index>:" entry, else the first choice), but bare-index fallback
    # choices match exactly: Citron would silently switch a saved "1" to "0".
    assert choice_for_index(choices, saved) == expected
