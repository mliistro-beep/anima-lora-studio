import pytest

from anima_trainer.system.gpu_choice import gpu_index_from_choice


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
