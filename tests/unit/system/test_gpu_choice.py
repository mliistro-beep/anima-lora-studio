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
