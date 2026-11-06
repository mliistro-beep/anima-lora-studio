"""The LoRA options of generate.py paired with their strengths (runs in the training venv)."""

import pytest

from anima_trainer.runtime_scripts.lora_inputs import pair_loras


def test_each_lora_gets_the_strength_given_after_it():
    assert pair_loras(["/l/a.safetensors", "/l/b.safetensors"], [1.0, -0.5]) == (
        ("/l/a.safetensors", 1.0),
        ("/l/b.safetensors", -0.5),
    )


def test_no_lora_is_no_pair():
    assert pair_loras(None, None) == ()
    assert pair_loras([], []) == ()


@pytest.mark.parametrize(("loras", "strengths"), [(["/l/a.safetensors"], []), ([], [1.0])])
def test_a_lora_without_a_strength_or_the_reverse_is_an_error(loras, strengths):
    with pytest.raises(ValueError, match="--lora"):
        pair_loras(loras, strengths)
