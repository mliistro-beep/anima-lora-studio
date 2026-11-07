"""The text chunks generate.py writes into the PNG (runtime_scripts/png_text.py)."""

from anima_trainer.runtime_scripts.png_text import PARAMETERS_KEY, SETTINGS_KEY, png_texts


def test_both_texts_are_written_under_their_keys():
    assert png_texts("a cat\nSteps: 25", "{}") == {
        PARAMETERS_KEY: "a cat\nSteps: 25",
        SETTINGS_KEY: "{}",
    }


def test_the_keys_are_the_a1111_one_and_ours():
    assert (PARAMETERS_KEY, SETTINGS_KEY) == ("parameters", "anima_lora_studio")


def test_empty_texts_are_left_out():
    assert png_texts("", "") == {}
