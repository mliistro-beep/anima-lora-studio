import pytest
from pydantic import ValidationError

from anima_trainer.preview.generator_settings import (
    GeneratorSettings,
    load_generator_settings,
    save_generator_settings,
)


def test_defaults_match_the_preview_defaults():
    settings = GeneratorSettings()

    assert (settings.sampler, settings.steps, settings.cfg_scale, settings.seed) == (
        "er_sde", 25, 4.0, 42,
    )  # fmt: skip
    assert (settings.width, settings.height, settings.lora_multiplier) == (1024, 1024, 1.0)


@pytest.mark.parametrize(
    "field, value",
    [("width", 1000), ("height", 4096), ("lora_multiplier", -0.1), ("lora_multiplier", 2.5),
     ("steps", 0), ("cfg_scale", True), ("sampler", "x")],
)  # fmt: skip
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        GeneratorSettings(**{field: value})


def test_prompt_is_required_to_generate():
    assert GeneratorSettings().problems() == ("Write a prompt first.",)
    assert GeneratorSettings(prompt="  a girl ").problems() == ()
    assert GeneratorSettings(prompt="  a girl ").prompt == "a girl"


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "generator.json"
    settings = GeneratorSettings(prompt="a", width=832, height=1216, lora_multiplier=0.7)

    save_generator_settings(path, settings)

    assert load_generator_settings(path).settings == settings
