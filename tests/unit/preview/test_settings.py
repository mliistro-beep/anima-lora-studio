import json

import pytest
from pydantic import ValidationError

from anima_trainer.preview.settings import (
    MAX_PROMPTS,
    PreviewSettings,
    load_preview_settings,
    save_preview_settings,
)


def test_defaults_are_off_with_er_sde_25_steps_and_auto_size():
    settings = PreviewSettings()

    assert settings.enabled is False
    assert settings.prompts == ()
    assert (settings.sampler, settings.steps, settings.cfg_scale) == ("er_sde", 25, 4.0)
    assert (settings.auto_size, settings.width, settings.height) == (True, 1024, 1024)
    assert (settings.every_n_epochs, settings.sample_at_first) == (1, False)


def test_prompts_are_stripped_and_blank_ones_dropped():
    settings = PreviewSettings(prompts=("  a girl  ", "", "   ", "a cat"))

    assert settings.prompts == ("a girl", "a cat")


def test_too_many_prompts_are_rejected():
    with pytest.raises(ValidationError):
        PreviewSettings(prompts=tuple(f"p{i}" for i in range(MAX_PROMPTS + 1)))


@pytest.mark.parametrize("side", [100, 1000, 4096, 0])
def test_sides_must_be_multiples_of_16_in_range(side):
    with pytest.raises(ValidationError):
        PreviewSettings(width=side)


@pytest.mark.parametrize(
    "field, value",
    [("steps", 0), ("steps", 201), ("cfg_scale", 0.5), ("flow_shift", 0), ("seed", -1),
     ("every_n_epochs", 0), ("sampler", "dpmpp"), ("steps", True), ("cfg_scale", True),
     ("flow_shift", False)],
)  # fmt: skip
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        PreviewSettings(**{field: value})


def test_is_frozen_and_with_changes_returns_a_new_validated_copy():
    settings = PreviewSettings()

    changed = settings.with_changes(enabled=True, prompts=["x"])

    assert changed.enabled and changed.prompts == ("x",)
    assert settings.enabled is False
    with pytest.raises(ValidationError):
        settings.with_changes(steps=0)


def test_missing_prompts_only_matter_when_enabled():
    assert PreviewSettings().problems() == ()
    assert PreviewSettings(enabled=True).problems() == (
        "Previews are on but there is no preview prompt.",
    )
    assert PreviewSettings(enabled=True, prompts=("a",)).problems() == ()


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "preview.json"
    settings = PreviewSettings(enabled=True, prompts=("a", "b"), sampler="euler", seed=7)

    save_preview_settings(path, settings)
    loaded = load_preview_settings(path)

    assert loaded.settings == settings
    assert loaded.warnings == ()
    assert json.loads(path.read_text())["prompts"] == ["a", "b"]


def test_load_missing_file_gives_defaults(tmp_path):
    loaded = load_preview_settings(tmp_path / "preview.json")

    assert loaded.settings == PreviewSettings()
    assert loaded.warnings == ()


def test_load_invalid_value_keeps_the_others_and_warns(tmp_path):
    path = tmp_path / "preview.json"
    path.write_text(json.dumps({"enabled": True, "steps": 0, "seed": 9}))

    loaded = load_preview_settings(path)

    assert (loaded.settings.enabled, loaded.settings.seed, loaded.settings.steps) == (True, 9, 25)
    assert loaded.warnings == ("Invalid value for 'steps' (0), using default.",)
