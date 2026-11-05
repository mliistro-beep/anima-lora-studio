"""A preset file is checked in full before anything can use it."""

import pytest

from anima_trainer.presets.preset_file import (
    KEPT_FIELDS,
    PRESET_FIELDS,
    PresetError,
    parse_preset,
)
from anima_trainer.settings.schema import TrainingSettings

VALID = """
name = "Mine"
description = "About 40 images"

[settings]
network_dim = 24
learning_rate = 0.0002
"""


def _preset_with(setting_line):
    return f'name = "Mine"\n[settings]\n{setting_line}\n'


def test_a_valid_file_gives_its_name_description_and_values():
    preset = parse_preset(VALID)
    assert preset.name == "Mine"
    assert preset.description == "About 40 images"
    assert dict(preset.values) == {"network_dim": 24, "learning_rate": 0.0002}


def test_the_description_and_the_values_are_optional():
    preset = parse_preset('name = "Only a name"')
    assert preset.description == ""
    assert dict(preset.values) == {}


def test_the_values_cannot_be_changed_afterwards():
    preset = parse_preset(VALID)
    with pytest.raises(TypeError):
        preset.values["network_dim"] = 1


def test_broken_toml_is_rejected():
    with pytest.raises(PresetError, match="not valid TOML"):
        parse_preset("name = ")


def test_a_missing_name_is_rejected():
    with pytest.raises(PresetError, match="name"):
        parse_preset("[settings]\nnetwork_dim = 8\n")


@pytest.mark.parametrize("name", ['""', '"   "', '"bad\\u001b[31m"', "12"])
def test_a_name_that_cannot_be_shown_is_rejected(name):
    with pytest.raises(PresetError, match="name"):
        parse_preset(f"name = {name}")


def test_control_characters_in_the_description_are_rejected():
    with pytest.raises(PresetError, match="description"):
        parse_preset('name = "x"\ndescription = "bad\\u001b[31m"')


def test_an_unknown_top_level_key_is_rejected():
    with pytest.raises(PresetError, match="author"):
        parse_preset('name = "x"\nauthor = "me"')


def test_an_unknown_setting_is_rejected():
    with pytest.raises(PresetError, match="network_size"):
        parse_preset(_preset_with("network_size = 8"))


@pytest.mark.parametrize(
    "line",
    [
        'project_name = "x"',
        'image_directory = "/x"',
        'output_directory = "/x"',
        "repeats = 10",
        "resolution = 1024",
    ],
)
def test_values_chosen_for_each_training_are_rejected(line):
    with pytest.raises(PresetError, match="each training"):
        parse_preset(_preset_with(line))


@pytest.mark.parametrize(
    "line",
    [
        'gpu_index = "1"',
        "num_cpu_threads_per_process = 4",
        "cache_latents = false",
        "cache_text_encoder_outputs = false",
        "vae_chunk_size = 32",
        "vae_disable_cache = false",
    ],
)
def test_values_of_this_machine_are_rejected(line):
    with pytest.raises(PresetError, match="this machine"):
        parse_preset(_preset_with(line))


def test_the_app_own_values_are_rejected():
    with pytest.raises(PresetError, match="last_train_config"):
        parse_preset(_preset_with('last_train_config = "x"'))


@pytest.mark.parametrize(
    "line",
    ["network_dim = 0", 'network_dim = "big"', "network_dim = true", "learning_rate = -1.0"],
)
def test_a_value_the_settings_refuse_is_rejected(line):
    with pytest.raises(PresetError, match=line.split()[0]):
        parse_preset(_preset_with(line))


def test_settings_must_be_a_table():
    with pytest.raises(PresetError, match="settings"):
        parse_preset('name = "x"\nsettings = 3')


def test_every_setting_is_either_allowed_or_kept():
    """A new setting must be classified on purpose: it would otherwise reach presets silently."""
    assert set(TrainingSettings.model_fields) == PRESET_FIELDS | KEPT_FIELDS
    assert not PRESET_FIELDS & KEPT_FIELDS


def test_toml_nested_too_deeply_is_rejected():
    with pytest.raises(PresetError, match="nested"):
        parse_preset("x = " + "[" * 5000)


@pytest.mark.parametrize(
    ("line", "field", "expected", "kind"),
    [
        ('network_dim = "32"', "network_dim", 32, int),
        ("network_dim = 32.0", "network_dim", 32, int),
        ("learning_rate = 1", "learning_rate", 1.0, float),
    ],
)
def test_the_values_are_stored_as_the_settings_understand_them(line, field, expected, kind):
    value = parse_preset(_preset_with(line)).values[field]
    assert value == expected
    assert type(value) is kind
