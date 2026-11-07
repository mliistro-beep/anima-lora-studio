"""From the TUI generator's fields to GeneratorSettings, and the LoRA list edits."""

import json

from anima_trainer.cli.generator_values import (
    add_loras,
    gpu_index_for,
    image_defaults,
    settings_from_fields,
)
from anima_trainer.preview.generator_settings import MAX_LORAS, GeneratorSettings, LoraEntry


def _fields(**changes):
    return {**GeneratorSettings(prompt="a cat").model_dump(), **changes}


def test_typed_values_become_settings():
    settings = settings_from_fields(
        _fields(steps="30", cfg_scale="7.5", pre_prompt=" score_9 ", width="832")
    )
    assert isinstance(settings, GeneratorSettings)
    assert (settings.steps, settings.cfg_scale, settings.width) == (30, 7.5, 832)
    assert settings.pre_prompt == "score_9"


def test_wrong_values_give_one_error_per_field_with_its_label():
    errors = settings_from_fields(_fields(steps="many", width="1000"))
    assert isinstance(errors, tuple)
    assert len(errors) == 2
    assert errors[0].startswith("ERROR: Width: ")
    assert errors[1].startswith("ERROR: Steps: ")


def test_a_lora_strength_out_of_range_names_the_lora():
    loras = ({"path": "/l/a.safetensors", "strength": "3", "enabled": True, "trigger": ""},)
    errors = settings_from_fields(_fields(loras=loras))
    assert isinstance(errors, tuple)
    assert errors[0].startswith("ERROR: LoRAs.0.strength: ")


def test_added_files_get_the_trigger_found_in_them():
    entries, warnings = add_loras(
        (), ("/l/a.safetensors",), {"/l/a.safetensors": "t-a"}.__getitem__
    )
    assert entries == (LoraEntry(path="/l/a.safetensors", trigger="t-a"),)
    assert warnings == ()


def test_files_beyond_the_maximum_are_left_out_with_a_warning():
    full = tuple(LoraEntry(path=f"/l/{n}.safetensors") for n in range(MAX_LORAS - 1))
    entries, warnings = add_loras(full, ("/l/x.safetensors", "/l/y.safetensors"), lambda _p: "")
    assert len(entries) == MAX_LORAS
    assert entries[-1].path == "/l/x.safetensors"
    assert warnings == (f"WARN: At most {MAX_LORAS} LoRAs: /l/y.safetensors was not added.",)


def test_image_defaults_are_the_fields_reset_puts_back_and_not_prompts_loras_or_folder():
    defaults = image_defaults()
    assert defaults["steps"] == GeneratorSettings().steps
    assert defaults["cfg_scale"] == GeneratorSettings().cfg_scale
    assert {"width", "height", "sampler", "seed", "flow_shift"} <= defaults.keys()
    assert not {"prompt", "pre_prompt", "loras", "model_file", "output_folder"} & defaults.keys()


def test_the_gpu_is_the_one_saved_by_the_window(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"gpu_index": "1"}))
    assert gpu_index_for(config) == "1"


def test_without_saved_settings_the_default_gpu_is_used(tmp_path):
    assert gpu_index_for(tmp_path / "config.json") == "0"


def test_a_file_already_in_the_list_is_not_added_twice():
    entries = (LoraEntry(path="/l/a.safetensors"),)
    added, notes = add_loras(entries, ("/l/a.safetensors",), lambda _p: "")
    assert added == entries
    assert notes == ("INFO: /l/a.safetensors is already in the list.",)
