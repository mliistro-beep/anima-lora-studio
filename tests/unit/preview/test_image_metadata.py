"""The text saved in a generated PNG: A1111 `parameters` and the entry to load it again."""

import json
from pathlib import Path

from anima_trainer.preview.generator_settings import GenerationEntry, LoraEntry
from anima_trainer.preview.image_metadata import APP_NAME, image_metadata
from anima_trainer.preview.inference_command import LoraUse

VERSION = "9.9.9"


def _entry(**changes):
    values = {
        "pre_prompt": "score_9",
        "prompt": "a cat",
        "negative_prompt": "blurry",
        "steps": 30,
        "sampler": "euler",
        "cfg_scale": 7.0,
        "flow_shift": 3.0,
        "width": 832,
        "height": 1216,
        "random_seed": True,
        "count": 4,
        "loras": (
            LoraEntry(path="/home/someone/loras/style.safetensors", strength=0.8, trigger="sks"),
            LoraEntry(path="/home/someone/loras/off.safetensors", enabled=False),
        ),
    }
    return GenerationEntry(**{**values, **changes})


def test_parameters_follow_the_a1111_layout_with_the_used_seed():
    text = image_metadata(_entry(), 1234, None, VERSION).parameters
    first, negative, settings = text.split("\n")
    assert first == "score_9, sks, a cat <lora:style:0.8>"
    assert negative == "Negative prompt: blurry"
    assert settings == (
        "Steps: 30, Sampler: euler, CFG scale: 7.0, Seed: 1234, Size: 832x1216, "
        f"Model: anima-base-v1.0, Flow shift: 3.0, Version: {APP_NAME} {VERSION}"
    )


def test_without_a_negative_prompt_its_line_is_left_out():
    text = image_metadata(_entry(negative_prompt=""), 1, None, VERSION).parameters
    assert "Negative prompt" not in text
    assert len(text.split("\n")) == 2


def test_the_project_checkpoint_comes_first_and_a_model_file_is_named_by_its_file():
    checkpoint = LoraUse(Path("/home/someone/out/my_lora-000004.safetensors"), 0.9)
    entry = _entry(model_file="/home/someone/models/my_dit.safetensors")
    text = image_metadata(entry, 1, checkpoint, VERSION).parameters
    assert "a cat <lora:my_lora-000004:0.9> <lora:style:0.8>" in text
    assert "Model: my_dit," in text


def test_no_folder_of_the_user_is_saved():
    checkpoint = LoraUse(Path("/home/someone/out/my_lora.safetensors"), 1.0)
    entry = _entry(model_file="/home/someone/models/my_dit.safetensors")
    metadata = image_metadata(entry, 1, checkpoint, VERSION)
    assert "/home/someone" not in metadata.parameters
    assert "/home/someone" not in metadata.settings


def test_the_settings_load_back_as_the_entry_of_this_very_image():
    checkpoint = LoraUse(Path("/out/my_lora.safetensors"), 0.9)
    saved = json.loads(image_metadata(_entry(), 1234, checkpoint, VERSION).settings)
    assert (saved["generator"], saved["version"]) == (APP_NAME, VERSION)
    entry = GenerationEntry.model_validate(saved["entry"])
    assert (entry.seed, entry.random_seed, entry.count) == (1234, False, 1)
    assert [(lora.path, lora.strength, lora.trigger) for lora in entry.loras] == [
        ("my_lora.safetensors", 0.9, ""),
        ("style.safetensors", 0.8, "sks"),
    ]
    assert (entry.pre_prompt, entry.prompt, entry.sampler) == ("score_9", "a cat", "euler")
