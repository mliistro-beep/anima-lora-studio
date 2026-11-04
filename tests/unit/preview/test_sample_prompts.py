from pathlib import Path

from anima_trainer.preview.sample_prompts import (
    SampleSize,
    build_sample_prompts,
    sample_config_keys,
)
from anima_trainer.preview.settings import PreviewSettings


def test_each_prompt_becomes_a_sd_scripts_prompt_dict():
    settings = PreviewSettings(
        enabled=True,
        prompts=("a girl", "a cat --n not a flag"),
        negative_prompt="lowres",
        steps=25,
        cfg_scale=4.5,
        flow_shift=3.0,
        seed=7,
    )

    prompts = build_sample_prompts(settings, SampleSize(768, 512))

    assert prompts == (
        {"prompt": "a girl", "negative_prompt": "lowres", "width": 768, "height": 512,
         "sample_steps": 25, "scale": 4.5, "flow_shift": 3.0, "seed": 7,
         "sample_sampler": "er_sde"},
        {"prompt": "a cat --n not a flag", "negative_prompt": "lowres", "width": 768,
         "height": 512, "sample_steps": 25, "scale": 4.5, "flow_shift": 3.0, "seed": 7,
         "sample_sampler": "er_sde"},
    )  # fmt: skip


def test_config_keys_point_to_the_prompt_file_and_schedule():
    settings = PreviewSettings(enabled=True, prompts=("a",), every_n_epochs=2, sample_at_first=True)

    keys = sample_config_keys(settings, Path("/data/configs/p_samples.json"))

    assert keys == {
        "sample_prompts": "/data/configs/p_samples.json",
        "sample_every_n_epochs": 2,
        "sample_at_first": True,
    }
