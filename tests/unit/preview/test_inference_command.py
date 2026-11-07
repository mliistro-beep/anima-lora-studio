import argparse
import json
from datetime import datetime
from pathlib import Path

import pytest

from anima_trainer.preview.image_metadata import ImageMetadata
from anima_trainer.preview.inference_command import (
    GenerationRequest,
    LoraUse,
    build_generation_launch,
    default_generated_folder,
    output_image_path,
    project_generated_folder,
)
from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.vram_profile import (
    IDLE_MARGIN_MIB,
    IDLE_RAM_MARGIN_MIB,
    PROFILES,
    TRAINING_MARGIN_MIB,
    TRAINING_RAM_MARGIN_MIB,
    estimate_mib,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime_scripts import GENERATE

ROOT = Path("/data/anima-trainer")
PATHS = RuntimePaths(ROOT, platform="posix")
NOW = datetime(2026, 10, 27, 12, 30, 5)


def _request(**changes):
    values = {
        "dit": Path("/models/merge.safetensors"),
        "loras": (LoraUse(Path("/out/my_lora-000002.safetensors"), 0.8),),
        "prompt": "a girl",
        "negative_prompt": "lowres",
        "size": SampleSize(1024, 768),
        "steps": 25,
        "cfg_scale": 4.0,
        "flow_shift": 3.0,
        "seed": 7,
        "sampler": "er_sde",
        "output": Path("/out/generated/x.png"),
    }
    return GenerationRequest(**{**values, **changes})


def _option(arguments, name):
    return arguments[arguments.index(name) + 1]


def test_launch_runs_generate_py_with_the_training_venv_and_models():
    spec = build_generation_launch(PATHS, _request(), PROFILES[1:], "0", training=False)

    assert spec.program == PATHS.venv_python
    assert spec.arguments[:2] == ("-B", str(GENERATE))
    assert spec.working_dir == ROOT
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert spec.env["ANIMA_TRAINER_SD_SCRIPTS"] == str(PATHS.sd_scripts_dir)
    # less memory held between the text encoder, DiT and VAE phases
    assert spec.env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    args = spec.arguments
    assert _option(args, "--dit") == "/models/merge.safetensors"
    assert _option(args, "--qwen3") == str(PATHS.qwen3_model)
    assert _option(args, "--vae") == str(PATHS.vae_model)
    assert _option(args, "--lora") == "/out/my_lora-000002.safetensors"
    assert (_option(args, "--lora_multiplier"), _option(args, "--sampler")) == ("0.8", "er_sde")
    assert (_option(args, "--steps"), _option(args, "--cfg"), _option(args, "--seed")) == (
        "25", "4.0", "7",
    )  # fmt: skip
    assert _option(args, "--output") == "/out/generated/x.png"


def test_profiles_are_passed_in_order_with_the_image_size():
    spec = build_generation_launch(PATHS, _request(), PROFILES, "0", training=False)

    profiles = json.loads(_option(spec.arguments, "--profiles"))
    assert [entry["name"] for entry in profiles] == [profile.name for profile in PROFILES]
    assert (profiles[0]["width"], profiles[0]["height"]) == (1024, 768)
    assert all((entry["width"], entry["height"]) == (1024, 768) for entry in profiles)
    assert set(profiles[0]) == {
        "name", "vae_tiling", "blocks_to_swap", "text_encoder_cpu", "width", "height",
        "estimate_mib", "ram_mib",
    }  # fmt: skip
    assert profiles[0]["estimate_mib"] == estimate_mib(PROFILES[0], SampleSize(1024, 768))
    assert [entry["ram_mib"] for entry in profiles] == [profile.ram_mib for profile in PROFILES]


def test_a_fallback_that_would_not_save_memory_is_not_passed():
    large = _request(size=SampleSize(1536, 1024))  # text encoder no longer the peak

    spec = build_generation_launch(PATHS, large, PROFILES, "0", training=False)

    profiles = json.loads(_option(spec.arguments, "--profiles"))
    assert [entry["name"] for entry in profiles] == ["standard", "block_swap"]


def test_the_margin_is_larger_while_a_training_runs():
    idle = build_generation_launch(PATHS, _request(), PROFILES, "0", training=False)
    training = build_generation_launch(PATHS, _request(), PROFILES, "0", training=True)

    assert _option(idle.arguments, "--margin_mib") == str(IDLE_MARGIN_MIB)
    assert _option(training.arguments, "--margin_mib") == str(TRAINING_MARGIN_MIB)
    assert _option(idle.arguments, "--ram_margin_mib") == str(IDLE_RAM_MARGIN_MIB)
    assert _option(training.arguments, "--ram_margin_mib") == str(TRAINING_RAM_MARGIN_MIB)


def test_without_a_lora_no_lora_option_is_passed():
    spec = build_generation_launch(PATHS, _request(loras=()), PROFILES, "0", training=False)

    assert "--lora" not in spec.arguments
    assert "--lora_multiplier" not in spec.arguments


def test_every_lora_is_passed_with_its_own_strength_in_order():
    loras = (LoraUse(Path("/l/a.safetensors"), 1.0), LoraUse(Path("/l/b.safetensors"), -0.5))

    spec = build_generation_launch(PATHS, _request(loras=loras), PROFILES, "0", training=False)

    args = list(spec.arguments)
    start = args.index("--lora")
    assert args[start : start + 8] == [
        "--lora", "/l/a.safetensors", "--lora_multiplier", "1.0",
        "--lora", "/l/b.safetensors", "--lora_multiplier", "-0.5",
    ]  # fmt: skip


def test_output_path_names_project_checkpoint_time_and_seed():
    path = output_image_path(Path("/out/generated"), "my_lora", "Epoch 2", 7, NOW)

    assert path == Path("/out/generated/my_lora_epoch-2_20261027-123005_7.png")
    assert output_image_path(Path("/out"), "p", "No LoRA", 1, NOW).name == (
        "p_no-lora_20261027-123005_1.png"
    )


def test_an_existing_image_is_never_overwritten(tmp_path):
    first = output_image_path(tmp_path / "pictures", "p", "Final", 7, NOW)
    assert first.parent == tmp_path / "pictures"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"png")
    first.with_name(first.stem + "_2.png").write_bytes(b"png")

    path = output_image_path(tmp_path / "pictures", "p", "Final", 7, NOW)

    assert path == first.with_name(first.stem + "_3.png")


@pytest.mark.parametrize("text", ["-low quality", "--steps 3", "a=b", ""])
def test_prompts_reach_argparse_whatever_they_start_with(text):
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    request = _request(prompt=text, negative_prompt=text)

    arguments = build_generation_launch(PATHS, request, PROFILES, "0", False).arguments
    options = [item for item in arguments if item.startswith(("--prompt=", "--negative_prompt="))]

    parsed = parser.parse_args(options)
    assert (parsed.prompt, parsed.negative_prompt) == (text, text)


def test_the_default_folder_is_in_the_data_folder_and_a_project_uses_its_output():
    assert default_generated_folder(PATHS) == ROOT / "generated"
    assert project_generated_folder(Path("/out")) == Path("/out/generated")


def test_a_name_already_planned_gets_the_next_number(tmp_path):
    first = output_image_path(tmp_path, "p", "LoRAs", 42, NOW)
    second = output_image_path(tmp_path, "p", "LoRAs", 42, NOW, taken={first})
    assert second.name == first.name.replace(".png", "_2.png")


def test_the_png_texts_are_passed_when_the_request_has_them():
    metadata = ImageMetadata(parameters="-a cat\nSteps: 25", settings='{"entry": {}}')
    spec = build_generation_launch(PATHS, _request(metadata=metadata), PROFILES, "0", False)
    assert "--png_parameters=-a cat\nSteps: 25" in spec.arguments
    assert '--png_settings={"entry": {}}' in spec.arguments


def test_without_metadata_no_png_texts_are_passed():
    spec = build_generation_launch(PATHS, _request(), PROFILES, "0", False)
    assert not any(part.startswith("--png_") for part in spec.arguments)


def test_an_image_of_a_group_has_its_number_before_the_seed(tmp_path):
    path = output_image_path(tmp_path, "image", "LoRAs", 1838785401, NOW, number=3)
    assert path.name.endswith("_03_1838785401.png")
