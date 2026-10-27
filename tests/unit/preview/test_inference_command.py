import json
from datetime import datetime
from pathlib import Path

from anima_trainer.preview.inference_command import (
    GenerationRequest,
    build_generation_launch,
    output_image_path,
)
from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.vram_profile import (
    IDLE_MARGIN_MIB,
    PROFILES,
    TRAINING_MARGIN_MIB,
    estimate_mib,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime_scripts import GENERATE

ROOT = Path("/data/anima-trainer")
PATHS = RuntimePaths(ROOT, platform="posix")
NOW = datetime(2026, 10, 27, 12, 30, 5)


def _request(**changes):
    values = {
        "base_model": "anima-base-v1.0",
        "lora": Path("/out/my_lora-000002.safetensors"),
        "lora_multiplier": 0.8,
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
    args = spec.arguments
    assert _option(args, "--dit") == str(PATHS.dit_model("anima-base-v1.0"))
    assert _option(args, "--qwen3") == str(PATHS.qwen3_model)
    assert _option(args, "--vae") == str(PATHS.vae_model)
    assert _option(args, "--lora") == "/out/my_lora-000002.safetensors"
    assert (_option(args, "--lora_multiplier"), _option(args, "--sampler")) == ("0.8", "er_sde")
    assert (_option(args, "--steps"), _option(args, "--cfg"), _option(args, "--seed")) == (
        "25", "4.0", "7",
    )  # fmt: skip
    assert _option(args, "--output") == "/out/generated/x.png"


def test_profiles_are_passed_in_order_with_their_fitted_size():
    spec = build_generation_launch(PATHS, _request(), PROFILES, "0", training=False)

    profiles = json.loads(_option(spec.arguments, "--profiles"))
    assert [entry["name"] for entry in profiles] == [profile.name for profile in PROFILES]
    assert (profiles[0]["width"], profiles[0]["height"]) == (1024, 768)
    assert (profiles[-1]["width"], profiles[-1]["height"]) == (512, 384)
    assert set(profiles[0]) == {
        "name", "fp8", "vae_chunk_size", "blocks_to_swap", "text_encoder_cpu", "width", "height",
        "estimate_mib",
    }  # fmt: skip
    assert profiles[0]["estimate_mib"] == estimate_mib(PROFILES[0], SampleSize(1024, 768))


def test_the_margin_is_larger_while_a_training_runs():
    idle = build_generation_launch(PATHS, _request(), PROFILES, "0", training=False)
    training = build_generation_launch(PATHS, _request(), PROFILES, "0", training=True)

    assert _option(idle.arguments, "--margin_mib") == str(IDLE_MARGIN_MIB)
    assert _option(training.arguments, "--margin_mib") == str(TRAINING_MARGIN_MIB)


def test_without_a_lora_no_lora_option_is_passed():
    spec = build_generation_launch(PATHS, _request(lora=None), PROFILES, "0", training=False)

    assert "--lora" not in spec.arguments


def test_output_path_names_project_checkpoint_time_and_seed():
    path = output_image_path(Path("/out"), "my_lora", "Epoch 2", 7, NOW)

    assert path == Path("/out/generated/my_lora_epoch-2_20261027-123005_7.png")
    assert output_image_path(Path("/out"), "p", "No LoRA", 1, NOW).name == (
        "p_no-lora_20261027-123005_1.png"
    )


def test_an_existing_image_is_never_overwritten(tmp_path):
    first = output_image_path(tmp_path, "p", "Final", 7, NOW)
    first.parent.mkdir(parents=True)
    first.write_bytes(b"png")
    first.with_name(first.stem + "_2.png").write_bytes(b"png")

    path = output_image_path(tmp_path, "p", "Final", 7, NOW)

    assert path == first.with_name(first.stem + "_3.png")
