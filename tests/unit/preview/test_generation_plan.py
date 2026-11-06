"""From the Generator settings to the images to generate, or the reasons not to."""

from datetime import datetime

import pytest

from anima_trainer.preview.checkpoints import Checkpoint
from anima_trainer.preview.generation_plan import GenerationTarget, plan_generation
from anima_trainer.preview.generator_settings import GeneratorSettings, LoraEntry
from anima_trainer.preview.inference_command import LoraUse
from anima_trainer.runtime.paths import RuntimePaths
from tests.safetensors_helpers import checkpoint_keys, kohya_blocks, write_safetensors

NOW = datetime(2026, 11, 6, 21, 0, 0)


@pytest.fixture
def paths(tmp_path):
    runtime = RuntimePaths(tmp_path / "data")
    runtime.dit_model("anima-base-v1.0").parent.mkdir(parents=True)
    write_safetensors(runtime.dit_model("anima-base-v1.0"), checkpoint_keys(28))
    return runtime


def _target(checkpoints=(None,), project=""):
    return GenerationTarget(project=project, checkpoints=checkpoints)


def _lora(tmp_path, name, blocks=28):
    return write_safetensors(tmp_path / name, kohya_blocks(blocks))


def test_the_base_model_alone_goes_to_the_default_folder(paths):
    (request,) = plan_generation(GeneratorSettings(prompt="a cat"), paths, _target(), NOW)

    assert request.dit == paths.dit_model("anima-base-v1.0")
    assert request.loras == ()
    assert request.output.parent == paths.root / "generated"
    assert request.output.name.startswith("image_no-lora_")
    assert request.prompt == "a cat"


def test_the_chosen_folder_and_the_project_name_are_used(paths, tmp_path):
    settings = GeneratorSettings(prompt="a cat", output_folder=str(tmp_path / "pictures"))

    (request,) = plan_generation(settings, paths, _target(project="my_lora"), NOW)

    assert request.output.parent == tmp_path / "pictures"
    assert request.output.name.startswith("my_lora_")


def test_the_enabled_loras_follow_the_project_checkpoint(paths, tmp_path):
    first, second = _lora(tmp_path, "a.safetensors"), _lora(tmp_path, "b.safetensors")
    settings = GeneratorSettings(
        prompt="a cat",
        loras=(
            LoraEntry(path=str(first), strength=0.5),
            LoraEntry(path=str(second), enabled=False),
        ),
        lora_multiplier=0.8,
    )
    checkpoint = Checkpoint(tmp_path / "out" / "p-000002.safetensors", 2)

    (request,) = plan_generation(settings, paths, _target((checkpoint,), "p"), NOW)

    assert request.loras == (LoraUse(checkpoint.path, 0.8), LoraUse(first, 0.5))
    assert "epoch-2" in request.output.name


def test_one_request_per_checkpoint_with_the_same_seed_and_loras(paths, tmp_path):
    extra = _lora(tmp_path, "style.safetensors")
    settings = GeneratorSettings(prompt="a cat", seed=7, loras=(LoraEntry(path=str(extra)),))
    checkpoints = tuple(Checkpoint(tmp_path / f"p-00000{n}.safetensors", n) for n in (1, 2))

    requests = plan_generation(settings, paths, _target(checkpoints, "p"), NOW)

    assert [request.loras[0].path for request in requests] == [c.path for c in checkpoints]
    assert {request.loras[1].path for request in requests} == {extra}
    assert {request.seed for request in requests} == {7}


def test_a_model_file_of_the_user_replaces_the_catalog_model(paths, tmp_path):
    own = write_safetensors(
        tmp_path / "merge.safetensors", checkpoint_keys(28, "model.diffusion_model.")
    )
    settings = GeneratorSettings(prompt="a cat", model_file=str(own))

    (request,) = plan_generation(settings, paths, _target(), NOW)

    assert request.dit == own


def test_the_catalog_model_chosen_in_the_generator_is_used(paths):
    settings = GeneratorSettings(prompt="a cat", base_model="anima-preview")

    (request,) = plan_generation(settings, paths, _target(), NOW)

    assert request.dit == paths.dit_model("anima-preview")


def test_without_a_prompt_nothing_is_planned(paths):
    assert plan_generation(GeneratorSettings(), paths, _target(), NOW) == (
        "ERROR: Write a prompt first.",
    )


def test_a_model_file_that_is_not_a_model_is_an_error(paths, tmp_path):
    lora = _lora(tmp_path, "style.safetensors")
    settings = GeneratorSettings(prompt="a cat", model_file=str(lora))

    (line,) = plan_generation(settings, paths, _target(), NOW)

    assert line.startswith("ERROR: ") and "is a LoRA" in line


def test_a_lora_that_cannot_be_applied_is_an_error_and_nothing_is_planned(paths, tmp_path):
    good = _lora(tmp_path, "good.safetensors")
    bad = write_safetensors(tmp_path / "loha.safetensors", ["lora_unet_blocks_0_mlp.hada_w1_a"])
    settings = GeneratorSettings(
        prompt="a cat", loras=(LoraEntry(path=str(good)), LoraEntry(path=str(bad)))
    )

    lines = plan_generation(settings, paths, _target(), NOW)

    assert lines == (
        f"ERROR: {bad.name} is a LoHa file: only plain LoRAs (kohya or PEFT) can be applied.",
    )


def test_a_lora_for_a_larger_model_is_an_error(paths, tmp_path):
    big = _lora(tmp_path, "big.safetensors", blocks=40)
    settings = GeneratorSettings(prompt="a cat", loras=(LoraEntry(path=str(big)),))

    (line,) = plan_generation(settings, paths, _target(), NOW)

    assert line.startswith("ERROR: big.safetensors changes block 39")


def test_warnings_about_a_lora_are_left_to_the_generation_log(paths, tmp_path):
    keys = kohya_blocks(28) + ["lora_te_layers_0_mlp.lora_down.weight"]
    lora = write_safetensors(tmp_path / "te.safetensors", keys)
    settings = GeneratorSettings(prompt="a cat", loras=(LoraEntry(path=str(lora)),))

    (request,) = plan_generation(settings, paths, _target(), NOW)

    assert request.loras == (LoraUse(lora, 1.0),)


def test_a_disabled_lora_is_not_even_read(paths, tmp_path):
    settings = GeneratorSettings(
        prompt="a cat", loras=(LoraEntry(path=str(tmp_path / "gone.safetensors"), enabled=False),)
    )

    (request,) = plan_generation(settings, paths, _target(), NOW)

    assert request.loras == ()


def test_the_catalog_model_is_read_only_when_loras_are_added(paths, tmp_path):
    paths.dit_model("anima-base-v1.0").write_bytes(b"")  # as in a fresh data folder
    plain = plan_generation(GeneratorSettings(prompt="a cat"), paths, _target(), NOW)
    with_lora = plan_generation(
        GeneratorSettings(
            prompt="a cat", loras=(LoraEntry(path=str(_lora(tmp_path, "a.safetensors"))),)
        ),
        paths,
        _target(),
        NOW,
    )

    assert len(plain) == 1 and not isinstance(plain[0], str)
    assert len(with_lora) == 1 and with_lora[0].startswith("ERROR: ")
