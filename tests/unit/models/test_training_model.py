from pathlib import Path

import pytest

from anima_trainer.models.training_model import (
    TRAINABLE_BLOCKS,
    TrainingModelRejected,
    check_training_model,
    training_dit,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings
from tests.safetensors_helpers import kohya_blocks, write_anima_dit, write_safetensors


def _model(tmp_path: Path, **options) -> Path:
    return write_anima_dit(tmp_path / "finetune.safetensors", **options)


@pytest.mark.parametrize("prefix", ["net.", "model.diffusion_model."])
def test_an_anima_finetune_is_accepted_with_either_prefix(tmp_path, prefix):
    description = check_training_model(_model(tmp_path, prefix=prefix))

    assert "finetune.safetensors" in description
    assert "28 blocks" in description
    assert "BF16" in description


@pytest.mark.parametrize("dtype", ["F16", "F32"])
def test_half_and_full_precision_files_are_accepted(tmp_path, dtype):
    assert dtype in check_training_model(_model(tmp_path, dtype=dtype))


def test_a_lora_is_refused(tmp_path):
    path = write_safetensors(tmp_path / "style.safetensors", kohya_blocks(TRAINABLE_BLOCKS))

    with pytest.raises(TrainingModelRejected, match="style.safetensors is a LoRA"):
        check_training_model(path)


@pytest.mark.parametrize("blocks", [40, 52])
def test_a_larger_anima_is_refused_because_sd_scripts_trains_only_28_blocks(tmp_path, blocks):
    path = _model(tmp_path, blocks=blocks, changes={"net.pos_embedder.seq": (128,)})

    with pytest.raises(TrainingModelRejected, match=rf"{blocks} blocks.*28"):
        check_training_model(path)


@pytest.mark.parametrize(
    "extra",
    [
        "net.anima_v2_connector.proj.weight",
        "text_encoders.qwen3_06b.model.embed_tokens.weight",
        "vae.decoder.conv_in.weight",
    ],
)
def test_parts_sd_scripts_cannot_load_are_refused_by_name(tmp_path, extra):
    path = _model(tmp_path, changes={extra: (1,)})

    with pytest.raises(TrainingModelRejected, match=extra.split(".")[0]):
        check_training_model(path)


def test_a_file_mixing_the_two_prefixes_is_refused(tmp_path):
    changes = {
        "net.final_layer.linear.weight": None,
        "model.diffusion_model.final_layer.linear.weight": (64, 2048),
    }

    with pytest.raises(TrainingModelRejected, match="prefix"):
        check_training_model(_model(tmp_path, changes=changes))


def test_an_fp8_file_is_refused(tmp_path):
    with pytest.raises(TrainingModelRejected, match="F8_E4M3"):
        check_training_model(_model(tmp_path, dtype="F8_E4M3"))


def test_a_model_with_a_whole_block_missing_is_refused(tmp_path):
    from tests.safetensors_helpers import anima_dit_shapes  # noqa: PLC0415

    changes = {name: None for name in anima_dit_shapes() if name.startswith("net.blocks.5.")}

    with pytest.raises(TrainingModelRejected, match="block 5"):
        check_training_model(_model(tmp_path, changes=changes))


@pytest.mark.parametrize(
    "missing",
    [
        "net.blocks.3.mlp.layer2.weight",
        "net.llm_adapter.embed.weight",
        "net.t_embedding_norm.weight",
    ],
)
def test_a_missing_tensor_is_refused_by_name_as_sd_scripts_would(tmp_path, missing):
    path = _model(tmp_path, changes={missing: None})

    with pytest.raises(TrainingModelRejected, match=rf"missing.*{missing.removeprefix('net.')}"):
        check_training_model(path)


def test_an_unexpected_tensor_inside_a_part_is_refused_by_name(tmp_path):
    path = _model(tmp_path, changes={"net.blocks.3.foo.weight": (8,)})

    with pytest.raises(TrainingModelRejected, match=r"unexpected.*blocks\.3\.foo\.weight"):
        check_training_model(path)


def test_a_tensor_of_another_shape_is_refused_by_name(tmp_path):
    path = _model(tmp_path, changes={"net.x_embedder.proj.1.weight": (3072, 68)})

    with pytest.raises(TrainingModelRejected, match=r"x_embedder\.proj\.1\.weight.*3072"):
        check_training_model(path)


def test_a_file_without_transformer_blocks_is_not_an_anima_model(tmp_path):
    path = write_safetensors(tmp_path / "vae.safetensors", ["decoder.conv_in.weight"])

    with pytest.raises(TrainingModelRejected, match="not an Anima model"):
        check_training_model(path)


def test_a_file_that_is_not_safetensors_is_refused(tmp_path):
    path = tmp_path / "notes.safetensors"
    path.write_bytes(b"not a model")

    with pytest.raises(TrainingModelRejected, match="notes.safetensors"):
        check_training_model(path)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(TrainingModelRejected, match="not found"):
        check_training_model(tmp_path / "gone.safetensors")


def test_a_folder_is_refused(tmp_path):
    with pytest.raises(TrainingModelRejected, match="not a file"):
        check_training_model(tmp_path)


def test_without_a_model_file_the_base_model_is_trained(tmp_path):
    paths = RuntimePaths(tmp_path / "data")
    settings = TrainingSettings(base_model="anima-preview")

    assert training_dit(settings, paths) == paths.dit_model("anima-preview")


def test_a_model_file_wins_over_the_base_model(tmp_path):
    paths = RuntimePaths(tmp_path / "data")
    own = tmp_path / "finetune.safetensors"

    assert training_dit(TrainingSettings(model_file=str(own)), paths) == own


def test_a_relative_model_file_is_read_from_the_data_folder_like_sd_scripts(tmp_path):
    paths = RuntimePaths(tmp_path / "data")

    dit = training_dit(TrainingSettings(model_file="models/mine.safetensors"), paths)

    assert dit == tmp_path / "data" / "models" / "mine.safetensors"
