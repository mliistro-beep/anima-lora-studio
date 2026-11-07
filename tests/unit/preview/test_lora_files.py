"""LoRA and checkpoint files seen from their header: format, blocks, and what cannot be used."""

import pytest

from anima_trainer.preview.lora_files import (
    CheckpointFileError,
    LoraFileError,
    LoraFormat,
    check_lora,
    lora_trigger,
    read_checkpoint_blocks,
    read_lora,
)
from tests.safetensors_helpers import (
    checkpoint_keys,
    kohya_blocks,
    kohya_module,
    peft_blocks,
    peft_module,
    write_safetensors,
)


def _lora(tmp_path, keys, name="style.safetensors"):
    return read_lora(write_safetensors(tmp_path / name, keys))


def test_a_kohya_lora_is_read_with_its_blocks(tmp_path):
    lora = _lora(tmp_path, kohya_blocks(28))

    assert lora.format is LoraFormat.KOHYA
    assert lora.blocks == tuple(range(28))
    assert lora.highest_block == 27
    assert not lora.has_text_encoder
    assert lora.unused_keys == 0
    assert lora.name == "style.safetensors"


def test_a_peft_lora_is_read_with_its_blocks(tmp_path):
    lora = _lora(tmp_path, peft_blocks(28))

    assert lora.format is LoraFormat.PEFT
    assert lora.blocks == tuple(range(28))


def test_a_lora_on_some_blocks_only_lists_those(tmp_path):
    keys = kohya_module("lora_unet_blocks_3_cross_attn_k_proj")
    keys += kohya_module("lora_unet_blocks_11_mlp_layer2")

    assert _lora(tmp_path, keys).blocks == (3, 11)


def test_the_llm_adapter_blocks_are_not_counted_as_transformer_blocks(tmp_path):
    keys = kohya_blocks(2) + list(kohya_module("lora_unet_llm_adapter_blocks_5_self_attn_q_proj"))

    assert _lora(tmp_path, keys).blocks == (0, 1)


def test_a_text_encoder_part_is_noticed(tmp_path):
    keys = kohya_blocks(28) + list(kohya_module("lora_te_layers_0_self_attn_q_proj"))

    assert _lora(tmp_path, keys).has_text_encoder


def test_keys_of_other_models_are_counted_as_unused(tmp_path):
    keys = kohya_blocks(28) + list(kohya_module("lora_te1_text_model_encoder_layers_0_mlp_fc1"))

    assert _lora(tmp_path, keys).unused_keys == 3


@pytest.mark.parametrize(
    ("marker", "kind"),
    [
        ("lora_unet_blocks_0_mlp_layer1.hada_w1_a", "LoHa"),
        ("lora_unet_blocks_0_mlp_layer1.lokr_w1", "LoKr"),
        ("lora_unet_blocks_0_mlp_layer1.dora_scale", "DoRA"),
        ("diffusion_model.blocks.0.mlp.layer1.lora_magnitude_vector", "DoRA"),
        ("lora_unet_blocks_0_mlp_layer1.lora_mid.weight", "LoCon"),
        ("lora_unet_blocks_0_mlp_layer1.oft_blocks", "OFT"),
        ("diffusion_model.blocks.0.mlp.layer1.diff", "difference"),
    ],
)
def test_a_lora_variant_that_cannot_be_applied_is_refused_by_name(tmp_path, marker, kind):
    with pytest.raises(LoraFileError, match=kind) as error:
        _lora(tmp_path, [*kohya_blocks(1), marker])

    assert "style.safetensors" in str(error.value)


def test_a_file_mixing_the_two_formats_is_refused(tmp_path):
    with pytest.raises(LoraFileError, match="kohya and PEFT"):
        _lora(tmp_path, kohya_blocks(2) + peft_blocks(2))


def test_a_model_file_chosen_as_a_lora_is_refused(tmp_path):
    with pytest.raises(LoraFileError, match="no LoRA weights"):
        _lora(tmp_path, checkpoint_keys(28))


def test_a_lora_for_another_model_is_refused(tmp_path):
    keys = kohya_module("lora_te1_text_model_encoder_layers_0_mlp_fc1")

    with pytest.raises(LoraFileError, match="not an Anima LoRA"):
        _lora(tmp_path, keys)


def test_a_broken_lora_file_is_a_lora_error(tmp_path):
    path = tmp_path / "broken.safetensors"
    path.write_bytes(b"nonsense")

    with pytest.raises(LoraFileError, match="broken.safetensors"):
        read_lora(path)


@pytest.mark.parametrize("prefix", ["net.", "model.diffusion_model.", ""])
def test_the_blocks_of_a_checkpoint_are_counted_with_any_prefix(tmp_path, prefix):
    path = write_safetensors(tmp_path / "model.safetensors", checkpoint_keys(28, prefix))

    assert read_checkpoint_blocks(path) == 28


def test_a_checkpoint_with_a_gap_in_its_blocks_is_refused(tmp_path):
    keys = [key for key in checkpoint_keys(28) if ".blocks.5." not in key]
    path = write_safetensors(tmp_path / "model.safetensors", keys)

    with pytest.raises(CheckpointFileError, match="block 5"):
        read_checkpoint_blocks(path)


def test_a_lora_chosen_as_a_model_is_refused(tmp_path):
    path = write_safetensors(tmp_path / "style.safetensors", kohya_blocks(28))

    with pytest.raises(CheckpointFileError, match="is a LoRA"):
        read_checkpoint_blocks(path)


def test_a_file_without_transformer_blocks_is_not_an_anima_model(tmp_path):
    path = write_safetensors(tmp_path / "vae.safetensors", ["decoder.conv_in.weight"])

    with pytest.raises(CheckpointFileError, match="not an Anima model"):
        read_checkpoint_blocks(path)


def test_a_broken_checkpoint_file_is_a_checkpoint_error(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"")

    with pytest.raises(CheckpointFileError, match="model.safetensors"):
        read_checkpoint_blocks(path)


def test_a_lora_that_fits_the_model_has_nothing_to_say(tmp_path):
    assert check_lora(_lora(tmp_path, kohya_blocks(28)), 28) == ()


def test_a_lora_on_some_blocks_only_has_nothing_to_say(tmp_path):
    lora = _lora(tmp_path, kohya_module("lora_unet_blocks_10_mlp_layer1"))

    assert check_lora(lora, 28) == ()


def test_a_lora_beyond_the_blocks_of_the_model_is_an_error(tmp_path):
    lines = check_lora(_lora(tmp_path, kohya_blocks(40)), 28)

    assert len(lines) == 1
    assert lines[0].startswith("ERROR: ")
    assert "style.safetensors" in lines[0]
    assert "28" in lines[0]


def test_a_lora_made_for_an_older_generation_is_a_warning(tmp_path):
    lines = check_lora(_lora(tmp_path, kohya_blocks(28)), 40)

    assert len(lines) == 1
    assert lines[0].startswith("WARN: ")
    assert "28 blocks" in lines[0]
    assert "40" in lines[0]


def test_a_text_encoder_part_is_reported_as_ignored(tmp_path):
    keys = kohya_blocks(28) + list(kohya_module("lora_te_layers_0_self_attn_q_proj"))

    lines = check_lora(_lora(tmp_path, keys), 28)

    assert lines == (
        "WARN: The text encoder part of style.safetensors is ignored: "
        "only its image model part is applied.",
    )


def test_unused_keys_are_reported(tmp_path):
    keys = kohya_blocks(28) + list(kohya_module("lora_te1_text_model_encoder_layers_0_mlp_fc1"))

    (line,) = check_lora(_lora(tmp_path, keys), 28)

    assert line.startswith("WARN: ")
    assert "3 keys" in line


def test_a_lora_on_the_text_encoder_only_says_so(tmp_path):
    keys = kohya_module("lora_te_layers_0_self_attn_q_proj")

    with pytest.raises(LoraFileError, match="only the text encoder"):
        _lora(tmp_path, keys)


@pytest.mark.parametrize(
    "key",
    [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_1_attn2_to_k",  # SDXL
        "lora_unet_double_blocks_0_img_attn_proj",  # Flux, kohya
    ],
)
def test_a_kohya_lora_of_another_model_is_not_an_anima_lora(tmp_path, key):
    with pytest.raises(LoraFileError, match="not an Anima LoRA"):
        _lora(tmp_path, kohya_module(key))


@pytest.mark.parametrize(
    "module",
    [
        "layers.0.adaLN_modulation.0",  # Z-Image
        "blocks.0.attn.gate",  # Krea2: blocks, but not Anima's modules
        "double_blocks.0.img_attn.proj",  # Flux 2
    ],
)
def test_a_peft_lora_of_another_model_is_not_an_anima_lora(tmp_path, module):
    with pytest.raises(LoraFileError, match="not an Anima LoRA"):
        _lora(tmp_path, peft_module(module))


def test_a_peft_lora_with_the_adaln_modules_is_read_with_its_blocks(tmp_path):
    keys = [
        *peft_module("blocks.0.adaln_modulation_cross_attn.1"),
        *peft_module("blocks.27.cross_attn.output_proj"),
    ]

    lora = _lora(tmp_path, keys)

    assert (lora.format, lora.blocks, lora.unused_keys) == (LoraFormat.PEFT, (0, 27), 0)


def test_modules_that_anima_does_not_have_are_counted_as_unused(tmp_path):
    keys = kohya_blocks(28) + list(kohya_module("lora_unet_final_layer_linear"))

    lora = _lora(tmp_path, keys)

    assert lora.unused_keys == 3


def test_the_trigger_comes_from_the_training_metadata(tmp_path):
    path = write_safetensors(
        tmp_path / "a.safetensors", kohya_blocks(1), {"modelspec.trigger_phrase": " @bisley "}
    )

    assert lora_trigger(path) == "@bisley"


def test_without_a_trigger_in_the_metadata_or_a_readable_file_there_is_none(tmp_path):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"x")

    assert lora_trigger(write_safetensors(tmp_path / "a.safetensors", kohya_blocks(1))) == ""
    assert lora_trigger(broken) == ""
