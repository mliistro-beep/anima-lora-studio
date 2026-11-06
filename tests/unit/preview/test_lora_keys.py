"""PEFT LoRA keys renamed to the kohya names sd-scripts applies (runs in the training venv)."""

from anima_trainer.runtime_scripts.lora_keys import kohya_key, kohya_names, lora_notes

PEFT = "diffusion_model.blocks.0.cross_attn.k_proj"
KOHYA = "lora_unet_blocks_0_cross_attn_k_proj"


def test_the_three_peft_parts_get_the_kohya_names():
    assert kohya_key(f"{PEFT}.lora_A.weight") == f"{KOHYA}.lora_down.weight"
    assert kohya_key(f"{PEFT}.lora_B.weight") == f"{KOHYA}.lora_up.weight"
    assert kohya_key(f"{PEFT}.alpha") == f"{KOHYA}.alpha"


def test_the_name_is_the_one_sd_scripts_builds_from_the_model_weight():
    model_weight = "llm_adapter.blocks.2.mlp.layer1.weight"
    built = "lora_unet_" + model_weight.removesuffix(".weight").replace(".", "_")

    assert kohya_key("diffusion_model.llm_adapter.blocks.2.mlp.layer1.lora_A.weight") == (
        f"{built}.lora_down.weight"
    )


def test_keys_that_are_not_peft_lora_keys_have_no_kohya_name():
    for key in (f"{KOHYA}.lora_down.weight", "lora_te_layers_0_mlp.alpha", f"{PEFT}.weight"):
        assert kohya_key(key) is None


def test_only_peft_keys_are_renamed():
    keys = [f"{PEFT}.lora_A.weight", f"{PEFT}.lora_B.weight", "other.weight"]

    assert kohya_names(keys) == {
        f"{PEFT}.lora_A.weight": f"{KOHYA}.lora_down.weight",
        f"{PEFT}.lora_B.weight": f"{KOHYA}.lora_up.weight",
    }


def test_a_kohya_lora_has_nothing_to_say():
    keys = [f"{KOHYA}.lora_down.weight", f"{KOHYA}.lora_up.weight", f"{KOHYA}.alpha"]

    assert lora_notes("a.safetensors", keys) == ()


def test_a_peft_lora_is_said_to_be_renamed():
    keys = [f"{PEFT}.lora_A.weight", f"{PEFT}.lora_B.weight"]

    assert lora_notes("a.safetensors", keys) == (
        ("INFO", "a.safetensors is a PEFT LoRA: its keys are renamed for sd-scripts in memory."),
    )


def test_the_text_encoder_part_and_unused_keys_are_warnings():
    keys = [
        f"{KOHYA}.lora_down.weight",
        f"{KOHYA}.lora_up.weight",
        "lora_te_layers_0_mlp_up_proj.lora_down.weight",
        "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
    ]

    notes = lora_notes("a.safetensors", keys)

    assert notes == (
        ("WARN", "The text encoder part of a.safetensors is ignored: only its image model part "
                 "is applied."),
        ("WARN", "1 key of a.safetensors matches nothing in Anima and is ignored."),
    )  # fmt: skip
