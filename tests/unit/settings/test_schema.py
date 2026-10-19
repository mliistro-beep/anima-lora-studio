import pytest
from pydantic import ValidationError

from anima_trainer.settings.defaults import DEFAULTS
from anima_trainer.settings.schema import TrainingSettings


def test_defaults_match_citron_values():
    settings = TrainingSettings()

    assert settings.model_dump() == DEFAULTS


def test_defaults_cover_all_citron_keys_in_order():
    expected_keys = [
        "project_name",
        "base_model",
        "image_directory",
        "output_directory",
        "network_dim",
        "network_alpha",
        "learning_rate",
        "max_train_epochs",
        "resolution",
        "repeats",
        "caption_dropout",
        "gpu_index",
        "optimizer_type",
        "lr_scheduler",
        "lr_scheduler_num_cycles",
        "lr_warmup_steps",
        "train_batch_size",
        "gradient_accumulation_steps",
        "max_grad_norm",
        "save_every_n_epochs",
        "save_last_n_epochs",
        "mixed_precision",
        "gradient_checkpointing",
        "seed",
        "noise_offset",
        "multires_noise_discount",
        "timestep_sampling",
        "discrete_flow_shift",
        "cache_latents",
        "cache_text_encoder_outputs",
        "vae_chunk_size",
        "vae_disable_cache",
        "num_cpu_threads_per_process",
        "last_train_config",
        "last_dataset_config",
    ]

    assert list(TrainingSettings().model_dump().keys()) == expected_keys


def test_settings_are_immutable():
    settings = TrainingSettings()

    with pytest.raises(ValidationError):
        settings.network_dim = 64


def test_with_changes_returns_new_validated_copy():
    original = TrainingSettings()

    updated = original.with_changes(network_dim=64)

    assert updated.network_dim == 64
    assert original.network_dim == 20


def test_with_changes_rejects_invalid_values():
    with pytest.raises(ValidationError):
        TrainingSettings().with_changes(network_dim=0)


def test_numeric_strings_are_coerced_like_citron_casts():
    settings = TrainingSettings(network_dim="32", learning_rate="1e-4", gpu_index=1)

    assert settings.network_dim == 32
    assert settings.learning_rate == pytest.approx(1e-4)
    assert settings.gpu_index == "1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("network_dim", 0),
        ("network_alpha", 0),
        ("max_train_epochs", 0),
        ("resolution", 63),
        ("repeats", 0),
        ("caption_dropout", 1.5),
        ("caption_dropout", -0.1),
        ("learning_rate", 0),
        ("lr_scheduler_num_cycles", 0),
        ("lr_warmup_steps", -1),
        ("train_batch_size", 0),
        ("gradient_accumulation_steps", 0),
        ("save_every_n_epochs", 0),
        ("save_last_n_epochs", 0),
        ("vae_chunk_size", 0),
        ("num_cpu_threads_per_process", 0),
        ("base_model", "sdxl"),
        ("optimizer_type", "Adafactor"),
        ("lr_scheduler", "step"),
        ("mixed_precision", "fp32"),
        ("timestep_sampling", "shift"),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        TrainingSettings(**{field: value})


def test_unknown_keys_are_ignored():
    settings = TrainingSettings(unknown_key="x")

    assert "unknown_key" not in settings.model_dump()


def test_required_for_configure_reports_empty_fields():
    settings = TrainingSettings(project_name="  ", image_directory="", output_directory="")

    assert settings.missing_required_fields() == (
        "Project name",
        "Image directory",
        "Output directory",
    )


def test_required_for_configure_is_empty_when_filled():
    settings = TrainingSettings(image_directory="/data/img", output_directory="/data/out")

    assert settings.missing_required_fields() == ()


@pytest.mark.parametrize(
    "name", ["../../evil", "a/b", "a\\b", "..", ".", "bad\x00name", "with space"]
)
def test_project_name_rejects_path_like_values(name):
    with pytest.raises(ValidationError):
        TrainingSettings(project_name=name)


@pytest.mark.parametrize("name", ["my_lora", "my_style_lora_test1", "v1.2-final"])
def test_project_name_accepts_safe_values(name):
    assert TrainingSettings(project_name=name).project_name == name


@pytest.mark.parametrize("gpu", [True, None, 1.5, "0; rm", "CPU (no CUDA detected)", ""])
def test_gpu_index_rejects_non_numeric_values(gpu):
    with pytest.raises(ValidationError):
        TrainingSettings(gpu_index=gpu)


@pytest.mark.parametrize(("gpu", "expected"), [(0, "0"), ("1", "1"), ("0,1", "0,1")])
def test_gpu_index_accepts_indices(gpu, expected):
    assert TrainingSettings(gpu_index=gpu).gpu_index == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", float("inf")),
        ("max_grad_norm", float("nan")),
        ("max_grad_norm", -1.0),
        ("noise_offset", -0.1),
        ("multires_noise_discount", 1.5),
        ("discrete_flow_shift", 0.0),
        ("network_dim", True),
        ("seed", False),
    ],
)
def test_numeric_fields_reject_non_finite_negative_and_bool(field, value):
    with pytest.raises(ValidationError):
        TrainingSettings(**{field: value})
