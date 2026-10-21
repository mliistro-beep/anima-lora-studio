from pathlib import Path

import pytest

from anima_trainer.models.catalog import BASE_MODEL_URLS, dit_filename
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.choices import BASE_MODELS

ROOT = Path("/data/anima-trainer")


def test_layout_matches_a_citron_installation():
    paths = RuntimePaths(ROOT)

    assert paths.config_file == ROOT / "config.json"
    assert paths.configs_dir == ROOT / "configs"
    assert paths.logs_dir == ROOT / "logs"
    assert paths.qwen3_model == ROOT / "models/anima/text_encoder/qwen_3_06b_base.safetensors"
    assert paths.vae_model == ROOT / "models/anima/vae/qwen_image_vae.safetensors"
    assert paths.train_script == ROOT / "sd-scripts/anima_train_network.py"
    assert paths.inference_script == ROOT / "sd-scripts/anima_minimal_inference.py"
    assert paths.accelerate_config == Path("app_configs/accelerate_gpu.yaml")
    assert paths.accelerate_config_file == ROOT / "app_configs/accelerate_gpu.yaml"


@pytest.mark.parametrize("base_model", BASE_MODELS)
def test_dit_path_per_base_model(base_model):
    expected = ROOT / "models/anima/dit" / f"{base_model}.safetensors"

    assert RuntimePaths(ROOT).dit_model(base_model) == expected


def test_unknown_base_model_falls_back_like_citron():
    assert dit_filename("whatever") == "anima-base-v1.0.safetensors"


def test_every_base_model_has_a_download_url():
    assert set(BASE_MODEL_URLS) == set(BASE_MODELS)
    assert all(url.startswith("https://huggingface.co/") for url in BASE_MODEL_URLS.values())


def test_venv_executables_on_posix():
    paths = RuntimePaths(ROOT, platform="posix")

    assert paths.venv_python == ROOT / ".venv/bin/python"
    assert paths.venv_accelerate == ROOT / ".venv/bin/accelerate"


def test_venv_executables_on_windows():
    paths = RuntimePaths(ROOT, platform="nt")

    assert paths.venv_python == ROOT / ".venv/Scripts/python.exe"
    assert paths.venv_accelerate == ROOT / ".venv/Scripts/accelerate.exe"


def test_missing_required_models_reports_text_encoder_and_vae_only(tmp_path):
    paths = RuntimePaths(tmp_path)

    assert paths.missing_required_models() == ("Qwen3", "VAE")


def test_missing_required_models_empty_when_present(tmp_path):
    paths = RuntimePaths(tmp_path)
    for model in (paths.qwen3_model, paths.vae_model):
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"")

    assert paths.missing_required_models() == ()


def test_required_models_lists_labels_and_paths():
    paths = RuntimePaths(ROOT)

    assert paths.required_models() == (("Qwen3", paths.qwen3_model), ("VAE", paths.vae_model))


def test_resolve_keeps_absolute_paths():
    assert RuntimePaths(ROOT).resolve("/data/images") == Path("/data/images")


def test_resolve_makes_relative_paths_relative_to_the_data_dir():
    assert RuntimePaths(ROOT).resolve("datasets/me") == ROOT / "datasets/me"


def test_resolve_expands_home():
    assert RuntimePaths(ROOT).resolve("~/x") == Path.home() / "x"
