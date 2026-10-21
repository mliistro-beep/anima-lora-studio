from datetime import datetime
from pathlib import Path

import toml

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.store import load_settings
from anima_trainer.training.toml_builder import (
    build_dataset_config,
    build_training_config,
    config_file_name,
    render_toml,
    write_configs,
)

CITRON_ROOT = Path("/home/user/citron-anima-lora-trainer-ui")
GOLDEN_TIME = datetime(2026, 10, 18, 21, 54, 0)


def _golden_settings(citron_fixtures):
    return load_settings(citron_fixtures / "config.json").settings


def test_training_toml_is_identical_to_citron_output(citron_fixtures):
    config = build_training_config(
        _golden_settings(citron_fixtures), RuntimePaths(CITRON_ROOT), GOLDEN_TIME.date()
    )

    assert render_toml(config) == (citron_fixtures / "golden_training.toml").read_text()


def test_dataset_toml_is_identical_to_citron_output(citron_fixtures):
    config = build_dataset_config(_golden_settings(citron_fixtures))

    assert render_toml(config) == (citron_fixtures / "golden_dataset.toml").read_text()


def test_training_config_uses_selected_base_model(citron_fixtures):
    settings = _golden_settings(citron_fixtures).with_changes(base_model="anima-preview")

    config = build_training_config(settings, RuntimePaths(CITRON_ROOT), GOLDEN_TIME.date())

    assert config["pretrained_model_name_or_path"].endswith("dit/anima-preview.safetensors")


def test_build_returns_independent_dicts(citron_fixtures):
    settings = _golden_settings(citron_fixtures)
    paths = RuntimePaths(CITRON_ROOT)

    first = build_training_config(settings, paths, GOLDEN_TIME.date())
    first["optimizer_args"].append("mutated")
    second = build_training_config(settings, paths, GOLDEN_TIME.date())

    assert second["optimizer_args"] == ["weight_decay=0.1", "betas=[0.9, 0.99]"]


def test_config_file_name_matches_citron_pattern():
    assert (
        config_file_name("proj", "training", GOLDEN_TIME) == "proj_training_2026-10-18_215400.toml"
    )
    assert config_file_name("proj", "dataset", GOLDEN_TIME) == "proj_dataset_2026-10-18_215400.toml"


def test_write_configs_creates_both_files_and_output_dir(tmp_path, citron_fixtures):
    output_dir = tmp_path / "out" / "lora"
    settings = _golden_settings(citron_fixtures).with_changes(output_directory=str(output_dir))
    paths = RuntimePaths(tmp_path / "citron")

    generated = write_configs(settings, paths, GOLDEN_TIME)

    assert output_dir.is_dir()
    assert (
        generated.training
        == paths.configs_dir / "my_style_lora_test1_training_2026-10-18_215400.toml"
    )
    assert generated.dataset.parent == paths.configs_dir
    assert toml.loads(generated.training.read_text())["output_dir"] == str(output_dir)
    assert toml.loads(generated.dataset.read_text())["general"]["resolution"] == 768
