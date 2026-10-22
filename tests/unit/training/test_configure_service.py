import json
import os
from datetime import datetime

import pytest
import toml

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.settings.store import load_settings
from anima_trainer.training.configure_service import MAX_LISTED_MISSING, configure_training

NOW = datetime(2026, 10, 19, 10, 30, 0)


@pytest.fixture
def citron(tmp_path):
    paths = RuntimePaths(tmp_path / "citron")
    for model in (paths.qwen3_model, paths.vae_model):
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"")
    return paths


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(3):
        (folder / f"img{i}.png").write_bytes(b"")
        (folder / f"img{i}.txt").write_text("caption")
    return folder


def _settings(dataset, tmp_path, **changes):
    base = TrainingSettings(
        project_name="proj", image_directory=str(dataset), output_directory=str(tmp_path / "out")
    )
    return base.with_changes(**changes)


def _text(result):
    return "\n".join(result.messages)


def test_successful_configuration_writes_configs_and_saves_settings(citron, dataset, tmp_path):
    result = configure_training(_settings(dataset, tmp_path), citron, NOW)

    assert result.success
    assert result.configs.training.exists()
    assert result.configs.dataset.exists()
    assert result.settings.last_train_config == str(result.configs.training)
    assert result.settings.last_dataset_config == str(result.configs.dataset)
    assert load_settings(citron.config_file).settings == result.settings
    assert "Configuration complete" in result.messages[-1]


def test_report_contains_step_estimate(citron, dataset, tmp_path):
    result = configure_training(_settings(dataset, tmp_path), citron, NOW)

    assert result.estimate.steps_per_epoch == 30
    assert result.estimate.total_steps == 300
    assert "Steps per epoch: 30  (3 imgs x 10 repeats)" in _text(result)
    assert "Total steps:     300  (30 x 10 epochs)" in _text(result)


def test_blank_required_fields_fail_without_touching_disk(citron, dataset, tmp_path):
    settings = _settings(dataset, tmp_path, project_name=" ", output_directory="")

    result = configure_training(settings, citron, NOW)

    assert not result.success
    assert result.messages == (
        "ERROR: Project name cannot be empty.",
        "ERROR: Output directory cannot be empty.",
    )
    assert not citron.config_file.exists()


def test_missing_dataset_directory_fails(citron, tmp_path):
    settings = _settings(tmp_path / "missing", tmp_path)

    result = configure_training(settings, citron, NOW)

    assert not result.success
    assert result.messages[-1].startswith("ERROR: Directory not found")


def test_empty_dataset_fails(citron, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    result = configure_training(_settings(empty, tmp_path), citron, NOW)

    assert not result.success
    assert result.messages[-1] == "ERROR: Cannot configure, no images found."
    assert result.configs is None


def test_missing_captions_are_listed_but_do_not_block(citron, dataset, tmp_path):
    for i in range(MAX_LISTED_MISSING + 2):
        (dataset / f"nocap{i:02d}.jpg").write_bytes(b"")

    result = configure_training(_settings(dataset, tmp_path), citron, NOW)

    text = _text(result)
    assert result.success
    assert f"WARN: Missing captions ({MAX_LISTED_MISSING + 2}):" in text
    assert "    - nocap00.jpg" in text
    assert "    ... and 2 more" in text


def test_all_captions_present_is_reported(citron, dataset, tmp_path):
    result = configure_training(_settings(dataset, tmp_path), citron, NOW)

    assert "OK: All images have caption files." in result.messages


def test_missing_dit_is_reported_as_auto_download(citron, dataset, tmp_path):
    result = configure_training(_settings(dataset, tmp_path), citron, NOW)

    assert result.success
    assert "(will auto-download when training starts)" in _text(result)


def test_present_dit_is_reported_ok(citron, dataset, tmp_path):
    dit = citron.dit_model("anima-base-v1.0")
    dit.parent.mkdir(parents=True)
    dit.write_bytes(b"")

    result = configure_training(_settings(dataset, tmp_path), citron, NOW)

    assert f"  OK: DiT: {dit}" in result.messages


def test_missing_text_encoder_or_vae_fails(tmp_path, dataset):
    paths = RuntimePaths(tmp_path / "bare")

    result = configure_training(_settings(dataset, tmp_path), paths, NOW)

    assert not result.success
    assert "ERROR: Missing models: Qwen3, VAE" in result.messages
    assert result.configs is None


def test_config_write_failure_is_reported(citron, dataset, tmp_path):
    blocker = tmp_path / "out"
    blocker.write_text("a file where the output directory should be")

    result = configure_training(_settings(dataset, tmp_path), citron, NOW)

    assert not result.success
    assert result.messages[-1].startswith("ERROR: Failed to generate configs:")


def test_settings_save_failure_keeps_configs_and_warns(citron, dataset, tmp_path):
    citron.config_file.mkdir(parents=True)

    result = configure_training(_settings(dataset, tmp_path), citron, NOW)

    assert result.success
    assert result.configs.training.exists()
    assert any(m.startswith("WARN: Settings could not be saved") for m in result.messages)


def test_input_settings_are_not_modified(citron, dataset, tmp_path):
    settings = _settings(dataset, tmp_path)

    configure_training(settings, citron, NOW)

    assert settings.last_train_config == ""


def test_saved_file_is_citron_json(citron, dataset, tmp_path):
    configure_training(_settings(dataset, tmp_path), citron, NOW)

    saved = json.loads(citron.config_file.read_text())
    assert saved["project_name"] == "proj"
    assert saved["last_train_config"].endswith("proj_training_2026-10-19_103000.toml")


def test_relative_paths_are_resolved_against_the_data_dir(citron):
    images = citron.root / "datasets" / "me"
    images.mkdir(parents=True)
    (images / "a.png").write_bytes(b"")
    (images / "a.txt").write_text("c")
    settings = TrainingSettings(
        project_name="proj", image_directory="datasets/me", output_directory="out/proj"
    )

    result = configure_training(settings, citron, NOW)

    assert result.success
    assert result.settings.image_directory == str(images)
    assert result.settings.output_directory == str(citron.root / "out" / "proj")
    assert (citron.root / "out" / "proj").is_dir()


def test_absolute_paths_are_kept_verbatim(citron, dataset, tmp_path):
    output = str(tmp_path / "out") + "/"

    result = configure_training(_settings(dataset, tmp_path, output_directory=output), citron, NOW)

    assert result.settings.output_directory == output


def _venv_with_version(paths, version):
    paths.venv_dir.mkdir(parents=True, exist_ok=True)
    (paths.venv_dir / "pyvenv.cfg").write_text(f"version = {version}\n", encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="forkserver default only applies to POSIX")
def test_python_314_venv_disables_dataloader_workers(citron, dataset, tmp_path):
    _venv_with_version(citron, "3.14.4")

    result = configure_training(_settings(dataset, tmp_path), citron, NOW)

    written = toml.loads(result.configs.training.read_text())
    assert written["max_data_loader_n_workers"] == 0
    assert "DataLoader workers disabled" in _text(result)


def test_older_venv_keeps_citron_toml(citron, dataset, tmp_path):
    _venv_with_version(citron, "3.12.3")

    result = configure_training(_settings(dataset, tmp_path), citron, NOW)

    written = toml.loads(result.configs.training.read_text())
    assert "max_data_loader_n_workers" not in written
    assert "DataLoader workers disabled" not in _text(result)
