import json

from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.settings.store import load_settings, save_settings


def test_load_missing_file_returns_defaults_without_warnings(tmp_path):
    result = load_settings(tmp_path / "config.json")

    assert result.settings == TrainingSettings()
    assert result.warnings == ()


def test_load_real_citron_config(citron_fixtures):
    result = load_settings(citron_fixtures / "config.json")

    assert result.warnings == ()
    assert result.settings.project_name == "my_style_lora_test1"
    assert result.settings.network_dim == 32
    assert result.settings.network_alpha == 16
    assert result.settings.last_train_config.endswith("_training_2026-10-18_215400.toml")


def test_load_fills_missing_keys_and_ignores_unknown(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"network_dim": 8, "something_else": 1}))

    result = load_settings(path)

    assert result.settings.network_dim == 8
    assert result.settings.repeats == 10
    assert result.warnings == ()


def test_load_corrupt_file_returns_defaults_and_warns(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json")

    result = load_settings(path)

    assert result.settings == TrainingSettings()
    assert len(result.warnings) == 1
    assert "could not be read" in result.warnings[0]


def test_load_non_object_json_returns_defaults_and_warns(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("[1, 2]")

    result = load_settings(path)

    assert result.settings == TrainingSettings()
    assert len(result.warnings) == 1


def test_load_invalid_value_keeps_other_fields_and_warns(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"network_dim": 0, "repeats": 3, "optimizer_type": "Nope"}))

    result = load_settings(path)

    assert result.settings.network_dim == 20
    assert result.settings.optimizer_type == "AdamW8bit"
    assert result.settings.repeats == 3
    assert len(result.warnings) == 2
    assert any("network_dim" in w for w in result.warnings)
    assert any("optimizer_type" in w for w in result.warnings)


def test_save_writes_citron_compatible_json(tmp_path, citron_fixtures):
    source = citron_fixtures / "config.json"
    target = tmp_path / "config.json"
    settings = load_settings(source).settings

    save_settings(target, settings)

    assert json.loads(target.read_text()) == json.loads(source.read_text())
    assert target.read_text() == source.read_text()


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "nested" / "config.json"
    settings = TrainingSettings(network_dim=48, gradient_checkpointing=False)

    save_settings(path, settings)

    assert load_settings(path).settings == settings


def test_save_does_not_leave_temporary_files(tmp_path):
    path = tmp_path / "config.json"

    save_settings(path, TrainingSettings())

    assert [p.name for p in tmp_path.iterdir()] == ["config.json"]
