"""Built-in presets come with the app; a bad file of the user never hides the others."""

from pathlib import Path

import pytest

from anima_trainer.presets.apply import apply_preset
from anima_trainer.presets.catalog import find_preset, load_presets
from anima_trainer.settings.schema import TrainingSettings

BUILTIN = {
    # The user's note of 2026-11-04, see M8 in DOCS/milestones.md.
    "Style": (32, 32, 0.0001, 16, 0.1, 10),
    "Character": (32, 32, 0.0001, 12, 0.1, 4),
    "Concept": (16, 16, 0.00008, 10, 0.1, 5),
    "Person": (32, 16, 0.0001, 10, 0.05, 4),
}


def _write(folder, file_name, text):
    folder.mkdir(exist_ok=True)
    (folder / file_name).write_text(text, encoding="utf-8")


def test_the_built_in_presets_are_there_in_order(tmp_path):
    loaded = load_presets(tmp_path / "missing")
    assert [preset.name for preset in loaded.presets] == list(BUILTIN)
    assert loaded.warnings == ()
    assert all(preset.builtin for preset in loaded.presets)


@pytest.mark.parametrize(("name", "expected"), BUILTIN.items())
def test_the_built_in_values(tmp_path, name, expected):
    preset = find_preset(load_presets(tmp_path), name)
    settings = apply_preset(preset, TrainingSettings())
    assert (
        settings.network_dim,
        settings.network_alpha,
        settings.learning_rate,
        settings.max_train_epochs,
        settings.caption_dropout,
        settings.save_last_n_epochs,
    ) == expected
    assert preset.description


@pytest.mark.parametrize("name", BUILTIN)
def test_the_built_in_presets_keep_the_default_common_values(tmp_path, name):
    settings = apply_preset(find_preset(load_presets(tmp_path), name), TrainingSettings())
    defaults = TrainingSettings()
    for field in (
        "base_model",
        "optimizer_type",
        "lr_scheduler",
        "lr_scheduler_num_cycles",
        "lr_warmup_steps",
        "max_grad_norm",
        "train_batch_size",
        "save_every_n_epochs",
        "noise_offset",
        "timestep_sampling",
        "discrete_flow_shift",
    ):
        assert getattr(settings, field) == getattr(defaults, field), field
    assert settings.gradient_accumulation_steps == 1


def test_the_user_presets_follow_the_built_in_ones_by_file_name(tmp_path):
    _write(tmp_path, "b.toml", 'name = "Second"')
    _write(tmp_path, "a.toml", 'name = "First"')
    presets = load_presets(tmp_path).presets
    assert [preset.name for preset in presets[-2:]] == ["First", "Second"]
    assert not presets[-1].builtin


def test_other_files_are_ignored(tmp_path):
    _write(tmp_path, "notes.txt", "not a preset")
    loaded = load_presets(tmp_path)
    assert len(loaded.presets) == len(BUILTIN)
    assert loaded.warnings == ()


def test_a_bad_file_is_reported_and_the_others_still_load(tmp_path):
    _write(tmp_path, "bad.toml", 'name = "Bad"\n[settings]\nrepeats = 3\n')
    _write(tmp_path, "good.toml", 'name = "Good"')
    loaded = load_presets(tmp_path)
    assert find_preset(loaded, "Good") is not None
    assert find_preset(loaded, "Bad") is None
    assert len(loaded.warnings) == 1
    assert "bad.toml" in loaded.warnings[0]
    assert "repeats" in loaded.warnings[0]


def test_a_file_that_is_not_text_is_reported(tmp_path):
    tmp_path.joinpath("binary.toml").write_bytes(b"\xff\xfe\x00")
    loaded = load_presets(tmp_path)
    assert "binary.toml" in loaded.warnings[0]


def test_a_folder_named_like_a_preset_is_reported(tmp_path):
    (tmp_path / "folder.toml").mkdir()
    assert "folder.toml" in load_presets(tmp_path).warnings[0]


def test_a_user_preset_cannot_replace_a_built_in_one(tmp_path):
    _write(tmp_path, "mine.toml", 'name = "person"\n[settings]\nnetwork_dim = 8\n')
    loaded = load_presets(tmp_path)
    assert find_preset(loaded, "Person").builtin
    assert "mine.toml" in loaded.warnings[0]
    assert "already" in loaded.warnings[0]


def test_two_user_presets_with_one_name_keep_the_first(tmp_path):
    _write(tmp_path, "a.toml", 'name = "Same"\ndescription = "first"')
    _write(tmp_path, "b.toml", 'name = "same"\ndescription = "second"')
    loaded = load_presets(tmp_path)
    assert find_preset(loaded, "SAME").description == "first"
    assert "b.toml" in loaded.warnings[0]


def test_find_ignores_case_and_gives_none_for_an_unknown_name(tmp_path):
    loaded = load_presets(tmp_path)
    assert find_preset(loaded, "pErSoN").name == "Person"
    assert find_preset(loaded, "nope") is None


def test_a_file_nested_too_deeply_does_not_hide_the_others(tmp_path):
    _write(tmp_path, "deep.toml", "x = " + "[" * 5000)
    _write(tmp_path, "good.toml", 'name = "Good"')
    loaded = load_presets(tmp_path)
    assert find_preset(loaded, "Person") is not None
    assert find_preset(loaded, "Good") is not None
    assert "deep.toml" in loaded.warnings[0]


def test_an_unreadable_folder_is_reported_and_the_built_in_presets_still_load(
    tmp_path, monkeypatch
):
    def denied(_folder, _pattern):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "glob", denied)
    loaded = load_presets(tmp_path)
    assert [preset.name for preset in loaded.presets] == list(BUILTIN)
    assert len(loaded.warnings) == 1
    assert str(tmp_path) in loaded.warnings[0]
