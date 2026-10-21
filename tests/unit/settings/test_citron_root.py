import pytest
from PySide6.QtCore import QSettings

from anima_trainer.models.paths import CitronPaths
from anima_trainer.settings.citron_root import (
    ROOT_KEY,
    citron_root_problems,
    load_citron_root,
    open_citron_root,
    save_citron_root,
)
from anima_trainer.settings.schema import TrainingSettings


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)


def _make_citron(root):
    paths = CitronPaths(root)
    for file in (paths.train_script, paths.venv_accelerate):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("")
    return paths


def test_returns_saved_root(store, tmp_path):
    store.setValue(ROOT_KEY, str(tmp_path / "citron"))

    assert load_citron_root(store, candidates=()) == tmp_path / "citron"


def test_falls_back_to_first_existing_candidate_when_nothing_saved(store, tmp_path):
    existing = tmp_path / "found"
    existing.mkdir()

    root = load_citron_root(store, candidates=(tmp_path / "missing", existing))

    assert root == existing


def test_returns_none_without_saved_root_or_existing_candidate(store, tmp_path):
    assert load_citron_root(store, candidates=(tmp_path / "missing",)) is None


def test_blank_saved_value_counts_as_unset(store):
    store.setValue(ROOT_KEY, "  ")

    assert load_citron_root(store, candidates=()) is None


def test_save_round_trips_through_a_new_store(store, tmp_path):
    save_citron_root(store, tmp_path / "citron")

    reopened = QSettings(store.fileName(), QSettings.Format.IniFormat)
    assert load_citron_root(reopened, candidates=()) == tmp_path / "citron"


def test_complete_citron_install_has_no_problems(tmp_path):
    assert citron_root_problems(_make_citron(tmp_path / "citron")) == ()


def test_missing_root_is_the_only_problem_reported(tmp_path):
    problems = citron_root_problems(CitronPaths(tmp_path / "nowhere"))

    assert len(problems) == 1
    assert "does not exist" in problems[0]


def test_reports_missing_training_script_and_accelerate(tmp_path):
    root = tmp_path / "citron"
    root.mkdir()

    problems = citron_root_problems(CitronPaths(root))

    assert any("anima_train_network.py" in p for p in problems)
    assert any("accelerate" in p for p in problems)


def test_open_root_loads_its_config_json(tmp_path):
    paths = _make_citron(tmp_path / "citron")
    paths.config_file.write_text('{"project_name": "saved", "network_dim": "bad"}')

    opened = open_citron_root(paths.root)

    assert opened.paths == paths
    assert opened.settings.project_name == "saved"
    assert opened.problems == ()
    assert len(opened.warnings) == 1 and "network_dim" in opened.warnings[0]


def test_open_missing_root_uses_defaults_and_reports_problem(tmp_path):
    opened = open_citron_root(tmp_path / "missing")

    assert opened.settings == TrainingSettings()
    assert opened.warnings == ()
    assert "does not exist" in opened.problems[0]


def test_save_failure_is_raised(tmp_path):
    folder = tmp_path / "readonly"
    folder.mkdir()
    store = QSettings(str(folder / "app.ini"), QSettings.Format.IniFormat)
    folder.chmod(0o500)
    try:
        with pytest.raises(OSError, match="could not be saved"):
            save_citron_root(store, tmp_path / "citron")
    finally:
        folder.chmod(0o700)
