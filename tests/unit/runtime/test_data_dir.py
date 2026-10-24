import pytest
from PySide6.QtCore import QSettings

from anima_trainer.runtime.data_dir import (
    DATA_DIR_KEY,
    data_dir_problem,
    default_data_dir,
    load_data_dir,
    open_data_dir,
    runtime_problems,
    save_data_dir,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)


def _make_runtime(root):
    paths = RuntimePaths(root)
    for file in (paths.train_script, paths.venv_accelerate, paths.accelerate_config_file):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("")
    return paths


def test_default_data_dir_follows_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))

    assert default_data_dir() == tmp_path / "share" / "anima-trainer"


def test_returns_saved_data_dir(store, tmp_path):
    store.setValue(DATA_DIR_KEY, str(tmp_path / "data"))

    assert load_data_dir(store, default=tmp_path / "default") == tmp_path / "data"


def test_falls_back_to_default_when_nothing_saved(store, tmp_path):
    assert load_data_dir(store, default=tmp_path / "default") == tmp_path / "default"


def test_blank_saved_value_counts_as_unset(store, tmp_path):
    store.setValue(DATA_DIR_KEY, "  ")

    assert load_data_dir(store, default=tmp_path / "default") == tmp_path / "default"


def test_old_citron_root_setting_is_ignored(store, tmp_path):
    store.setValue("citron/root", str(tmp_path / "citron"))

    assert load_data_dir(store, default=tmp_path / "default") == tmp_path / "default"


def test_save_round_trips_through_a_new_store(store, tmp_path):
    save_data_dir(store, tmp_path / "data")

    reopened = QSettings(store.fileName(), QSettings.Format.IniFormat)
    assert load_data_dir(reopened, default=tmp_path / "default") == tmp_path / "data"


def test_save_failure_is_raised(tmp_path):
    folder = tmp_path / "readonly"
    folder.mkdir()
    store = QSettings(str(folder / "app.ini"), QSettings.Format.IniFormat)
    folder.chmod(0o500)
    try:
        with pytest.raises(OSError, match="could not be saved"):
            save_data_dir(store, tmp_path / "data")
    finally:
        folder.chmod(0o700)


def test_installed_runtime_has_no_problems(tmp_path):
    assert runtime_problems(_make_runtime(tmp_path / "data")) == ()


def test_missing_data_dir_asks_to_install(tmp_path):
    problems = runtime_problems(RuntimePaths(tmp_path / "nowhere"))

    assert len(problems) == 1
    assert "not installed" in problems[0]


def test_reports_each_missing_runtime_part(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    problems = runtime_problems(RuntimePaths(root))

    assert any("anima_train_network.py" in p for p in problems)
    assert any("accelerate" in p and ".venv" in p for p in problems)
    assert any("accelerate_gpu.yaml" in p for p in problems)


def test_open_data_dir_loads_its_config_json(tmp_path):
    paths = _make_runtime(tmp_path / "data")
    paths.config_file.write_text('{"project_name": "saved", "network_dim": "bad"}')

    opened = open_data_dir(paths.root)

    assert opened.paths == paths
    assert opened.settings.project_name == "saved"
    assert opened.problems == ()
    assert len(opened.warnings) == 1 and "network_dim" in opened.warnings[0]


def test_open_missing_data_dir_uses_defaults_and_reports_problem(tmp_path):
    opened = open_data_dir(tmp_path / "missing")

    assert opened.settings == TrainingSettings()
    assert opened.warnings == ()
    assert "not installed" in opened.problems[0]


def test_a_new_or_existing_folder_can_be_a_data_folder(tmp_path):
    assert data_dir_problem(tmp_path) is None
    assert data_dir_problem(tmp_path / "new" / "nested") is None


def test_a_file_or_a_path_below_a_file_cannot(tmp_path):
    a_file = tmp_path / "a_file"
    a_file.write_text("")

    assert data_dir_problem(a_file) == f"Not a folder: {a_file}"
    assert data_dir_problem(a_file / "sub") == f"Not a folder: {a_file}"
