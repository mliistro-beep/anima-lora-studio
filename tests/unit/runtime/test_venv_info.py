from anima_trainer.runtime.venv_info import (
    FORKSERVER_DEFAULT_VERSION,
    needs_single_process_loader,
    read_venv_version,
)


def _venv(tmp_path, cfg_text):
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text(cfg_text, encoding="utf-8")
    return venv


def test_reads_major_minor_from_pyvenv_cfg(tmp_path):
    venv = _venv(tmp_path, "home = /usr/bin\nversion = 3.14.4\nexecutable = /usr/bin/python3.14\n")

    assert read_venv_version(venv) == (3, 14)


def test_accepts_older_version_info_key(tmp_path):
    venv = _venv(tmp_path, "home = /usr/bin\nversion_info = 3.12.3.final.0\n")

    assert read_venv_version(venv) == (3, 12)


def test_none_without_a_venv(tmp_path):
    assert read_venv_version(tmp_path / ".venv") is None


def test_none_when_version_is_missing_or_garbage(tmp_path):
    assert read_venv_version(_venv(tmp_path, "version = three\n")) is None
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    assert read_venv_version(tmp_path / ".venv") is None


def test_single_process_loader_only_from_forkserver_default_version():
    assert FORKSERVER_DEFAULT_VERSION == (3, 14)
    assert needs_single_process_loader((3, 14))
    assert needs_single_process_loader((3, 15))
    assert not needs_single_process_loader((3, 13))
    assert not needs_single_process_loader(None)


def test_windows_is_never_forced(tmp_path):
    # Windows always used spawn and Citron trains there with workers; only Linux changed in 3.14.
    assert not needs_single_process_loader((3, 14), platform="nt")
