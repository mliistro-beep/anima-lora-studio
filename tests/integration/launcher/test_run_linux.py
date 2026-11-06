"""run_linux.sh with a fake python3, in a copy of the checkout and a temporary XDG_DATA_HOME.

PATH holds only the fakes and the few system tools the launcher needs, so no real Python, pip,
desktop-file-validate or notify-send is ever found by the launcher.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = "run_linux.sh"
ICON = Path("src/anima_trainer/ui/assets/anima-lora-studio.svg")
DESKTOP_FILE = "anima-lora-studio.desktop"
TERMINAL_DESKTOP_FILE = "anima-lora-studio-terminal.desktop"
COMMAND = Path(".local/bin/anima-trainer")
TOOLS = (
    "bash",
    "env",
    "basename",
    "dirname",
    "readlink",
    "touch",
    "mkdir",
    "sha256sum",
    "cat",
    "rm",
    "cp",
    "sed",
    "tee",
    "flock",
    "ln",
)
BASH = shutil.which("bash")

FAKE_PYTHON = r"""#!/usr/bin/env bash
# Fake system Python: the version check, --version and -m venv. FAKE_TOO_OLD lists the names
# (python3, python3.12, ...) that are older than 3.12; FAKE_VENV_FAIL leaves a half-made venv.
name="$(basename "$0")"
echo "$name $*" >> "$FAKE_LOG"
case "$1" in
    --version) echo "Python ${FAKE_VERSION:-3.12.3}" ;;
    -c) [[ " ${FAKE_TOO_OLD:-} " != *" $name "* ]] ;;
    -m)
        [[ "$2" == venv ]] || exit 2
        mkdir -p "$3/bin"
        cp "$FAKE_BIN/venv_python" "$3/bin/python"
        [[ -z "${FAKE_VENV_FAIL:-}" ]] || { touch "$3/no-pip"; exit 1; }
        ;;
    *) exit 2 ;;
esac
"""

FAKE_VENV_PYTHON = r"""#!/usr/bin/env bash
# Fake venv python: -m pip --version (health check) and -m pip install -e, which creates the
# app script. A venv made with FAKE_VENV_FAIL has no pip.
echo "venv-python $*" >> "$FAKE_LOG"
venv="$(dirname "$(dirname "$0")")"
[[ ! -e "$venv/no-pip" ]] || exit 1
case "$1" in
    -m)
        [[ "$2 $3" == "pip --version" ]] && exit 0
        [[ "$2 $3 $4" == "pip install -e" ]] || exit 2
        [[ -z "${FAKE_PIP_FAIL:-}" ]] || exit 1
        cp "$FAKE_BIN/app" "$venv/bin/anima-trainer"
        ;;
    *) exit 2 ;;
esac
"""

FAKE_APP = r"""#!/usr/bin/env bash
# The launcher must release its install lock (fd 9) before it starts the app.
[[ ! -e /dev/fd/9 ]] || echo "app still holds fd 9" >> "$FAKE_LOG"
echo "app $* (cwd $PWD)" >> "$FAKE_LOG"
"""

FAKE_NOTIFY = r"""#!/usr/bin/env bash
echo "notify-send $*" >> "$FAKE_LOG"
"""

FAKE_VALIDATOR = r"""#!/usr/bin/env bash
echo "error: fake complaint about $1"
exit 1
"""


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "anima lora studio"  # a space, as in many home folders
    root.mkdir()
    shutil.copy2(REPO / LAUNCHER, root / LAUNCHER)
    shutil.copy2(REPO / "pyproject.toml", root / "pyproject.toml")
    (root / ICON).parent.mkdir(parents=True)
    shutil.copy2(REPO / ICON, root / ICON)
    return root


@pytest.fixture
def fake_bin(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in TOOLS:
        path = shutil.which(tool)
        if path is None:
            pytest.fail(f"{tool} is needed by run_linux.sh and its tests but is not installed")
        (bin_dir / tool).symlink_to(path)
    for name, text in (
        ("python3", FAKE_PYTHON),
        ("venv_python", FAKE_VENV_PYTHON),
        ("app", FAKE_APP),
    ):
        _write_script(bin_dir / name, text)
    return bin_dir


def _write_script(path, text):
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def run(tmp_path, checkout, fake_bin):
    log = tmp_path / "calls.log"
    home = tmp_path / "home"
    home.mkdir()

    def run(*args, root=checkout, **env):
        environment = {
            "PATH": str(fake_bin),
            "HOME": str(home),
            "XDG_DATA_HOME": str(tmp_path / "share"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "FAKE_LOG": str(log),
            "FAKE_BIN": str(fake_bin),
            **env,
        }
        environment = {key: value for key, value in environment.items() if value is not None}
        result = subprocess.run(
            [BASH, str(root / LAUNCHER), *args],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        calls = log.read_text().splitlines() if log.exists() else []
        log.unlink(missing_ok=True)
        return result, calls

    return run


def _pip_installs(calls):
    return [call for call in calls if " -m pip install -e " in call]


def test_first_run_creates_the_venv_installs_the_app_and_starts_it(run, checkout):
    result, calls = run("--some-qt-arg", "value")

    assert result.returncode == 0, result.stderr
    assert f"python3 -m venv {checkout}/.venv" in calls
    assert _pip_installs(calls) == [f"venv-python -m pip install -e {checkout}"]
    assert calls[-1] == f"app --some-qt-arg value (cwd {checkout})"


def test_second_run_does_not_reinstall(run):
    run()

    result, calls = run()

    assert result.returncode == 0, result.stderr
    assert _pip_installs(calls) == []
    assert not any(call.startswith("python3 -m venv") for call in calls)
    assert calls[-1].startswith("app ")


def test_a_changed_pyproject_reinstalls_once(run, checkout):
    run()
    pyproject = checkout / "pyproject.toml"
    pyproject.write_text(pyproject.read_text() + "\n# new dependency\n")

    _, calls = run()
    _, calls_after = run()

    assert len(_pip_installs(calls)) == 1
    assert _pip_installs(calls_after) == []


def test_an_existing_venv_without_the_stamp_is_reused_and_installed_once(run, checkout):
    # The development .venv (scripts/dev-run.sh --setup) has no stamp yet.
    run()
    (checkout / ".venv" / ".anima-pyproject.sha256").unlink()

    _, calls = run()

    assert not any(call.startswith("python3 -m venv") for call in calls)
    assert len(_pip_installs(calls)) == 1


def test_a_failed_install_does_not_start_the_app_and_is_retried(run):
    result, calls = run(FAKE_PIP_FAIL="1")

    assert result.returncode != 0
    assert "ERROR:" in result.stderr
    assert not any(call.startswith("app ") for call in calls)

    result, calls = run()

    assert result.returncode == 0, result.stderr
    assert len(_pip_installs(calls)) == 1


def test_a_too_old_python_stops_with_a_clear_error(run, checkout):
    result, calls = run(FAKE_TOO_OLD="python3", FAKE_VERSION="3.10.12")

    assert result.returncode == 1
    assert "ERROR: Python 3.12 or newer is needed" in result.stderr
    assert "3.10.12" in result.stderr
    assert not (checkout / ".venv").exists()


def test_no_python_stops_with_a_clear_error(run, fake_bin, checkout):
    (fake_bin / "python3").unlink()

    result, _ = run()

    assert result.returncode == 1
    assert "ERROR: Python 3.12 or newer is needed" in result.stderr
    assert not (checkout / ".venv").exists()


def test_a_broken_venv_python_stops_with_a_clear_error(run, checkout):
    run()
    (checkout / ".venv" / "bin" / "python").write_text("#!/usr/bin/env bash\nexit 1\n")

    result, calls = run()

    assert result.returncode == 1
    assert "ERROR:" in result.stderr
    assert ".venv" in result.stderr
    assert not any(call.startswith("app ") for call in calls)


def test_help_shows_the_options_without_installing(run):
    result, calls = run("--help")

    assert result.returncode == 0
    assert "--install-desktop" in result.stdout
    assert "--uninstall-desktop" in result.stdout
    assert calls == []


def _desktop_entry(tmp_path):
    return tmp_path / "share" / "applications" / DESKTOP_FILE


def _installed_icon(tmp_path):
    return tmp_path / "share/icons/hicolor/scalable/apps/anima-lora-studio.svg"


def test_install_desktop_sets_up_the_venv_and_writes_the_entry_and_the_icon(
    run, tmp_path, checkout
):
    result, calls = run("--install-desktop")

    assert result.returncode == 0, result.stderr
    assert len(_pip_installs(calls)) == 1
    assert not any(call.startswith("app ") for call in calls)
    entry = _desktop_entry(tmp_path).read_text()
    assert "Name=Anima LoRA Studio\n" in entry
    assert f'Exec="{checkout}/run_linux.sh"\n' in entry
    assert "Icon=anima-lora-studio\n" in entry
    assert "Terminal=false\n" in entry
    assert _installed_icon(tmp_path).read_bytes() == (checkout / ICON).read_bytes()


def test_install_desktop_without_xdg_data_home_uses_the_home_folder(run, tmp_path):
    result, _ = run("--install-desktop", XDG_DATA_HOME=None)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "home/.local/share/applications" / DESKTOP_FILE).is_file()


@pytest.mark.skipif(not shutil.which("desktop-file-validate"), reason="desktop-file-validate")
@pytest.mark.parametrize("entry_name", [DESKTOP_FILE, TERMINAL_DESKTOP_FILE])
def test_the_desktop_entries_are_valid(run, tmp_path, entry_name):
    run("--install-desktop")

    check = subprocess.run(
        [shutil.which("desktop-file-validate"), str(tmp_path / "share/applications" / entry_name)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert check.returncode == 0, check.stdout + check.stderr
    assert check.stdout == ""


def _terminal_entry(tmp_path):
    return tmp_path / "share" / "applications" / TERMINAL_DESKTOP_FILE


def _command(tmp_path):
    return tmp_path / "home" / COMMAND


def test_install_desktop_adds_a_terminal_entry_that_opens_the_tui(run, tmp_path, checkout):
    result, _ = run("--install-desktop")

    assert result.returncode == 0, result.stderr
    entry = _terminal_entry(tmp_path).read_text()
    assert "Name=Anima LoRA Studio (terminal)\n" in entry
    assert f'Exec="{checkout}/run_linux.sh" tui\n' in entry
    assert "Icon=anima-lora-studio\n" in entry
    assert "Terminal=true\n" in entry


def test_install_desktop_links_the_command_into_the_users_bin_folder(run, tmp_path, checkout):
    result, _ = run("--install-desktop")

    assert result.returncode == 0, result.stderr
    command = _command(tmp_path)
    assert command.is_symlink()
    assert command.readlink() == checkout / ".venv/bin/anima-trainer"
    assert f"OK: Command installed: {command}" in result.stdout


def test_a_bin_folder_missing_from_path_is_reported(run, tmp_path, fake_bin):
    outside, _ = run("--install-desktop")
    inside, _ = run("--install-desktop", PATH=f"{fake_bin}:{tmp_path / 'home/.local/bin'}")

    assert "is not in PATH" in outside.stdout
    assert "is not in PATH" not in inside.stdout


def test_a_command_file_that_is_not_ours_is_kept(run, tmp_path):
    command = _command(tmp_path)
    command.parent.mkdir(parents=True)
    command.write_text("#!/bin/sh\necho mine\n")

    result, _ = run("--install-desktop")

    assert result.returncode == 0, result.stderr
    assert "WARN:" in result.stderr
    assert str(command) in result.stderr
    assert _terminal_entry(tmp_path).is_file()

    run("--uninstall-desktop")

    assert command.read_text() == "#!/bin/sh\necho mine\n"


def test_a_command_link_to_another_checkout_is_replaced(run, tmp_path, checkout):
    command = _command(tmp_path)
    command.parent.mkdir(parents=True)
    command.symlink_to(tmp_path / "old checkout/.venv/bin/anima-trainer")

    result, _ = run("--install-desktop")

    assert result.returncode == 0, result.stderr
    assert command.readlink() == checkout / ".venv/bin/anima-trainer"


def test_uninstall_desktop_removes_the_entries_the_icon_and_the_command_and_can_run_twice(
    run, tmp_path
):
    run("--install-desktop")

    first, calls = run("--uninstall-desktop")
    second, _ = run("--uninstall-desktop")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert not _desktop_entry(tmp_path).exists()
    assert not _terminal_entry(tmp_path).exists()
    assert not _installed_icon(tmp_path).exists()
    assert not _command(tmp_path).is_symlink()
    assert _pip_installs(calls) == []


def test_uninstall_desktop_keeps_a_command_link_to_another_checkout(run, tmp_path):
    command = _command(tmp_path)
    command.parent.mkdir(parents=True)
    other = tmp_path / "other checkout/.venv/bin/anima-trainer"
    command.symlink_to(other)

    result, _ = run("--uninstall-desktop")

    assert result.returncode == 0, result.stderr
    assert command.readlink() == other


def test_a_checkout_path_that_cannot_be_written_in_exec_is_refused(run, tmp_path, checkout):
    odd = tmp_path / 'odd "quote"'
    shutil.copytree(checkout, odd)

    result, _ = run("--install-desktop", root=odd)

    assert result.returncode == 1
    assert "ERROR:" in result.stderr
    assert not _desktop_entry(tmp_path).exists()


def _stamp(checkout):
    return checkout / ".venv" / ".anima-pyproject.sha256"


def test_the_app_starts_without_the_install_lock(run):
    _, calls = run()

    assert "app still holds fd 9" not in calls


def test_a_too_old_python3_falls_back_to_a_versioned_python(run, fake_bin, checkout):
    (fake_bin / "python3.12").symlink_to(fake_bin / "python3")

    result, calls = run(FAKE_TOO_OLD="python3")

    assert result.returncode == 0, result.stderr
    assert f"python3.12 -m venv {checkout}/.venv" in calls
    assert calls[-1].startswith("app ")


def test_a_failed_reinstall_removes_the_stamp_and_still_starts_the_installed_app(run, checkout):
    run()
    pyproject = checkout / "pyproject.toml"
    pyproject.write_text(pyproject.read_text() + "\n# new dependency\n")

    result, calls = run(FAKE_PIP_FAIL="1")

    assert result.returncode == 0, result.stderr
    assert "WARN:" in result.stderr
    assert not _stamp(checkout).exists()
    assert calls[-1].startswith("app ")

    _, calls = run()

    assert len(_pip_installs(calls)) == 1


def test_a_failed_venv_is_removed_so_the_next_start_creates_it_again(run, checkout):
    result, calls = run(FAKE_VENV_FAIL="1")

    assert result.returncode == 1
    assert "python3-venv" in result.stderr
    assert not (checkout / ".venv").exists()
    assert not any(call.startswith("app ") for call in calls)

    result, calls = run()

    assert result.returncode == 0, result.stderr
    assert calls[-1].startswith("app ")


def test_a_venv_without_pip_stops_with_a_clear_error(run, checkout):
    run()
    (checkout / ".venv" / "no-pip").touch()

    result, calls = run()

    assert result.returncode == 1
    assert f"Remove {checkout}/.venv" in result.stderr
    assert not any(call.startswith("app ") for call in calls)


def test_a_moved_checkout_reinstalls(run, tmp_path, checkout):
    run()
    moved = tmp_path / "moved"
    shutil.copytree(checkout, moved, symlinks=True)

    _, calls = run(root=moved)

    assert _pip_installs(calls) == [f"venv-python -m pip install -e {moved}"]


def test_a_symlinked_launcher_uses_the_real_checkout(run, tmp_path, checkout):
    link_dir = tmp_path / "home-bin"
    link_dir.mkdir()
    (link_dir / LAUNCHER).symlink_to(checkout / LAUNCHER)

    result, calls = run(root=link_dir)

    assert result.returncode == 0, result.stderr
    assert calls[-1] == f"app  (cwd {checkout})"


def test_without_a_terminal_the_output_goes_to_a_log_and_errors_to_a_notification(
    run, tmp_path, fake_bin
):
    _write_script(fake_bin / "notify-send", FAKE_NOTIFY)

    result, calls = run(FAKE_TOO_OLD="python3")

    assert result.returncode == 1
    log = (tmp_path / "state/anima-trainer/launcher.log").read_text()
    assert "ERROR: Python 3.12 or newer is needed" in log
    assert "ERROR: Python 3.12 or newer is needed" in result.stderr
    notifications = [call for call in calls if call.startswith("notify-send ")]
    assert len(notifications) == 1
    assert "Python 3.12 or newer is needed" in notifications[0]


def test_without_a_terminal_the_first_install_is_announced(run, fake_bin):
    _write_script(fake_bin / "notify-send", FAKE_NOTIFY)

    _, calls = run()

    notifications = [call for call in calls if call.startswith("notify-send ")]
    assert len(notifications) == 1
    assert "Installing" in notifications[0]


def test_install_desktop_twice_rewrites_the_same_entry(run, tmp_path):
    run("--install-desktop")
    first = _desktop_entry(tmp_path).read_text()

    result, _ = run("--install-desktop")

    assert result.returncode == 0, result.stderr
    assert _desktop_entry(tmp_path).read_text() == first


def test_a_validator_complaint_is_a_warning(run, tmp_path, fake_bin):
    _write_script(fake_bin / "desktop-file-validate", FAKE_VALIDATOR)

    result, _ = run("--install-desktop")

    assert result.returncode == 0, result.stderr
    assert "WARN:" in result.stderr
    assert "fake complaint" in result.stdout + result.stderr
    assert _desktop_entry(tmp_path).is_file()
