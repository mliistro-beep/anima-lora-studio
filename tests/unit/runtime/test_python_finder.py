import subprocess

from anima_trainer.runtime.python_finder import (
    CANDIDATES,
    NO_PYTHON_MESSAGE,
    PythonInterpreter,
    find_python,
)


def _system(installed):
    """Fake which/run: `installed` maps a candidate name to (returncode, stdout) or an exception."""
    calls = []
    kwarg_log = []

    def which(name):
        return f"/usr/bin/{name}" if name in installed else None

    def run(cmd, **kwargs):
        calls.append(cmd)
        kwarg_log.append(kwargs)
        result = installed[cmd[0].rsplit("/", 1)[1]]
        if isinstance(result, Exception):
            raise result
        code, stdout = result
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr="")

    run.kwargs = kwarg_log
    return which, run, calls


def test_tested_versions_are_preferred_over_newer_ones():
    assert CANDIDATES.index("python3.10") < CANDIDATES.index("python3.14")
    assert CANDIDATES.index("python3.12") < CANDIDATES.index("python3.13")


def test_returns_first_working_candidate():
    which, run, _ = _system({"python3.12": (0, "3.12\n"), "python3.14": (0, "3.14\n")})

    found = find_python(which=which, run=run)

    assert found == PythonInterpreter("/usr/bin/python3.12", (3, 12))


def test_falls_back_to_a_newer_python_when_only_that_exists():
    which, run, _ = _system({"python3.14": (0, "3.14\n")})

    assert find_python(which=which, run=run) == PythonInterpreter("/usr/bin/python3.14", (3, 14))


def test_skips_python_without_venv_support():
    # Debian/Ubuntu without python3-venv: importing ensurepip fails.
    which, run, _ = _system({"python3.12": (1, ""), "python3": (0, "3.13\n")})

    assert find_python(which=which, run=run) == PythonInterpreter("/usr/bin/python3", (3, 13))


def test_skips_too_old_and_broken_interpreters():
    which, run, _ = _system(
        {
            "python3.11": OSError("broken"),
            "python3.10": subprocess.TimeoutExpired("python3.10", 1),
            "python3": (0, "3.8\n"),
            "python": (0, "garbage"),
        }
    )

    assert find_python(which=which, run=run) is None


def test_none_when_nothing_is_installed():
    which, run, calls = _system({})

    assert find_python(which=which, run=run) is None
    assert calls == []


def test_probe_checks_venv_and_ensurepip():
    which, run, calls = _system({"python3.12": (0, "3.12\n")})

    find_python(which=which, run=run)

    probe = calls[0][-1]
    assert "venv" in probe and "ensurepip" in probe


def test_message_explains_how_to_fix():
    assert "3.10" in NO_PYTHON_MESSAGE
    assert "python3-venv" in NO_PYTHON_MESSAGE


def test_undecodable_output_does_not_stop_the_search():
    which, run, _ = _system({"python3.12": (0, "3.12\n")})

    find_python(which=which, run=run)

    assert run.kwargs[0]["errors"] == "replace"


def test_python_with_forkserver_default_is_only_a_fallback():
    # Python 3.14 breaks sd-scripts' DataLoader workers ("too many fds"), so an older
    # interpreter wins even when the generic `python3` name comes first.
    which, run, _ = _system({"python3": (0, "3.14\n"), "python": (0, "3.13\n")})

    assert find_python(which=which, run=run) == PythonInterpreter("/usr/bin/python", (3, 13))


def test_preferred_python_stops_the_search():
    which, run, calls = _system({"python3.12": (0, "3.12\n"), "python3": (0, "3.12\n")})

    find_python(which=which, run=run)

    assert len(calls) == 1
