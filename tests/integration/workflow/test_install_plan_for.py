import shutil

import pytest

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime.python_finder import find_python
from anima_trainer.workflow import install_session
from anima_trainer.workflow.install_session import plan_for
from tests.runtime_helpers import mark_installed

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    find_python() is None or shutil.which("git") is None, reason="needs Python and git"
)
def test_plan_for_an_empty_folder_uses_the_python_and_git_found_here(tmp_path):
    plan = plan_for(RuntimePaths(tmp_path / "data"))

    assert plan.ready
    keys = [step.key for step in plan.steps]
    assert keys[0] == "venv"
    assert keys[-1] == "verify"


def test_plan_for_a_ready_venv_does_not_look_for_python(monkeypatch, tmp_path):
    # Probing interpreters runs on the GUI thread, so it is skipped when no venv is made.
    def fail():
        raise AssertionError("find_python must not run")

    monkeypatch.setattr(install_session, "find_python", fail)
    paths = mark_installed(RuntimePaths(tmp_path / "data"))

    assert plan_for(paths).steps == ()
