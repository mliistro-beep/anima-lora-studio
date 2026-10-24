import shutil

import pytest

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime.python_finder import find_python
from anima_trainer.workflow.install_session import plan_for

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    find_python() is None or shutil.which("git") is None, reason="needs Python and git"
)
def test_plan_for_an_empty_folder_uses_the_python_and_git_found_here(tmp_path):
    plan = plan_for(RuntimePaths(tmp_path / "data"))

    assert plan.ready
    keys = [step.key for step in plan.steps]
    assert keys[0] == "data_dir"
    assert keys[-1] == "verify"
