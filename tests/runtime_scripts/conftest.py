"""Tests of the scripts that run in the training venv (torch, sd-scripts).

They need an installed runtime: set ANIMA_TRAINER_TEST_RUNTIME to its data folder, e.g.
ANIMA_TRAINER_TEST_RUNTIME=~/anima-dev-scratch/share/anima-trainer. Skipped otherwise.
"""

import os
import subprocess
from pathlib import Path

import pytest

from anima_trainer.runtime.paths import RuntimePaths

RUNTIME_ENV = "ANIMA_TRAINER_TEST_RUNTIME"
SCRIPT_TIMEOUT_S = 600


@pytest.fixture
def runtime() -> RuntimePaths:
    root = os.environ.get(RUNTIME_ENV, "")
    paths = RuntimePaths(Path(root).expanduser())
    if not root or not paths.venv_python.is_file():
        pytest.skip(f"needs an installed runtime in ${RUNTIME_ENV}")
    return paths


@pytest.fixture
def run_script(runtime):
    def run(
        script: Path, *args: str, env_path: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run `script`; `env_path` is put on PYTHONPATH (for checks importing the scripts)."""
        # -B: no __pycache__ next to the scripts or in sd-scripts.
        command = [str(runtime.venv_python), "-B", str(script), *args]
        env = {**os.environ, "PYTHONPATH": str(env_path)} if env_path else None
        return subprocess.run(  # noqa: S603 - fixed interpreter and script paths
            command, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_S, check=False, env=env
        )

    return run
