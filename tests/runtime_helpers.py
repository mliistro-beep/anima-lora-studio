"""A data folder in the state a finished install leaves it, for tests that need one."""

from anima_trainer.runtime.accelerate_config import write_accelerate_config
from anima_trainer.runtime.install_plan import (
    REQUIREMENTS_STAMP,
    SD_SCRIPTS_COMMIT,
    VENV_STAMP,
    requirements_marker,
    sd_scripts_marker,
    venv_marker,
    verified_marker,
)
from anima_trainer.runtime.paths import RuntimePaths


def mark_installed(paths: RuntimePaths) -> RuntimePaths:
    """Create the missing runtime files, the install markers and the accelerate config.

    Files that already exist (e.g. a fake accelerate script) are kept.
    """
    for file in (paths.venv_python, paths.venv_accelerate, paths.train_script):
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists():
            file.write_text("")
    markers = (
        (venv_marker(paths), VENV_STAMP),
        (sd_scripts_marker(paths), SD_SCRIPTS_COMMIT),
        (requirements_marker(paths), REQUIREMENTS_STAMP),
        (verified_marker(paths), SD_SCRIPTS_COMMIT),
    )
    for marker, stamp in markers:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(stamp + "\n")
    write_accelerate_config(paths)
    return paths
