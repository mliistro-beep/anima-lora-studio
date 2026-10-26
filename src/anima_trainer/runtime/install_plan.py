"""Which install steps the data folder still needs, without running anything.

Every step can be repeated safely, so an interrupted install resumes with a new plan:
the venv is kept and sd-scripts is fetched at a pinned commit with init/fetch/checkout.
The checkout, the requirements and the GPU check each leave a marker in <data>/.install
naming the commit they were done for, so a failed check or a new pinned commit redoes
exactly what is needed. torch comes from the sd-scripts requirements, as in Citron's
standard Linux setup; torchvision is added to the same pip run because sd-scripts imports
it but does not list it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from anima_trainer.runtime.accelerate_config import ACCELERATE_GPU_YAML, write_accelerate_config
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime.python_finder import NO_PYTHON_MESSAGE, PythonInterpreter

SD_SCRIPTS_URL = "https://github.com/kohya-ss/sd-scripts.git"
# The commit Citron's install was verified with (real training run on 2026-10-21).
SD_SCRIPTS_COMMIT = "690ea7f96c23182352ec63def76d431c6120bd2f"
# Installed with the sd-scripts requirements in one pip run, so pip picks versions that match torch.
EXTRA_PACKAGES = ("torchvision",)
# The requirements marker names the extra packages too, so adding one reinstalls the requirements.
REQUIREMENTS_STAMP = " ".join((SD_SCRIPTS_COMMIT, *(f"+{name}" for name in EXTRA_PACKAGES)))
INSTALL_STATE_DIR = ".install"
NO_GIT_MESSAGE = "git not found. Install git (on Debian/Ubuntu: sudo apt install git)."
# Fails the step when torch cannot see a GPU, so a broken driver shows up at install time.
GPU_CHECK = (
    "import torch, torchvision; "
    "print('torch', torch.__version__, 'CUDA', torch.version.cuda); "
    "ok = torch.cuda.is_available(); "
    "print('GPU:', torch.cuda.get_device_name(0) if ok else 'none'); "
    "raise SystemExit(0 if ok else 'CUDA is not available: check the NVIDIA driver.')"
)


@dataclass(frozen=True)
class InstallStep:
    key: str
    title: str
    command: tuple[str, ...] = ()  # empty for steps done in-process by `action`
    working_dir: Path | None = None
    # The installer writes `stamp` here after success (creating its folder);
    # the step is planned again until the marker holds that stamp.
    marker: Path | None = None
    stamp: str = SD_SCRIPTS_COMMIT
    action: Callable[[], object] | None = field(default=None, compare=False)

    def run_action(self) -> None:
        if self.action is not None:
            self.action()


@dataclass(frozen=True)
class InstallPlan:
    steps: tuple[InstallStep, ...]
    problems: tuple[str, ...]  # missing tools that prevent running the steps

    @property
    def ready(self) -> bool:
        return not self.problems


def sd_scripts_marker(paths: RuntimePaths) -> Path:
    return paths.root / INSTALL_STATE_DIR / "sd-scripts.commit"


def requirements_marker(paths: RuntimePaths) -> Path:
    return paths.root / INSTALL_STATE_DIR / "requirements.commit"


def verified_marker(paths: RuntimePaths) -> Path:
    return paths.root / INSTALL_STATE_DIR / "verified.commit"


def plan_install(
    paths: RuntimePaths, python: PythonInterpreter | None, git: str | None
) -> InstallPlan:
    venv = _venv_steps(paths, python)
    scripts = _sd_scripts_steps(paths, git)
    rest = _requirements_steps(paths, refetched=bool(scripts)) + _accelerate_steps(paths)
    steps = venv + scripts + rest
    if steps or not _is_current(verified_marker(paths)):
        steps = (*steps, _verify_step(paths))
    problems = (
        *((NO_PYTHON_MESSAGE,) if venv and python is None else ()),
        *((NO_GIT_MESSAGE,) if scripts and git is None else ()),
    )
    return InstallPlan(steps, problems)


def _venv_steps(paths: RuntimePaths, python: PythonInterpreter | None) -> tuple[InstallStep, ...]:
    if paths.venv_python.exists():
        return ()
    root = paths.root
    create_root: tuple[InstallStep, ...] = ()
    if not root.is_dir():
        create_root = (
            InstallStep(
                "data_dir",
                f"Create the data folder {root}",
                action=lambda: root.mkdir(parents=True, exist_ok=True),
            ),
        )
    python_path = python.path if python is not None else "python3"
    venv_python = str(paths.venv_python)
    return (
        *create_root,
        InstallStep(
            "venv",
            "Create the training venv",
            (python_path, "-m", "venv", str(paths.venv_dir)),
            root,
        ),
        InstallStep(
            "pip",
            "Update pip in the training venv",
            (venv_python, "-m", "pip", "install", "--upgrade", "pip"),
            root,
        ),
    )


def _sd_scripts_steps(paths: RuntimePaths, git: str | None) -> tuple[InstallStep, ...]:
    if paths.train_script.exists() and _is_current(sd_scripts_marker(paths)):
        return ()
    git_path = git or "git"
    folder = str(paths.sd_scripts_dir)
    init: tuple[InstallStep, ...] = ()
    if not (paths.sd_scripts_dir / ".git").exists():
        init = (
            InstallStep(
                "sd_scripts_init",
                "Prepare the sd-scripts folder",
                (git_path, "init", folder),
                paths.root,
            ),
        )
    return (
        *init,
        InstallStep(
            "sd_scripts_fetch",
            f"Download sd-scripts (commit {SD_SCRIPTS_COMMIT[:7]})",
            (git_path, "-C", folder, "fetch", "--depth", "1", SD_SCRIPTS_URL, SD_SCRIPTS_COMMIT),
            paths.root,
        ),
        InstallStep(
            "sd_scripts_checkout",
            "Check out sd-scripts",
            # --force: the folder belongs to the app, and an interrupted checkout may have
            # left files that would otherwise block it.
            (git_path, "-C", folder, "checkout", "--detach", "--force", "FETCH_HEAD"),
            paths.root,
            sd_scripts_marker(paths),
        ),
    )


def _requirements_steps(paths: RuntimePaths, refetched: bool) -> tuple[InstallStep, ...]:
    marker = requirements_marker(paths)
    if not refetched and _is_current(marker, REQUIREMENTS_STAMP):
        return ()
    return (
        InstallStep(
            "requirements",
            "Install torch and the sd-scripts requirements (several GB, can take a while)",
            (
                str(paths.venv_python),
                "-m",
                "pip",
                "install",
                "-r",
                "requirements.txt",
                *EXTRA_PACKAGES,
            ),
            paths.sd_scripts_dir,
            marker,
            REQUIREMENTS_STAMP,
        ),
    )


def _accelerate_steps(paths: RuntimePaths) -> tuple[InstallStep, ...]:
    if _read(paths.accelerate_config_file) == ACCELERATE_GPU_YAML:
        return ()
    return (
        InstallStep(
            "accelerate_config",
            "Write the accelerate config",
            action=lambda: write_accelerate_config(paths),
        ),
    )


def _verify_step(paths: RuntimePaths) -> InstallStep:
    return InstallStep(
        "verify",
        "Check that torch sees the GPU",
        (str(paths.venv_python), "-c", GPU_CHECK),
        paths.root,
        verified_marker(paths),
    )


def _is_current(marker: Path, stamp: str = SD_SCRIPTS_COMMIT) -> bool:
    return (_read(marker) or "").strip() == stamp


def _read(path: Path) -> str | None:
    """File text, or None when it is missing or unreadable (the step is then redone)."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
