"""GPU discovery for the GPU dropdown.

The GUI venv does not ship torch, so devices are listed through the training venv's Python,
then through nvidia-smi, then Citron's own fallback of ["0", "1"].
"""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anima_trainer.runtime.paths import RuntimePaths

NO_CUDA_CHOICE = "CPU (no CUDA detected)"
FALLBACK_CHOICES = ("0", "1")
TORCH_TIMEOUT_S = 60
NVIDIA_SMI_TIMEOUT_S = 10

TORCH_PROBE = (
    "import json, torch; "
    "print(json.dumps([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] "
    "if torch.cuda.is_available() else None))"
)
NVIDIA_SMI_COMMAND = ("nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader")

Runner = Callable[..., subprocess.CompletedProcess[Any]]


def parse_torch_output(stdout: str) -> tuple[str, ...] | None:
    lines = stdout.strip().splitlines()
    try:
        names = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        return None
    if lines and names is None:
        return (NO_CUDA_CHOICE,)
    if not isinstance(names, list) or not names:
        return None
    return tuple(f"{index}: {name}" for index, name in enumerate(names))


def parse_nvidia_smi(stdout: str) -> tuple[str, ...] | None:
    rows = [line.split(",", 1) for line in stdout.strip().splitlines() if "," in line]
    choices = tuple(f"{index.strip()}: {name.strip()}" for index, name in rows)
    return choices or None


def detect_gpus(paths: RuntimePaths, run: Runner = subprocess.run) -> tuple[str, ...]:
    # Run from the data folder like sd-scripts does, but only if it exists: a missing cwd
    # makes subprocess fail before the program even starts.
    torch_cwd = paths.root if paths.root.is_dir() else None
    torch_command = (str(paths.venv_python), "-c", TORCH_PROBE)
    via_torch = _probe(run, torch_command, TORCH_TIMEOUT_S, parse_torch_output, torch_cwd)
    if via_torch:
        return via_torch
    via_smi = _probe(run, NVIDIA_SMI_COMMAND, NVIDIA_SMI_TIMEOUT_S, parse_nvidia_smi, None)
    return via_smi or FALLBACK_CHOICES


def _probe(
    run: Runner,
    command: tuple[str, ...],
    timeout_s: int,
    parse: Callable[[str], tuple[str, ...] | None],
    cwd: Path | None,
) -> tuple[str, ...] | None:
    try:
        result = run(list(command), capture_output=True, text=True, timeout=timeout_s, cwd=cwd)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse(result.stdout) if result.returncode == 0 else None
