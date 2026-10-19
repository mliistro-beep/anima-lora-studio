"""Detects likely out-of-memory failures after a training run and suggests a fix.

Citron only looked at the kernel log; this also checks the training log for CUDA OOM
errors and treats SIGKILL (the kernel OOM killer's signal) as a system RAM OOM.
"""

import subprocess
from collections.abc import Sequence

GPU_OOM_HINT = "GPU out of memory. Try: network_dim=8 and/or resolution=512, or batch size 1."
SYSTEM_OOM_HINT = (
    "The process was killed, probably by the system OOM killer. "
    "Try: network_dim=8 and/or resolution=512, and close other programs."
)

KERNEL_LOG_TAIL_LINES = 40
KERNEL_LOG_TIMEOUT_S = 5
SIGKILL_EXIT_CODES = frozenset({-9, 137})
GPU_OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")
KERNEL_OOM_MARKERS = ("Out of memory", "Killed process", "oom_reaper", "OOM")


def oom_hint(exit_code: int, log_tail: Sequence[str], kernel_log: str | None) -> str | None:
    if exit_code == 0:
        return None
    if any(marker in line for line in log_tail for marker in GPU_OOM_MARKERS):
        return GPU_OOM_HINT
    if kernel_log and _kernel_reports_oom(kernel_log):
        return SYSTEM_OOM_HINT
    if exit_code in SIGKILL_EXIT_CODES:
        return SYSTEM_OOM_HINT
    return None


def _kernel_reports_oom(kernel_log: str) -> bool:
    recent = kernel_log.splitlines()[-KERNEL_LOG_TAIL_LINES:]
    return any(marker in line for line in recent for marker in KERNEL_OOM_MARKERS)


def read_kernel_log() -> str | None:
    """`dmesg -T` output, or None when it is missing or not permitted (common on desktops)."""
    try:
        result = subprocess.run(
            ["dmesg", "-T"], capture_output=True, text=True, timeout=KERNEL_LOG_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None
