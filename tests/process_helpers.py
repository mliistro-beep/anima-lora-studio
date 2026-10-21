"""Process liveness checks shared by the tests that stop real child processes."""

import os
import time
from pathlib import Path

DEATH_TIMEOUT_S = 5
POLL_INTERVAL_S = 0.05


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat_file = Path(f"/proc/{pid}/stat")
    # Without /proc (macOS) a zombie still answers kill(0); treat "signal delivered" as alive.
    if not Path("/proc").is_dir():
        return True
    try:
        return stat_file.read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False  # reaped between the kill(0) probe and the read


def wait_until_dead(pid: int, timeout_s: float = DEATH_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while is_alive(pid) and time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_S)
    return not is_alive(pid)
