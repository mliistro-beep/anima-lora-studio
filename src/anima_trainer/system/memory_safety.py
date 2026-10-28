"""Whether this computer is set up to survive running out of RAM: enough swap and an
out-of-memory daemon (see DOCS/SYSTEM_SETUP.md).

Without them a training or a generation that uses up the RAM can freeze the whole computer
until a hard reset (it happened on 2026-10-27 with 512 MB of swap and no daemon). The check
only warns, it never blocks; it reads /proc, so it says nothing on systems without it.
"""

import re
import sys
from pathlib import Path

MIN_SWAP_MIB = 8 * 1024
MIB_PER_GB = 1024
OOM_DAEMONS = ("earlyoom", "systemd-oomd")
SETUP_GUIDE = "DOCS/SYSTEM_SETUP.md"
PROC = Path("/proc")

_SWAP_TOTAL = re.compile(r"^SwapTotal:\s+(\d+)\s+kB$", re.MULTILINE)
_KIB_PER_MIB = 1024
_WARN = "WARN: "


def swap_total_mib(meminfo: str) -> int | None:
    """SwapTotal from the text of /proc/meminfo, or None when it is not there."""
    match = _SWAP_TOTAL.search(meminfo)
    return int(match.group(1)) // _KIB_PER_MIB if match else None


def memory_safety_lines(
    meminfo: str | None, processes: frozenset[str] | None, platform: str
) -> tuple[str, ...]:
    """OK / WARN lines for the swap size and the out-of-memory daemon. A value that could
    not be read (None) gives no line, and systems other than Linux get none."""
    if not platform.startswith("linux"):
        return ()
    swap = None if meminfo is None else swap_total_mib(meminfo)
    return (*_swap_line(swap), *_daemon_line(processes))


def warnings_only(lines: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(line for line in lines if line.startswith(_WARN))


def read_memory_safety(proc: Path = PROC, platform: str = sys.platform) -> tuple[str, ...]:
    """memory_safety_lines for this computer, from <proc>/meminfo and <proc>/<pid>/comm."""
    return memory_safety_lines(_read_text(proc / "meminfo"), _process_names(proc), platform)


def _swap_line(swap_mib: int | None) -> tuple[str, ...]:
    if swap_mib is None:
        return ()
    size = f"{swap_mib / MIB_PER_GB:.1f} GB"
    if swap_mib >= MIN_SWAP_MIB:
        return (f"OK: Swap: {size}.",)
    return (
        f"{_WARN}Swap is {size}; 8-16 GB is recommended, otherwise running out of RAM during "
        f"a training or a generation can freeze the computer. See {SETUP_GUIDE}.",
    )


def _daemon_line(processes: frozenset[str] | None) -> tuple[str, ...]:
    if processes is None:
        return ()
    running = [daemon for daemon in OOM_DAEMONS if daemon in processes]
    if running:
        return (f"OK: Out-of-memory daemon running: {', '.join(running)}.",)
    return (
        f"{_WARN}No out-of-memory daemon (earlyoom or systemd-oomd) is running: if the RAM "
        f"runs out the computer can freeze instead of stopping the training. See {SETUP_GUIDE}.",
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii", errors="replace")
    except OSError:
        return None


def _process_names(proc: Path) -> frozenset[str] | None:
    """Names of the running processes; None when the folder cannot be listed."""
    try:
        entries = tuple(proc.iterdir())
    except OSError:
        return None
    names = (_read_text(entry / "comm") for entry in entries if entry.name.isdigit())
    # A process that ended between the listing and the read has no comm: it is just skipped.
    return frozenset(name.strip() for name in names if name is not None)
