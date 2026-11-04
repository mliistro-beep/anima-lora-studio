"""RAM, swap and GPU usage as one line for the training log, written while a run is active.

On 2026-10-28 a long training left the computer unreachable and nothing showed what it was
doing at the time; with one such line a minute the log tells what RAM, swap and GPU did up to
the end. Values that cannot be read are left out of the line, never guessed.
"""

import re
from dataclasses import dataclass
from datetime import datetime

NVIDIA_SMI_USAGE_ARGUMENTS = (
    "--query-gpu=index,memory.used,memory.total,temperature.gpu,power.draw,power.limit,"
    "utilization.gpu",
    "--format=csv,noheader,nounits",
)
LINE_PREFIX = "RESOURCES"

_MEMINFO_LINE = re.compile(r"^(\w+):\s+(\d+)\s+kB$", re.MULTILINE)
_KIB_PER_MIB = 1024
_MIB_PER_GB = 1024
_GPU_FIELDS = 7


@dataclass(frozen=True)
class MemoryUsage:
    available_mib: int
    total_mib: int
    swap_used_mib: int | None
    swap_total_mib: int | None


@dataclass(frozen=True)
class GpuUsage:
    index: int
    used_mib: int
    total_mib: int
    temperature_c: int | None
    power_w: float | None
    power_limit_w: float | None
    utilization_pct: int | None


def parse_meminfo(meminfo: str) -> MemoryUsage | None:
    """RAM and swap from the text of /proc/meminfo; None without MemTotal and MemAvailable."""
    values = {name: int(kib) // _KIB_PER_MIB for name, kib in _MEMINFO_LINE.findall(meminfo)}
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    if swap_total is None or swap_free is None:
        return MemoryUsage(values["MemAvailable"], values["MemTotal"], None, None)
    return MemoryUsage(
        values["MemAvailable"], values["MemTotal"], swap_total - swap_free, swap_total
    )


def parse_gpu_usage(stdout: str) -> tuple[GpuUsage, ...]:
    """Rows of NVIDIA_SMI_USAGE_ARGUMENTS output; rows without index and memory are skipped."""
    rows = (tuple(part.strip() for part in line.split(",")) for line in stdout.splitlines())
    return tuple(_gpu_usage(row) for row in rows if _has_index_and_memory(row))


def resource_line(now: datetime, memory: MemoryUsage | None, gpus: tuple[GpuUsage, ...]) -> str:
    parts = [*_memory_parts(memory), *(_gpu_part(gpu) for gpu in gpus)]
    stamp = now.isoformat(timespec="seconds")
    return f"{LINE_PREFIX} {stamp} | {' | '.join(parts) if parts else 'not available'}"


def _has_index_and_memory(row: tuple[str, ...]) -> bool:
    return len(row) == _GPU_FIELDS and all(part.isdecimal() for part in row[:3])


def _gpu_usage(row: tuple[str, ...]) -> GpuUsage:
    return GpuUsage(
        index=int(row[0]),
        used_mib=int(row[1]),
        total_mib=int(row[2]),
        temperature_c=_whole(row[3]),
        power_w=_decimal(row[4]),
        power_limit_w=_decimal(row[5]),
        utilization_pct=_whole(row[6]),
    )


def _whole(text: str) -> int | None:
    # nvidia-smi prints "[N/A]" or "[Not Supported]" for values a GPU does not report.
    return int(text) if text.isdecimal() else None


def _decimal(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _gb(mib: int) -> str:
    return f"{mib / _MIB_PER_GB:.1f}"


def _memory_parts(memory: MemoryUsage | None) -> tuple[str, ...]:
    if memory is None:
        return ()
    ram = f"RAM available {_gb(memory.available_mib)} / {_gb(memory.total_mib)} GB"
    if memory.swap_used_mib is None or memory.swap_total_mib is None:
        return (ram,)
    return (ram, f"swap used {_gb(memory.swap_used_mib)} / {_gb(memory.swap_total_mib)} GB")


def _gpu_part(gpu: GpuUsage) -> str:
    details = [f"VRAM {_gb(gpu.used_mib)} / {_gb(gpu.total_mib)} GB"]
    if gpu.temperature_c is not None:
        details.append(f"{gpu.temperature_c} C")
    if gpu.power_w is not None:
        limit = "" if gpu.power_limit_w is None else f" / {gpu.power_limit_w:.0f}"
        details.append(f"{gpu.power_w:.0f}{limit} W")
    if gpu.utilization_pct is not None:
        details.append(f"load {gpu.utilization_pct}%")
    return f"GPU {gpu.index}: {', '.join(details)}"
