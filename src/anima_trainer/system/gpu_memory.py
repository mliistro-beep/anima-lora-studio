"""Total and free memory of a GPU, from nvidia-smi (the GUI venv has no torch).

nvidia-smi numbers GPUs in PCI bus order; with several GPUs CUDA may order them differently
unless CUDA_DEVICE_ORDER=PCI_BUS_ID. With one GPU, the common case, both agree.
"""

import subprocess
from dataclasses import dataclass

from anima_trainer.system.gpu_info import NVIDIA_SMI_TIMEOUT_S, Runner

NVIDIA_SMI_MEMORY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.free",
    "--format=csv,noheader,nounits",
)


@dataclass(frozen=True)
class GpuMemory:
    index: int
    total_mib: int
    free_mib: int


def parse_memory_rows(stdout: str) -> tuple[GpuMemory, ...]:
    rows = (tuple(part.strip() for part in line.split(",")) for line in stdout.splitlines())
    return tuple(
        GpuMemory(int(row[0]), int(row[1]), int(row[2]))
        for row in rows
        if len(row) == 3 and all(part.isdecimal() for part in row)
    )


def read_gpu_memory(gpu_index: str, run: Runner = subprocess.run) -> GpuMemory | None:
    """Memory of the first GPU in `gpu_index` ("0" or "0,1"); None when it cannot be read."""
    first = gpu_index.split(",", 1)[0].strip()
    if not first.isdecimal():  # isdigit() also accepts '²', which int() rejects
        return None
    try:
        result = run(
            list(NVIDIA_SMI_MEMORY_COMMAND),
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return next((gpu for gpu in parse_memory_rows(result.stdout) if gpu.index == int(first)), None)
