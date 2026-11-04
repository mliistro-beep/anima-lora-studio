"""Which memory profiles generate.py may use, given the free GPU memory and system RAM.

Pure Python (no torch), so the app's tests cover it too. Both values are read by the
generator itself right before loading the models: they are the current ones (GPU memory for
the GPU CUDA_VISIBLE_DEVICES selects, RAM with a running training already counted), and
reading them does not block the app's window.

The two memories pull in opposite directions: the profiles that are lighter on the GPU keep
DiT blocks (and the text encoder) in system RAM. A profile that does not fit the RAM is never
tried, because running out of RAM does not raise an error like the GPU does: the system
swaps, and with little swap the whole computer freezes.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

MIB = 1 << 20
KIB_PER_MIB = 1024
FREE_RAM_ADVICE = "Close other programs, or wait for the training to end."

_MEM_AVAILABLE = re.compile(r"^MemAvailable:\s+(\d+)\s+kB$", re.MULTILINE)


@dataclass(frozen=True)
class ProfileNeed:
    vram_mib: int
    ram_mib: int


@dataclass(frozen=True)
class MemoryBudget:
    free_vram_mib: int | None  # None: no CUDA GPU
    vram_margin_mib: int
    available_ram_mib: int | None  # None: unknown (no /proc/meminfo)
    ram_margin_mib: int

    def fits_vram(self, need: ProfileNeed) -> bool:
        if self.free_vram_mib is None:
            return True
        return need.vram_mib + self.vram_margin_mib <= self.free_vram_mib

    def fits_ram(self, need: ProfileNeed) -> bool:
        if self.available_ram_mib is None:
            return True
        return need.ram_mib + self.ram_margin_mib <= self.available_ram_mib


def _first_vram_fit(needs: Sequence[ProfileNeed], budget: MemoryBudget) -> int | None:
    return next((index for index, need in enumerate(needs) if budget.fits_vram(need)), None)


def usable_profiles(needs: Sequence[ProfileNeed], budget: MemoryBudget) -> tuple[int, ...]:
    """Indexes of the profiles to try in order: from the first that fits the GPU memory (the
    first one when it is unknown), without those that do not fit the RAM; empty when none."""
    start = _first_vram_fit(needs, budget)
    if start is None:
        return ()
    return tuple(index for index in range(start, len(needs)) if budget.fits_ram(needs[index]))


def lighter_skipped_for_ram(needs: Sequence[ProfileNeed], budget: MemoryBudget) -> tuple[int, ...]:
    """Indexes of the profiles lighter on the GPU than the last usable one: all of them were
    left out because they do not fit the RAM. Empty when nothing is usable."""
    usable = usable_profiles(needs, budget)
    return tuple(range(usable[-1] + 1, len(needs))) if usable else ()


def not_enough_memory(needs: Sequence[ProfileNeed], budget: MemoryBudget) -> str:
    """Why usable_profiles found nothing: no profile, the GPU memory, or else the RAM."""
    if not needs:
        return "No memory profile given."
    start = _first_vram_fit(needs, budget)
    if start is None:
        needed = min((need.vram_mib for need in needs), default=0) + budget.vram_margin_mib
        return (
            f"Not enough free GPU memory: about {needed} MiB needed, "
            f"{budget.free_vram_mib} MiB free."
        )
    needed = min(need.ram_mib for need in needs[start:]) + budget.ram_margin_mib
    return (
        f"Not enough free system RAM: about {needed} MiB needed, "
        f"{budget.available_ram_mib} MiB available. {FREE_RAM_ADVICE}"
    )


def next_profile_warning(failed: str, following: str) -> str:
    return f"Out of GPU memory with {failed}, trying {following}."


def all_out_of_gpu_memory(
    last: str, skipped_for_ram: Sequence[str], available_ram_mib: int | None
) -> str:
    """Every usable profile ran out of GPU memory. When lighter ones were left out for the
    RAM, the RAM is what the user has to free, so the message says that."""
    if not skipped_for_ram:
        return "Out of GPU memory even with the lightest profile."
    return (
        f"Out of GPU memory with {last}; the lighter profiles ({', '.join(skipped_for_ram)}) "
        f"need more system RAM than the {available_ram_mib} MiB available. {FREE_RAM_ADVICE}"
    )


def ram_unknown_warning(platform: str) -> str | None:
    """A WARN text when /proc/meminfo should exist (Linux) but could not be read: the RAM
    check is then off. Elsewhere unknown RAM is expected and needs no warning."""
    if not platform.startswith("linux"):
        return None
    return (
        "System RAM could not be read from /proc/meminfo: profiles are chosen by GPU memory only."
    )


def available_ram_mib(meminfo: str) -> int | None:
    """MemAvailable from the text of /proc/meminfo, or None when it is not there."""
    match = _MEM_AVAILABLE.search(meminfo)
    return int(match.group(1)) // KIB_PER_MIB if match else None
