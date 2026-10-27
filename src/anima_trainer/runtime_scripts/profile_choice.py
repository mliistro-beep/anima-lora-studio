"""Which memory profile generate.py starts from, given the free GPU memory it sees.

Pure Python (no torch), so the app's tests cover it too. The free memory is read by the
generator itself right before loading the models: it is the current value, for the same GPU
CUDA_VISIBLE_DEVICES selects, and reading it does not block the app's window.
"""

from collections.abc import Sequence

MIB = 1 << 20


def first_fitting(
    estimates_mib: Sequence[int], free_mib: int | None, margin_mib: int
) -> int | None:
    """Index of the first profile whose estimate plus the margin fits; None when none fits.

    Unknown free memory (no CUDA) starts from the first profile.
    """
    if free_mib is None:
        return 0
    return next(
        (
            index
            for index, estimate in enumerate(estimates_mib)
            if estimate + margin_mib <= free_mib
        ),
        None,
    )


def not_enough_memory(estimates_mib: Sequence[int], free_mib: int, margin_mib: int) -> str:
    needed = min(estimates_mib, default=0) + margin_mib
    return f"Not enough free GPU memory: about {needed} MiB needed, {free_mib} MiB free."
