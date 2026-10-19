"""Training step estimate, same formula as Citron's configure_training()."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StepEstimate:
    steps_per_epoch: int
    total_steps: int


def estimate_steps(
    image_count: int, repeats: int, batch_size: int, grad_accum: int, epochs: int
) -> StepEstimate:
    effective_batch = max(batch_size, 1) * max(grad_accum, 1)
    steps_per_epoch = math.ceil((image_count * repeats) / effective_batch)
    return StepEstimate(steps_per_epoch, steps_per_epoch * epochs)
