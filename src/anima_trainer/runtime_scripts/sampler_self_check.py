"""Check the samplers on Gaussian data, where the exact denoiser is known (runs with torch).

With x0 ~ N(mean, std^2) per element and x_t = (1 - sigma) x0 + sigma noise, the best
denoiser is E[x0 | x_t]; a correct sampler then turns pure noise into N(mean, std^2).
Prints one JSON object: the sample mean and std per sampler and the determinism checks.
"""

import json
import sys

import torch
from samplers import denoiser_from_velocity, er_sde, euler, flow_sigmas

MEAN, STD = 0.3, 0.5
ELEMENTS = 200_000
STEPS = 25
FLOW_SHIFT = 3.0


def exact_denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    alpha = 1 - sigma
    gain = alpha * STD**2 / (alpha**2 * STD**2 + sigma**2)
    return MEAN + gain * (x - alpha * MEAN)


def exact_velocity(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (x - exact_denoiser(x, sigma)) / sigma


def _er_sde(seed: int) -> torch.Tensor:
    noise = torch.randn(ELEMENTS, generator=torch.Generator().manual_seed(seed))
    sigmas = flow_sigmas(STEPS, FLOW_SHIFT, torch.device("cpu"))
    return er_sde(exact_denoiser, noise, sigmas, torch.Generator().manual_seed(seed + 1))


def main() -> int:
    sigmas = flow_sigmas(STEPS, FLOW_SHIFT, torch.device("cpu"))
    noise = torch.randn(ELEMENTS, generator=torch.Generator().manual_seed(0))
    euler_out = euler(exact_velocity, noise, sigmas)
    er_out = _er_sde(0)
    wrapped = denoiser_from_velocity(exact_velocity)(noise * 0.5, torch.tensor(0.5))
    report = {
        "euler": [euler_out.mean().item(), euler_out.std().item()],
        "er_sde": [er_out.mean().item(), er_out.std().item()],
        "er_sde_same_seed_equal": bool(torch.equal(_er_sde(1), _er_sde(1))),
        "er_sde_other_seed_differs": not torch.equal(_er_sde(1), _er_sde(2)),
        "velocity_wrapper_max_error": (wrapped - exact_denoiser(noise * 0.5, torch.tensor(0.5)))
        .abs()
        .max()
        .item(),
        "expected": [MEAN, STD],
    }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
