"""Check the samplers on Gaussian data, where the exact denoiser is known (runs with torch).

With x0 ~ N(mean, std^2) per element and x_t = (1 - sigma) x0 + sigma noise, the best
denoiser is E[x0 | x_t]; a correct sampler then turns pure noise into N(mean, std^2).
Prints one JSON object: the sample mean and std per sampler and the determinism checks.
"""

import json
import sys

import torch
from samplers import (
    comfy_noise_generator,
    denoiser_from_velocity,
    er_sde,
    euler,
    flow_sigmas,
    offset_first_sigma,
)

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
    generator = comfy_noise_generator(seed, torch.device("cpu"))
    return er_sde(exact_denoiser, noise, sigmas, generator, flow_shift=FLOW_SHIFT)


def _first_sigma(flow_shift: float, first: float = 1.0) -> float:
    sigmas = torch.tensor([first, 0.5, 0.0])
    return offset_first_sigma(sigmas, flow_shift)[0].item()


def _same_draw(generator: torch.Generator, reference: torch.Generator) -> bool:
    device = reference.device
    return torch.equal(
        torch.randn(8, generator=generator, device=device),
        torch.randn(8, generator=reference, device=device),
    )


def _cuda_checks() -> dict[str, bool | None]:
    """ComfyUI's step noise on a GPU; None when this machine has none."""
    if not torch.cuda.is_available():
        return {"cuda_noise_like_comfyui": None, "er_sde_noise_from_other_device": None}
    cuda = torch.device("cuda")
    like_comfy = _same_draw(
        comfy_noise_generator(7, cuda), torch.Generator(device=cuda).manual_seed(7)
    )
    sigmas = flow_sigmas(4, FLOW_SHIFT, torch.device("cpu"))
    x = torch.randn(16, generator=torch.Generator().manual_seed(0))
    out = er_sde(exact_denoiser, x, sigmas, comfy_noise_generator(3, cuda), flow_shift=FLOW_SHIFT)
    return {
        "cuda_noise_like_comfyui": like_comfy,
        "er_sde_noise_from_other_device": out.device == x.device and bool(out.isfinite().all()),
    }


def _comfyui_checks() -> dict[str, float | bool | None]:
    cpu = torch.device("cpu")
    return {
        "first_sigma_shift_3": _first_sigma(FLOW_SHIFT),
        "first_sigma_shift_1": _first_sigma(1.0),
        "first_sigma_below_one_kept": _first_sigma(FLOW_SHIFT, first=0.9)
        == torch.tensor(0.9).item(),
        "cpu_noise_like_comfyui": _same_draw(
            comfy_noise_generator(7, cpu), torch.Generator().manual_seed(8)
        ),
        **_cuda_checks(),
    }


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
        **_comfyui_checks(),
    }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
