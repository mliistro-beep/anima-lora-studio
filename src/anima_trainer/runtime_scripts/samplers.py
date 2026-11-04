"""Euler and ER-SDE samplers for rectified-flow models such as Anima (runs with torch).

ER-SDE is the VP ER-SDE-Solver-3 of Cui et al. (arXiv:2309.06169, reference code MIT),
ported from ComfyUI's comfy/k_diffusion/sampling.py `sample_er_sde` (GPL-3.0). For a flow
model x_t = (1 - sigma) x0 + sigma noise, the network predicts the velocity v and the
denoised latent is x0 = x_t - sigma v; ComfyUI's CONST model sampling does the same.
"""

from collections.abc import Callable

import torch

Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (x, sigma) -> x0
Velocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (x, sigma) -> v

NUM_INTEGRATION_POINTS = 200
FIRST_SIGMA_OFFSET = 1e-4  # ComfyUI starts flow models at sigma = 1 - 1e-4, where logSNR is finite
MAX_STAGE = 3


def flow_sigmas(steps: int, flow_shift: float, device: torch.device) -> torch.Tensor:
    """Same schedule as sd-scripts' Anima sampler: linear 1 -> 0, then the flow shift."""
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    if flow_shift != 1.0:
        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)
    return sigmas


def denoiser_from_velocity(velocity: Velocity) -> Denoiser:
    def denoise(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return x - velocity(x, sigma).float() * sigma

    return denoise


def euler(velocity: Velocity, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """sd-scripts' Euler step: x <- x + v (sigma_next - sigma)."""
    x = x.float()
    for i in range(len(sigmas) - 1):
        x = x + velocity(x, sigmas[i]).float() * (sigmas[i + 1] - sigmas[i])
    return x


def _noise_scaler(value: torch.Tensor) -> torch.Tensor:
    return value * ((value**0.3).exp() + 10.0)


def er_sde(
    denoise: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    generator: torch.Generator,
    s_noise: float = 1.0,
) -> torch.Tensor:
    """ER-SDE-Solver-3; the injected noise is drawn on the CPU from `generator`, so a seed
    gives the same image on any device."""
    x = x.float()
    sigmas = sigmas.float().clone()
    if sigmas[0] >= 1:
        sigmas[0] = 1.0 - FIRST_SIGMA_OFFSET
    er_lambdas = sigmas / (1.0 - sigmas)  # sigma_t / alpha_t, alpha_t = 1 - sigma_t
    points = torch.arange(0, NUM_INTEGRATION_POINTS, dtype=torch.float32, device=x.device)
    old_denoised: torch.Tensor | None = None
    old_denoised_d: torch.Tensor | None = None
    for i in range(len(sigmas) - 1):
        denoised = denoise(x, sigmas[i]).float()
        stage = min(MAX_STAGE, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lambda_s, lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s, alpha_t = sigmas[i] / lambda_s, sigmas[i + 1] / lambda_t
            r = _noise_scaler(lambda_t) / _noise_scaler(lambda_s)
            x = (alpha_t / alpha_s) * r * x + alpha_t * (1 - r) * denoised
            if stage >= 2 and old_denoised is not None:
                dt = lambda_t - lambda_s
                step = -dt / NUM_INTEGRATION_POINTS
                positions = lambda_t + points * step
                scaled = _noise_scaler(positions)
                s = torch.sum(1 / scaled) * step
                denoised_d = (denoised - old_denoised) / (lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * _noise_scaler(lambda_t)) * denoised_d
                if stage >= 3 and old_denoised_d is not None:
                    s_u = torch.sum((positions - lambda_s) / scaled) * step
                    denoised_u = (denoised_d - old_denoised_d) / (
                        (lambda_s - er_lambdas[i - 2]) / 2
                    )
                    x = x + alpha_t * ((dt**2) / 2 + s_u * _noise_scaler(lambda_t)) * denoised_u
                old_denoised_d = denoised_d
            if s_noise > 0:
                noise = torch.randn(x.shape, generator=generator, dtype=torch.float32)
                spread = (lambda_t**2 - lambda_s**2 * r**2).sqrt().nan_to_num(nan=0.0)
                x = x + alpha_t * noise.to(x.device) * s_noise * spread
        old_denoised = denoised
    return x
