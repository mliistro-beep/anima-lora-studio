import json

import pytest

from anima_trainer.runtime_scripts import SELF_CHECK

pytestmark = pytest.mark.runtime


def test_samplers_turn_noise_into_the_data_distribution(run_script):
    result = run_script(SELF_CHECK)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    mean, std = report["expected"]
    er_mean, er_std = report["er_sde"]
    euler_mean, euler_std = report["euler"]
    # 25 steps: ER-SDE is within 5% of the true spread, Euler (an ODE) a bit under it.
    assert er_mean == pytest.approx(mean, abs=0.01)
    assert er_std == pytest.approx(std, rel=0.05)
    assert euler_mean == pytest.approx(mean, abs=0.01)
    assert euler_std == pytest.approx(std, rel=0.15)
    assert er_std > euler_std
    assert report["er_sde_same_seed_equal"] and report["er_sde_other_seed_differs"]
    assert report["velocity_wrapper_max_error"] < 1e-5


def test_noise_and_first_sigma_follow_comfyui(run_script):
    """The same seed must give the image ComfyUI (and Krita) give: ER-SDE's step noise comes from
    a generator on the device seeded like ComfyUI's default_noise_sampler (seed on CUDA, seed + 1
    on the CPU) and the first sigma is ComfyUI's percent_to_sigma(1e-4) for the flow shift."""
    result = run_script(SELF_CHECK)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    # time_snr_shift(3, 1 - 1e-4) = 3 * 0.9999 / (1 + 2 * 0.9999)
    assert report["first_sigma_shift_3"] == pytest.approx(3 * 0.9999 / (1 + 2 * 0.9999), abs=1e-6)
    assert report["first_sigma_shift_1"] == pytest.approx(1 - 1e-4, abs=1e-6)
    assert report["first_sigma_below_one_kept"]
    assert report["cpu_noise_like_comfyui"]
    assert report["cuda_noise_like_comfyui"] in (True, None)  # None: no GPU on this machine
    assert report["er_sde_noise_from_other_device"] in (True, None)
