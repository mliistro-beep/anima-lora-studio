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
