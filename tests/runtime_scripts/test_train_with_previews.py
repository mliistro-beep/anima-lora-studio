import json
from pathlib import Path

import pytest

from anima_trainer.runtime_scripts import SCRIPTS_DIR

pytestmark = pytest.mark.runtime
CHECK = Path(__file__).with_name("check_preview_hooks.py")


def test_hooks_pick_the_sampler_and_skip_previews_that_run_out_of_memory(run_script):
    result = run_script(CHECK, env_path=SCRIPTS_DIR)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["results"] == [
        "er-latents",
        "euler-latents",
        "euler-latents",
        None,
        "euler-latents",
    ]
    assert report["calls"] == ["er_sde", "euler", "euler", "euler"]
    assert report["reports"] == [
        "WARN: Preview skipped, the GPU ran out of memory (prompt 3). Training goes on."
    ]
    assert report["oom_vae_device"] == "cpu"


def test_er_sde_do_sample_gives_anima_latents_repeatable_by_seed(run_script):
    result = run_script(CHECK, env_path=SCRIPTS_DIR)

    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["shape"] == [1, 16, 1, 8, 8]
    assert report["same_seed_equal"] and report["other_seed_differs"]
