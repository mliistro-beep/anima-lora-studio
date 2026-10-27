import json
from pathlib import Path

import pytest

from anima_trainer.runtime_scripts import SCRIPTS_DIR

pytestmark = pytest.mark.runtime
CHECK = Path(__file__).with_name("check_preview_hooks.py")


def _cases(run_script):
    result = run_script(CHECK, env_path=SCRIPTS_DIR)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    return report["cases"], report["reports"]


def test_hooks_pick_the_sampler_the_prompt_asks_for(run_script):
    cases, _ = _cases(run_script)

    assert (cases["er_sde"]["result"], cases["er_sde"]["samplers"]) == ("er-latents", ["er_sde"])
    assert (cases["euler"]["result"], cases["euler"]["samplers"]) == ("euler-latents", ["euler"])
    assert cases["default"]["samplers"] == ["euler"]  # sd-scripts' own sampler


def test_a_preview_that_runs_out_of_memory_is_skipped_and_the_training_goes_on(run_script):
    cases, reports = _cases(run_script)

    assert cases["oom"]["result"] is None and cases["oom"]["error"] is None
    assert cases["oom"]["vae_device"] == "cpu"  # moved back where it was
    assert reports == [
        "WARN: Preview skipped, the GPU ran out of memory (prompt 3). Training goes on."
    ]
    assert cases["after_oom"]["result"] == "euler-latents"


def test_other_errors_still_stop_the_preview(run_script):
    cases, _ = _cases(run_script)

    assert cases["other_error"]["error"] == "RuntimeError"


@pytest.mark.parametrize("case", ["er_sde", "oom", "other_error"])
def test_previews_decode_in_tiles_and_leave_the_training_vae_as_it_was(run_script, case):
    # tiles only around the preview: use_tiling also changes the training's own VAE encode
    cases, _ = _cases(run_script)

    assert (cases[case]["tiling_during"], cases[case]["tiling_after"]) == (True, False)


def test_a_vae_already_in_tiles_stays_in_tiles(run_script):
    cases, _ = _cases(run_script)

    already = cases["already_tiled"]
    assert (already["tiling_during"], already["tiling_after"]) == (True, True)
    assert not already["tiling_disabled"]


def test_a_vae_without_tiles_decodes_as_before(run_script):
    cases, _ = _cases(run_script)

    assert cases["vae_2d"]["result"] == "euler-latents"
    assert cases["vae_2d"]["tiling_during"] is None


def test_er_sde_do_sample_gives_anima_latents_repeatable_by_seed(run_script):
    result = run_script(CHECK, env_path=SCRIPTS_DIR)

    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["shape"] == [1, 16, 1, 8, 8]
    assert report["same_seed_equal"] and report["other_seed_differs"]
