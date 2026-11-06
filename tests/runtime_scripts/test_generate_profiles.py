import json
from pathlib import Path

import pytest

from anima_trainer.preview.inference_command import GenerationRequest, build_generation_launch
from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.vram_profile import PROFILES
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime_scripts import SCRIPTS_DIR

pytestmark = pytest.mark.runtime
CHECK = Path(__file__).with_name("check_generate_profiles.py")


def _app_profiles_json() -> str:
    request = GenerationRequest(
        dit=Path("/data/models/dit/anima-base-v1.0.safetensors"), loras=(), prompt="a cat",
        negative_prompt="", size=SampleSize(1024, 1024), steps=25, cfg_scale=4.0,
        flow_shift=3.0, seed=1, sampler="er_sde", output=Path("/out/x.png"),
    )  # fmt: skip
    spec = build_generation_launch(RuntimePaths(Path("/data")), request, PROFILES, "0", False)
    return spec.arguments[spec.arguments.index("--profiles") + 1]


def test_generate_reads_the_app_profiles_and_tiles_the_vae_where_asked(run_script):
    result = run_script(CHECK, _app_profiles_json(), env_path=SCRIPTS_DIR)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["names"] == [profile.name for profile in PROFILES]
    assert report["tiling"] == [profile.vae_tiling for profile in PROFILES]
    assert report["dtype"] == "torch.bfloat16"
