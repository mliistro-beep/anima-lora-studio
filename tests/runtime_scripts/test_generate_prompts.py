import json
from pathlib import Path

import pytest

from anima_trainer.preview.inference_command import GenerationRequest, build_generation_launch
from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.preview.vram_profile import PROFILES
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime_scripts import GENERATE, SCRIPTS_DIR

pytestmark = pytest.mark.runtime
CHECK = Path(__file__).with_name("check_generate_prompts.py")


@pytest.mark.parametrize("text", ["-lowres", "--steps 3", "a=b", "plain"])
def test_prompts_reach_sd_scripts_whatever_they_start_with(run_script, runtime, text):
    request = GenerationRequest(
        base_model="anima-base-v1.0", lora=None, lora_multiplier=1.0, prompt=text,
        negative_prompt=text, size=SampleSize(1024, 1024), steps=25, cfg_scale=4.0,
        flow_shift=3.0, seed=1, sampler="er_sde", output=Path("/out/x.png"),
    )  # fmt: skip
    spec = build_generation_launch(RuntimePaths(Path("/data")), request, PROFILES, "0", False)
    arguments = spec.arguments[spec.arguments.index(str(GENERATE)) + 1 :]

    result = run_script(
        CHECK, json.dumps(arguments), str(runtime.sd_scripts_dir), env_path=SCRIPTS_DIR
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"prompt": text, "negative_prompt": text}
