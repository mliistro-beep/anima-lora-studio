import json
from pathlib import Path

import pytest

from anima_trainer.runtime_scripts import SCRIPTS_DIR

pytestmark = pytest.mark.runtime
CHECK = Path(__file__).with_name("check_generate_png.py")


def test_the_saved_png_holds_the_texts(run_script, tmp_path):
    result = run_script(CHECK, str(tmp_path / "image.png"), env_path=SCRIPTS_DIR)

    assert result.returncode == 0, result.stderr
    texts = json.loads(result.stdout.strip().splitlines()[-1])
    assert texts == {"parameters": "a cat", "anima_lora_studio": "{}"}
