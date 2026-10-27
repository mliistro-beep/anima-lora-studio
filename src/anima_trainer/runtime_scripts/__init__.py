"""Scripts run by the training venv's Python (torch, sd-scripts), not imported by the app.

They are started as files (`python <path>/generate.py ...`), so each one imports its
siblings by plain module name and sd-scripts' `library` package from the sd-scripts folder.
"""

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
TRAIN_WITH_PREVIEWS = SCRIPTS_DIR / "train_with_previews.py"
GENERATE = SCRIPTS_DIR / "generate.py"
SELF_CHECK = SCRIPTS_DIR / "sampler_self_check.py"
