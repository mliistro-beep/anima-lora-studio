from datetime import datetime
from pathlib import Path

from anima_trainer.training.runner import log_file_path


def test_log_file_path_matches_citron_naming():
    path = log_file_path(Path("/c/logs"), "proj", datetime(2026, 10, 18, 21, 56, 29))

    assert path == Path("/c/logs/proj_2026-10-18_215629.log")
