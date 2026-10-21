import threading

import pytest

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.system.gpu_info import FALLBACK_CHOICES
from anima_trainer.system.gpu_worker import GpuDetector

pytestmark = pytest.mark.integration
TIMEOUT_MS = 5000


def test_reports_detected_gpus_from_a_background_thread(qtbot, tmp_path):
    seen_threads = []

    def detect(paths):
        seen_threads.append(threading.current_thread())
        return (f"0: GPU in {paths.root.name}",)

    detector = GpuDetector(detect)

    with qtbot.waitSignal(detector.detected, timeout=TIMEOUT_MS) as blocker:
        detector.start(RuntimePaths(tmp_path / "citron"))

    assert blocker.args == [("0: GPU in citron",)]
    assert seen_threads[0] is not threading.main_thread()
    assert not detector.is_running()


def test_unexpected_error_falls_back_to_citron_choices(qtbot, tmp_path):
    def detect(_paths):
        raise ValueError("boom")

    detector = GpuDetector(detect)

    with qtbot.waitSignal(detector.detected, timeout=TIMEOUT_MS) as blocker:
        detector.start(RuntimePaths(tmp_path))

    assert blocker.args == [FALLBACK_CHOICES]


def test_only_the_latest_detection_is_reported(qtbot, tmp_path):
    release_first = threading.Event()

    def detect(paths):
        if paths.root.name == "first":
            release_first.wait(TIMEOUT_MS / 1000)
        return (paths.root.name,)

    detector = GpuDetector(detect)
    results = []
    detector.detected.connect(results.append)
    detector.start(RuntimePaths(tmp_path / "first"))
    with qtbot.waitSignal(detector.detected, timeout=TIMEOUT_MS):
        detector.start(RuntimePaths(tmp_path / "second"))

    release_first.set()
    qtbot.waitUntil(lambda: not detector.is_running(), timeout=TIMEOUT_MS)

    assert results == [("second",)]


def test_detection_error_is_logged_and_reported(qtbot, tmp_path, caplog):
    def detect(_paths):
        raise ValueError("boom")

    detector = GpuDetector(detect)

    with qtbot.waitSignal(detector.warning, timeout=TIMEOUT_MS) as blocker:
        detector.start(RuntimePaths(tmp_path))

    assert "boom" in blocker.args[0]
    assert "boom" in caplog.text


def test_shutdown_waits_for_running_detection(qtbot, tmp_path):
    finished = threading.Event()

    def detect(_paths):
        finished.wait(0.2)
        finished.set()
        return ("0",)

    detector = GpuDetector(detect)
    detector.start(RuntimePaths(tmp_path))

    detector.shutdown()

    assert finished.is_set()
    assert not detector.is_running()
