import sys
from datetime import datetime

import pytest
from PySide6.QtCore import QProcess

from anima_trainer.system.resource_monitor import ResourceMonitor

pytestmark = pytest.mark.integration

NOW = datetime(2026, 10, 28, 19, 40, 0)
STAMP = "RESOURCES 2026-10-28T19:40:00"
MEMINFO = (
    "MemTotal:       14675968 kB\n"
    "MemAvailable:   10485760 kB\n"
    "SwapTotal:      16683008 kB\n"
    "SwapFree:        8493056 kB\n"
)
RAM = "RAM available 10.0 / 14.0 GB | swap used 7.8 / 15.9 GB"
GPU = "GPU 0: VRAM 9.1 / 12.0 GB, 72 C, 165 / 170 W, load 99%"
GPU_ROW = "0, 9318, 12288, 72, 165.40, 170.00, 99"
LONG_INTERVAL_MS = 600_000
WAIT_MS = 5000


@pytest.fixture
def meminfo(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(MEMINFO, encoding="ascii")
    return path


def _monitor(meminfo, code="", interval_ms=LONG_INTERVAL_MS, program=sys.executable, **kwargs):
    return ResourceMonitor(
        interval_ms=interval_ms,
        program=program,
        arguments=("-c", code),
        meminfo=meminfo,
        clock=lambda: NOW,
        **kwargs,
    )


def _first_line(qtbot, monitor):
    with qtbot.waitSignal(monitor.line_ready, timeout=WAIT_MS) as blocker:
        monitor.start()
    monitor.stop()
    return blocker.args[0]


def test_start_samples_ram_swap_and_gpu_at_once(qtbot, meminfo):
    monitor = _monitor(meminfo, f"print({GPU_ROW!r})")

    assert _first_line(qtbot, monitor) == f"{STAMP} | {RAM} | {GPU}"


def test_missing_nvidia_smi_gives_ram_only(qtbot, meminfo, tmp_path):
    monitor = _monitor(meminfo, program=str(tmp_path / "no-such-program"))

    assert _first_line(qtbot, monitor) == f"{STAMP} | {RAM}"


def test_failing_nvidia_smi_gives_ram_only(qtbot, meminfo):
    monitor = _monitor(meminfo, f"print({GPU_ROW!r}); raise SystemExit(9)")

    assert _first_line(qtbot, monitor) == f"{STAMP} | {RAM}"


def test_hanging_nvidia_smi_is_killed_after_the_timeout(qtbot, meminfo):
    monitor = _monitor(meminfo, "import time; time.sleep(30)", timeout_ms=200)

    assert _first_line(qtbot, monitor) == f"{STAMP} | {RAM}"


def test_unreadable_meminfo_gives_gpu_only(qtbot, tmp_path):
    monitor = _monitor(tmp_path / "missing", f"print({GPU_ROW!r})")

    assert _first_line(qtbot, monitor) == f"{STAMP} | {GPU}"


def test_samples_repeat_at_the_interval(qtbot, meminfo):
    monitor = _monitor(meminfo, f"print({GPU_ROW!r})", interval_ms=50)
    lines = []
    monitor.line_ready.connect(lines.append)

    monitor.start()
    qtbot.waitUntil(lambda: len(lines) >= 3, timeout=WAIT_MS)
    monitor.stop()

    assert set(lines) == {f"{STAMP} | {RAM} | {GPU}"}


def test_no_line_after_stop_even_with_a_sample_running(qtbot, meminfo):
    monitor = _monitor(meminfo, "import time; time.sleep(30)")
    lines = []
    monitor.line_ready.connect(lines.append)

    monitor.start()
    monitor.stop()
    qtbot.wait(300)

    assert lines == []
    assert not monitor.is_sampling()


def test_start_twice_and_stop_twice_are_harmless(qtbot, meminfo):
    monitor = _monitor(meminfo, f"print({GPU_ROW!r})")
    lines = []
    monitor.line_ready.connect(lines.append)

    monitor.start()
    monitor.start()
    qtbot.waitUntil(lambda: len(lines) == 1, timeout=WAIT_MS)
    qtbot.wait(200)
    monitor.stop()
    monitor.stop()

    assert len(lines) == 1


def test_a_stale_failed_start_after_a_restart_is_ignored(qtbot, meminfo, tmp_path):
    smi = tmp_path / "nvidia-smi"
    monitor = _monitor(meminfo, program=str(smi))
    lines = []
    monitor.line_ready.connect(lines.append)

    # The first sample fails to start (no program yet) and its error may still be queued when
    # the second sample, with a working program, begins: only the second may give a line.
    monitor.start()
    monitor.stop()
    smi.write_text(f"#!{sys.executable}\nprint({GPU_ROW!r})\n", encoding="utf-8")
    smi.chmod(0o755)
    monitor.start()
    qtbot.waitUntil(lambda: bool(lines), timeout=WAIT_MS)
    qtbot.wait(200)
    monitor.stop()

    assert lines == [f"{STAMP} | {RAM} | {GPU}"]


def test_an_unkillable_nvidia_smi_does_not_stop_the_samples(qtbot, meminfo, monkeypatch):
    # A driver hang can leave nvidia-smi in uninterruptible sleep, where kill() does nothing.
    real_kill = QProcess.kill
    monkeypatch.setattr(QProcess, "kill", lambda _self: None)
    monitor = _monitor(meminfo, "import time; time.sleep(30)", interval_ms=100, timeout_ms=100)
    lines = []
    monitor.line_ready.connect(lines.append)

    monitor.start()
    qtbot.waitUntil(lambda: len(lines) >= 2, timeout=WAIT_MS)
    monitor.stop()

    assert set(lines) == {f"{STAMP} | {RAM}"}
    for process in monitor.findChildren(QProcess):  # the abandoned samples
        real_kill(process)
        process.waitForFinished(WAIT_MS)
