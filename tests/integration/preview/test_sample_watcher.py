from datetime import datetime

import pytest

from anima_trainer.preview.sample_watcher import SampleWatcher

pytestmark = pytest.mark.integration
TIMEOUT_MS = 3000
SINCE = datetime(2026, 10, 27, 9, 0, 0)


def _touch(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"png")
    return folder / name


@pytest.fixture
def watcher(qtbot):
    instance = SampleWatcher(interval_ms=20)
    yield instance
    instance.stop()


def test_reports_images_that_appear_even_if_the_folder_is_created_later(qtbot, watcher, tmp_path):
    output = tmp_path / "out"
    watcher.watch(output, "my_lora", SINCE)

    with qtbot.waitSignal(watcher.samples_changed, timeout=TIMEOUT_MS) as blocker:
        image = _touch(output / "sample", "my_lora_e000001_00_20261027100000_42.png")

    assert [sample.path for sample in blocker.args[0]] == [image]


def test_only_changes_are_reported(qtbot, watcher, tmp_path):
    output = tmp_path / "out"
    _touch(output / "sample", "my_lora_e000001_00_20261027100000.png")
    reports = []
    watcher.samples_changed.connect(reports.append)

    with qtbot.waitSignal(watcher.samples_changed, timeout=TIMEOUT_MS):
        watcher.watch(output, "my_lora", SINCE)
    qtbot.wait(150)

    assert len(reports) == 1


def test_an_image_is_reported_again_when_its_file_finishes_writing(qtbot, watcher, tmp_path):
    # sd-scripts writes the PNG in place: a poll can see it half written.
    output = tmp_path / "out"
    image = _touch(output / "sample", "my_lora_e000001_00_20261027100000.png")
    with qtbot.waitSignal(watcher.samples_changed, timeout=TIMEOUT_MS):
        watcher.watch(output, "my_lora", SINCE)

    with qtbot.waitSignal(watcher.samples_changed, timeout=TIMEOUT_MS) as blocker:
        image.write_bytes(b"png, now complete")

    assert [sample.path for sample in blocker.args[0]] == [image]


def test_stop_ends_the_polling_and_refresh_reads_once(qtbot, watcher, tmp_path):
    output = tmp_path / "out"
    watcher.watch(output, "my_lora", SINCE)
    watcher.stop()
    _touch(output / "sample", "my_lora_e000001_00_20261027100000.png")

    with qtbot.assertNotEmitted(watcher.samples_changed, wait=150):
        pass
    assert not watcher.is_watching()
    assert len(watcher.refresh()) == 1
