from datetime import datetime

from anima_trainer.preview.sample_files import SampleImage, parse_sample_name, scan_samples

PROJECT = "my_lora"


def _touch(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"png")
    return folder / name


def test_parses_epoch_prompt_time_and_seed_from_sd_scripts_name(tmp_path):
    path = tmp_path / "my_lora_e000003_01_20261027101500_42.png"

    image = parse_sample_name(path, PROJECT)

    assert image == SampleImage(path, 3, 1, datetime(2026, 10, 27, 10, 15, 0), 42)


def test_seed_suffix_is_optional_and_epoch_zero_is_before_training(tmp_path):
    image = parse_sample_name(tmp_path / "my_lora_e000000_00_20261027101500.png", PROJECT)

    assert image is not None
    assert (image.epoch, image.seed) == (0, None)


def test_other_projects_step_samples_and_other_files_are_ignored(tmp_path):
    for name in (
        "other_e000001_00_20261027101500.png",
        "my_lora_000120_00_20261027101500.png",
        "my_lora_e000001_00_20261027101500.jpg",
        "my_lora_x_e000001_00_20261027101500.png",
        "my_lora_e000001_00_2026102710.png",
    ):
        assert parse_sample_name(tmp_path / name, PROJECT) is None


def test_project_names_with_regex_characters_are_matched_literally(tmp_path):
    path = tmp_path / "a.b-c_e000001_00_20261027101500.png"

    assert parse_sample_name(path, "a.b-c") is not None
    assert parse_sample_name(tmp_path / "aXb-c_e000001_00_20261027101500.png", "a.b-c") is None


def test_scan_sorts_by_epoch_then_prompt(tmp_path):
    folder = tmp_path / "sample"
    _touch(folder, "my_lora_e000002_00_20261027102000_42.png")
    _touch(folder, "my_lora_e000001_01_20261027101000_42.png")
    _touch(folder, "my_lora_e000001_00_20261027100000_42.png")

    images = scan_samples(folder, PROJECT)

    assert [(image.epoch, image.prompt_index) for image in images] == [(1, 0), (1, 1), (2, 0)]


def test_scan_keeps_only_images_of_the_current_run(tmp_path):
    folder = tmp_path / "sample"
    _touch(folder, "my_lora_e000005_00_20261026090000.png")
    current = _touch(folder, "my_lora_e000001_00_20261027100000.png")

    images = scan_samples(folder, PROJECT, since=datetime(2026, 10, 27, 9, 0, 0))

    assert [image.path for image in images] == [current]


def test_scan_of_a_missing_folder_is_empty(tmp_path):
    assert scan_samples(tmp_path / "sample", PROJECT) == ()


def test_impossible_timestamp_is_ignored(tmp_path):
    assert parse_sample_name(tmp_path / "my_lora_e000001_00_20261327101500.png", PROJECT) is None


def test_scan_without_a_start_shows_only_the_last_run(tmp_path):
    folder = tmp_path / "sample"
    _touch(folder, "my_lora_e000009_00_20261026090000.png")
    _touch(folder, "my_lora_e000010_00_20261026100000.png")
    first = _touch(folder, "my_lora_e000001_00_20261027100000.png")
    second = _touch(folder, "my_lora_e000002_00_20261027110000.png")

    images = scan_samples(folder, PROJECT)

    assert [image.path for image in images] == [first, second]


def test_a_run_stopped_after_its_first_prompt_starts_a_new_run_at_the_same_slot(tmp_path):
    folder = tmp_path / "sample"
    _touch(folder, "my_lora_e000001_00_20261026090000.png")
    current = _touch(folder, "my_lora_e000001_00_20261027100000.png")

    assert [image.path for image in scan_samples(folder, PROJECT)] == [current]


def test_images_of_one_second_stay_in_one_run(tmp_path):
    folder = tmp_path / "sample"
    second = _touch(folder, "my_lora_e000001_01_20261027100000.png")
    first = _touch(folder, "my_lora_e000001_00_20261027100000.png")

    assert [image.path for image in scan_samples(folder, PROJECT)] == [first, second]
