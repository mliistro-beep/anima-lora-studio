import pytest

from anima_trainer.dataset.validator import DatasetError, validate_dataset


def _touch(directory, *names):
    for name in names:
        (directory / name).write_text("")


def test_counts_images_and_finds_missing_captions(tmp_path):
    _touch(tmp_path, "a.png", "a.txt", "b.JPG", "c.webp", "c.txt", "notes.md")

    report = validate_dataset(tmp_path)

    assert report.image_count == 3
    assert report.missing_captions == ("b.JPG",)
    assert report.warnings == ("1 image(s) are missing caption (.txt) files.",)


def test_all_supported_extensions_are_counted(tmp_path):
    names = [
        f"img{i}{ext}" for i, ext in enumerate([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"])
    ]
    _touch(tmp_path, *names, *[n.rsplit(".", 1)[0] + ".txt" for n in names])

    report = validate_dataset(tmp_path)

    assert report.image_count == 6
    assert report.missing_captions == ()
    assert report.warnings == ()


def test_missing_captions_are_sorted(tmp_path):
    _touch(tmp_path, "z.png", "a.png", "m.png")

    report = validate_dataset(tmp_path)

    assert report.missing_captions == ("a.png", "m.png", "z.png")


def test_subdirectories_with_image_names_are_ignored(tmp_path):
    (tmp_path / "folder.png").mkdir()
    _touch(tmp_path, "a.png", "a.txt")

    assert validate_dataset(tmp_path).image_count == 1


def test_empty_directory_warns(tmp_path):
    report = validate_dataset(tmp_path)

    assert report.image_count == 0
    assert report.warnings == ("No image files found in directory.",)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(DatasetError, match="Directory not found"):
        validate_dataset(tmp_path / "nope")


def test_file_instead_of_directory_raises(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("")

    with pytest.raises(DatasetError, match="Not a directory"):
        validate_dataset(target)


def test_unreadable_directory_raises_dataset_error(tmp_path, monkeypatch):
    def deny(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(tmp_path), "iterdir", deny)

    with pytest.raises(DatasetError, match="Cannot read"):
        validate_dataset(tmp_path)
