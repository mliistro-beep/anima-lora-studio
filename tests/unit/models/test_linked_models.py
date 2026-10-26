import os

from anima_trainer.models.linked_models import LinkedModel, is_unchanged, link_for


def test_link_records_size_and_modification_time(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"abc")

    link = link_for(path)

    assert link == LinkedModel(path, 3, path.stat().st_mtime_ns)
    assert is_unchanged(link)


def test_moved_resized_or_touched_file_is_no_longer_valid(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"abc")
    link = link_for(path)

    os.utime(path, ns=(1, 1))
    assert not is_unchanged(link)
    assert not is_unchanged(LinkedModel(path, 4, path.stat().st_mtime_ns))
    path.unlink()
    assert not is_unchanged(LinkedModel(path, 3, 1))


def test_folder_is_not_a_valid_link(tmp_path):
    assert not is_unchanged(LinkedModel(tmp_path, tmp_path.stat().st_size, 0))
