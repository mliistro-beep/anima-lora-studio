"""Creating a folder from the TUI's folder browser."""

import pytest

from anima_trainer.cli.tui.folder_picker import NewFolderError, create_folder


def test_a_folder_is_created_inside_the_shown_one(tmp_path):
    created = create_folder(tmp_path, "  my_lora  ")

    assert created == tmp_path / "my_lora"
    assert created.is_dir()


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "ERROR: Write a folder name."),
        ("   ", "ERROR: Write a folder name."),
        ("a/b", "ERROR: A folder name cannot contain '/'."),
        ("a\0b", "ERROR: A folder name cannot contain a NUL character."),
        (".", "ERROR: '.' is not a folder name."),
        ("..", "ERROR: '..' is not a folder name."),
        (".hidden", "ERROR: A name starting with '.' makes a hidden folder, not listed here."),
    ],
)
def test_names_that_are_not_folder_names_are_refused(tmp_path, name, message):
    with pytest.raises(NewFolderError) as refused:
        create_folder(tmp_path, name)

    assert str(refused.value) == message
    assert list(tmp_path.iterdir()) == []


def test_an_existing_folder_is_not_reused(tmp_path):
    (tmp_path / "out").mkdir()

    with pytest.raises(NewFolderError) as refused:
        create_folder(tmp_path, "out")

    assert str(refused.value) == "ERROR: 'out' already exists here."


def test_a_name_used_by_a_file_is_refused(tmp_path):
    (tmp_path / "notes.txt").write_text("")

    with pytest.raises(NewFolderError) as refused:
        create_folder(tmp_path, "notes.txt")

    assert str(refused.value) == "ERROR: 'notes.txt' already exists here."


def test_a_folder_that_cannot_be_created_says_why(tmp_path):
    missing_parent = tmp_path / "gone"

    with pytest.raises(NewFolderError) as refused:
        create_folder(missing_parent, "out")

    assert str(refused.value).startswith(f"ERROR: Cannot create {missing_parent / 'out'}: ")
