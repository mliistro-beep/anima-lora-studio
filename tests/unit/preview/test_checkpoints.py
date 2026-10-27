from anima_trainer.preview.checkpoints import Checkpoint, list_checkpoints


def _touch(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"")
    return folder / name


def test_lists_epoch_checkpoints_in_order_then_the_final_lora(tmp_path):
    second = _touch(tmp_path, "my_lora-000002.safetensors")
    final = _touch(tmp_path, "my_lora.safetensors")
    first = _touch(tmp_path, "my_lora-000001.safetensors")

    assert list_checkpoints(tmp_path, "my_lora") == (
        Checkpoint(first, 1),
        Checkpoint(second, 2),
        Checkpoint(final, None),
    )


def test_ignores_other_projects_states_and_other_files(tmp_path):
    _touch(tmp_path, "other-000001.safetensors")
    _touch(tmp_path, "my_lora-000001.ckpt")
    _touch(tmp_path, "my_lora-state.safetensors")
    (tmp_path / "my_lora-000001-state").mkdir()

    assert list_checkpoints(tmp_path, "my_lora") == ()


def test_label_names_the_epoch_or_the_final_lora(tmp_path):
    assert Checkpoint(tmp_path / "a-000003.safetensors", 3).label == "Epoch 3"
    assert Checkpoint(tmp_path / "a.safetensors", None).label == "Final"


def test_missing_folder_has_no_checkpoints(tmp_path):
    assert list_checkpoints(tmp_path / "nope", "my_lora") == ()
