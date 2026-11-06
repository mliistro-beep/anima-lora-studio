"""The Generator tab on its own: any model, the user's LoRAs and an own folder (M9)."""

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from anima_trainer.preview.generator_settings import LoraEntry, load_generator_settings
from tests.e2e.test_previews_generator import OK, _checkpoints, _option, app  # noqa: F401
from tests.safetensors_helpers import checkpoint_keys, kohya_blocks, write_safetensors

pytestmark = pytest.mark.e2e


def _pairs(arguments):
    """(file, strength) of each --lora in order."""
    args = list(arguments)
    return [(args[index + 1], args[index + 3]) for index, arg in enumerate(args) if arg == "--lora"]


def _real_model(paths):
    """make_runtime leaves empty model files; LoRA checks read the DiT's header."""
    write_safetensors(paths.dit_model("anima-base-v1.0"), checkpoint_keys(28))


def _lora(folder, name, keys=None):
    folder.mkdir(parents=True, exist_ok=True)
    return write_safetensors(folder / name, kohya_blocks(28) if keys is None else keys)


def test_without_checkpoints_or_a_folder_the_image_goes_to_the_data_folder(app):  # noqa: F811
    window, _, generator_runner, paths = app
    tab = window.generator_tab
    tab.prompt.setPlainText("a girl")

    tab.generate_button.click()

    arguments = generator_runner.started[0][0].arguments
    assert "--lora" not in arguments
    assert _option(arguments, "--output").startswith(str(paths.root / "generated") + "/")
    assert str(paths.root / "generated") in tab.output_folder.edit.placeholderText()


def test_the_users_loras_follow_the_project_checkpoint_in_order(app, tmp_path):  # noqa: F811
    window, _, generator_runner, paths = app
    _real_model(paths)
    _, final = _checkpoints(paths)
    first = _lora(tmp_path / "loras", "style.safetensors")
    second = _lora(tmp_path / "loras", "detail.safetensors")
    window.tabs.setCurrentWidget(window.generator_tab)  # the checkpoints are listed when shown
    tab = window.generator_tab
    tab.prompt.setPlainText("a girl")
    tab.loras.add_files([str(first), str(second)])
    tab.loras.strength_box(1).setValue(-0.5)

    tab.generate_button.click()

    assert _pairs(generator_runner.started[0][0].arguments) == [
        (str(final), "1.0"),
        (str(first), "1.0"),
        (str(second), "-0.5"),
    ]


def test_a_lora_that_cannot_be_applied_stops_before_anything_starts(app, tmp_path):  # noqa: F811
    window, _, generator_runner, paths = app
    _real_model(paths)
    loha = _lora(tmp_path / "loras", "loha.safetensors", ["lora_unet_blocks_0_mlp.hada_w1_a"])
    tab = window.generator_tab
    tab.prompt.setPlainText("a girl")
    tab.loras.add_files([str(loha)])

    tab.generate_button.click()

    assert generator_runner.started == []
    assert "ERROR: loha.safetensors is a LoHa file" in tab.log.text()


def test_a_model_file_of_the_user_is_the_dit(app, tmp_path):  # noqa: F811
    window, _, generator_runner, _ = app
    own = tmp_path / "models" / "merge.safetensors"
    own.parent.mkdir()
    write_safetensors(own, checkpoint_keys(28))
    tab = window.generator_tab
    tab.prompt.setPlainText("a girl")
    tab.model_file.set_path(str(own))

    tab.generate_button.click()

    assert _option(generator_runner.started[0][0].arguments, "--dit") == str(own)


def test_the_catalog_model_of_the_generator_is_its_own_choice(app):  # noqa: F811
    window, _, generator_runner, paths = app
    tab = window.generator_tab
    tab.prompt.setPlainText("a girl")
    tab.base_model.setCurrentText("anima-preview")
    _lora(paths.dit_model("anima-preview").parent, paths.dit_model("anima-preview").name, [])

    tab.generate_button.click()

    assert _option(generator_runner.started[0][0].arguments, "--dit") == str(
        paths.dit_model("anima-preview")
    )


def test_the_choices_are_saved_with_the_generation(app, tmp_path):  # noqa: F811
    window, _, _, paths = app
    _real_model(paths)
    lora = _lora(tmp_path / "loras", "style.safetensors")
    tab = window.generator_tab
    tab.prompt.setPlainText("a girl")
    tab.loras.add_files([str(lora)])
    tab.output_folder.set_path(str(tmp_path / "pictures"))

    tab.generate_button.click()

    saved = load_generator_settings(paths.generator_settings_file).settings
    assert saved.loras == (LoraEntry(path=str(lora)),)
    assert saved.output_folder == str(tmp_path / "pictures")


def test_use_project_folder_fills_the_folder_of_the_project(app):  # noqa: F811
    window, _, _, paths = app
    tab = window.generator_tab

    tab.use_project_button.click()

    assert tab.output_folder.path() == str(paths.root / "out" / "generated")


def test_open_folder_creates_and_opens_the_folder_in_use(app, monkeypatch, tmp_path):  # noqa: F811
    window, _, _, paths = app
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    tab = window.generator_tab

    tab.open_folder_button.click()
    tab.output_folder.set_path(str(tmp_path / "pictures"))
    tab.open_folder_button.click()

    assert opened == [
        QUrl.fromLocalFile(str(paths.root / "generated")),
        QUrl.fromLocalFile(str(tmp_path / "pictures")),
    ]
    assert (tmp_path / "pictures").is_dir()
