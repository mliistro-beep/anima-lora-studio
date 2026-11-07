import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLineEdit,
    QSpinBox,
)

from anima_trainer.settings.field_specs import ADVANCED_FIELDS, BASIC_FIELDS
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.ui.widgets.path_picker import PathPicker
from anima_trainer.ui.widgets.settings_form import SettingsForm


@pytest.fixture
def basic(qtbot):
    form = SettingsForm(BASIC_FIELDS)
    qtbot.addWidget(form)
    return form


@pytest.fixture
def advanced(qtbot):
    form = SettingsForm(ADVANCED_FIELDS)
    qtbot.addWidget(form)
    return form


def test_defaults_round_trip_into_identical_settings(basic, advanced):
    defaults = TrainingSettings()
    basic.set_values(defaults)
    advanced.set_values(defaults)

    rebuilt = defaults.with_changes(**basic.values(), **advanced.values())

    assert rebuilt == defaults


def test_values_cover_exactly_the_form_fields(basic):
    assert set(basic.values()) == {spec.name for spec in BASIC_FIELDS}


def test_widget_kinds(basic, advanced):
    assert isinstance(basic.field_widget("project_name"), QLineEdit)
    assert isinstance(basic.field_widget("image_directory"), PathPicker)
    assert isinstance(basic.field_widget("base_model"), QComboBox)
    assert isinstance(basic.field_widget("network_dim"), QSpinBox)
    assert isinstance(basic.field_widget("learning_rate"), QLineEdit)
    assert isinstance(basic.field_widget("caption_dropout"), QDoubleSpinBox)
    assert isinstance(advanced.field_widget("cache_latents"), QCheckBox)


def test_small_floats_are_kept_exactly(basic):
    basic.set_values(TrainingSettings(learning_rate=0.00012345))

    assert TrainingSettings(**basic.values()).learning_rate == 0.00012345


def test_spin_box_limits_follow_the_specs(basic, advanced):
    assert basic.field_widget("resolution").minimum() == 64
    assert advanced.field_widget("lr_warmup_steps").minimum() == 0
    assert advanced.field_widget("seed").minimum() < 0


def test_edited_values_are_returned(basic, advanced):
    basic.field_widget("network_dim").setValue(8)
    basic.field_widget("base_model").setCurrentText("anima-preview")
    advanced.field_widget("cache_latents").setChecked(False)

    assert basic.values()["network_dim"] == 8
    assert basic.values()["base_model"] == "anima-preview"
    assert advanced.values()["cache_latents"] is False


def test_invalid_float_text_is_returned_for_validation(basic):
    basic.field_widget("learning_rate").setText("abc")

    assert basic.values()["learning_rate"] == "abc"


def test_gpu_choices_select_the_saved_index(basic):
    basic.set_gpu_choices(("0: NVIDIA A", "1: NVIDIA B"), "1")

    assert basic.field_widget("gpu_index").currentText() == "1: NVIDIA B"
    assert basic.values()["gpu_index"] == "1"


def test_gpu_value_before_detection_is_the_saved_index(basic):
    basic.set_values(TrainingSettings(gpu_index="2"))

    assert basic.values()["gpu_index"] == "2"


def test_single_value_reads_one_field(basic):
    basic.field_widget("network_dim").setValue(12)

    assert basic.value("network_dim") == 12


def test_ampersand_in_a_group_title_is_shown_and_makes_no_shortcut(basic, advanced):
    boxes = (*basic.findChildren(QGroupBox), *advanced.findChildren(QGroupBox))
    titles = [box.title() for box in boxes]

    assert "Project && Paths" in titles
    assert "Noise && Flow" in titles
    assert [title for title in titles if QKeySequence.mnemonic(title).count()] == []


def test_one_value_can_be_set(basic):
    basic.set_value("network_dim", 64)

    assert basic.value("network_dim") == 64


def test_only_the_named_fields_take_the_given_values(basic):
    basic.set_value("network_dim", 64)
    basic.set_value("network_alpha", 64)

    basic.set_values(TrainingSettings(), only={"network_alpha"})

    assert basic.value("network_dim") == 64
    assert basic.value("network_alpha") == TrainingSettings().network_alpha
