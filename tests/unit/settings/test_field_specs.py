import annotated_types
import pytest
from pydantic import ValidationError

from anima_trainer.settings.choices import BASE_MODELS, OPTIMIZERS
from anima_trainer.settings.field_specs import (
    ADVANCED_FIELDS,
    BASIC_FIELDS,
    INTERNAL_FIELDS,
    describe_validation_error,
    field_label,
)
from anima_trainer.settings.schema import TrainingSettings

# Citron's basic_inputs / adv_inputs order (app.py, build_ui).
CITRON_BASIC_ORDER = (
    "project_name", "base_model", "image_directory", "output_directory",
    "network_dim", "network_alpha", "learning_rate", "max_train_epochs",
    "resolution", "repeats", "caption_dropout", "gpu_index",
)  # fmt: skip


def _schema_minimum(name):
    for item in TrainingSettings.model_fields[name].metadata:
        if isinstance(item, annotated_types.Ge):
            return item.ge
    return None


def test_every_user_setting_has_exactly_one_spec():
    names = [spec.name for spec in (*BASIC_FIELDS, *ADVANCED_FIELDS)]

    assert len(names) == len(set(names))
    assert set(names) | set(INTERNAL_FIELDS) == set(TrainingSettings.model_fields)


def test_basic_fields_are_the_same_set_as_citron_training_tab():
    assert {spec.name for spec in BASIC_FIELDS} == set(CITRON_BASIC_ORDER)


@pytest.mark.parametrize("spec", [s for s in (*BASIC_FIELDS, *ADVANCED_FIELDS) if s.kind == "int"])
def test_int_minimums_match_the_schema(spec):
    expected = _schema_minimum(spec.name)

    assert spec.minimum == (expected if expected is not None else spec.minimum)
    assert spec.maximum is not None and spec.maximum > (spec.minimum or 0)


def test_choice_fields_use_the_shared_choice_lists():
    by_name = {spec.name: spec for spec in (*BASIC_FIELDS, *ADVANCED_FIELDS)}

    assert by_name["base_model"].choices == BASE_MODELS
    assert by_name["optimizer_type"].choices == OPTIMIZERS


def test_caption_dropout_is_a_fraction_with_citron_slider_step():
    spec = next(s for s in BASIC_FIELDS if s.name == "caption_dropout")

    assert (spec.kind, spec.minimum, spec.maximum, spec.step) == ("fraction", 0.0, 1.0, 0.05)


def test_field_label_uses_citron_label_and_falls_back_to_name():
    assert field_label("network_dim") == "Network Dim"
    assert field_label("last_train_config") == "last_train_config"


def test_validation_error_is_described_with_labels():
    with pytest.raises(ValidationError) as caught:
        TrainingSettings(network_dim=0, learning_rate="abc")

    lines = describe_validation_error(caught.value)

    assert len(lines) == 2
    assert lines[0].startswith("ERROR: Network Dim: ")
    assert lines[1].startswith("ERROR: Learning Rate: ")


def test_validator_messages_drop_the_pydantic_prefix():
    with pytest.raises(ValidationError) as caught:
        TrainingSettings(project_name="../x")

    assert describe_validation_error(caught.value) == (
        "ERROR: Project Name: use only letters, digits, '.', '_' and '-'",
    )
