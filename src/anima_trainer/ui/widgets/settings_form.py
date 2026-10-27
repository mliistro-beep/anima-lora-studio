"""A form built from FieldSpecs: one group box per spec group, one row per setting."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel
from PySide6.QtCore import QLocale
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from anima_trainer.settings.field_specs import INT_MAX, INT_MIN, FieldKind, FieldSpec
from anima_trainer.system.gpu_choice import choice_for_index, gpu_index_from_choice
from anima_trainer.ui.widgets.path_picker import PathPicker

FRACTION_DECIMALS = 2
DEFAULT_FRACTION_STEP = 0.05

# Floats stay text until the schema converts them; see field_specs.
FieldValue = str | int | float | bool


@dataclass(frozen=True)
class _Field:
    widget: QWidget
    read: Callable[[], FieldValue]
    write: Callable[[object], None]


def _text(spec: FieldSpec) -> _Field:
    edit = QLineEdit()
    edit.setPlaceholderText(spec.placeholder)
    return _Field(edit, edit.text, lambda value: edit.setText(str(value)))


def _float(spec: FieldSpec) -> _Field:
    # Returned as text: the schema converts it and reports a readable error if invalid.
    edit = QLineEdit()
    return _Field(edit, lambda: edit.text().strip(), lambda value: edit.setText(str(value)))


def _folder(spec: FieldSpec) -> _Field:
    picker = PathPicker(spec.label, "folder", placeholder=spec.placeholder)
    return _Field(picker, picker.path, lambda value: picker.set_path(str(value)))


def _choice(spec: FieldSpec) -> _Field:
    combo = QComboBox()
    combo.addItems(list(spec.choices))
    return _Field(combo, combo.currentText, lambda value: combo.setCurrentText(str(value)))


def _gpu(_spec: FieldSpec) -> _Field:
    combo = QComboBox()

    def write(value: object) -> None:
        choices = tuple(combo.itemText(i) for i in range(combo.count()))
        _select_gpu(combo, choices or (str(value),), str(value))

    return _Field(combo, lambda: gpu_index_from_choice(combo.currentText()), write)


def _int(spec: FieldSpec) -> _Field:
    spin = QSpinBox()
    low = INT_MIN if spec.minimum is None else int(spec.minimum)
    high = INT_MAX if spec.maximum is None else int(spec.maximum)
    spin.setRange(low, high)
    if spec.step is not None:
        spin.setSingleStep(int(spec.step))
    return _Field(spin, spin.value, lambda value: spin.setValue(int(str(value))))


def _fraction(spec: FieldSpec) -> _Field:
    spin = QDoubleSpinBox()
    spin.setLocale(QLocale.c())  # always "0.10", never "0,10"
    spin.setDecimals(FRACTION_DECIMALS)
    spin.setRange(
        0.0 if spec.minimum is None else spec.minimum,
        1.0 if spec.maximum is None else spec.maximum,
    )
    spin.setSingleStep(DEFAULT_FRACTION_STEP if spec.step is None else spec.step)
    return _Field(spin, spin.value, lambda value: spin.setValue(float(str(value))))


def _bool(_spec: FieldSpec) -> _Field:
    box = QCheckBox()
    return _Field(box, box.isChecked, lambda value: box.setChecked(bool(value)))


_BUILDERS: Mapping[FieldKind, Callable[[FieldSpec], _Field]] = {
    "text": _text,
    "float": _float,
    "folder": _folder,
    "choice": _choice,
    "gpu": _gpu,
    "int": _int,
    "fraction": _fraction,
    "bool": _bool,
}


def _select_gpu(combo: QComboBox, choices: tuple[str, ...], index: str) -> None:
    combo.clear()
    combo.addItems(list(choices))
    combo.setCurrentText(choice_for_index(choices, index))


class SettingsForm(QWidget):
    def __init__(self, specs: Sequence[FieldSpec], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        forms: dict[str, QFormLayout] = {}
        fields: dict[str, _Field] = {}
        for spec in specs:
            if spec.group not in forms:
                box = QGroupBox(spec.group)
                forms[spec.group] = QFormLayout(box)
                layout.addWidget(box)
            field = _BUILDERS[spec.kind](spec)
            field.widget.setAccessibleName(spec.label)
            field.widget.setToolTip(spec.tooltip)
            forms[spec.group].addRow(spec.label, field.widget)
            fields[spec.name] = field
        layout.addStretch(1)
        self._fields: Mapping[str, _Field] = fields
        gpu = fields.get("gpu_index")
        self._gpu_combo = gpu.widget if gpu and isinstance(gpu.widget, QComboBox) else None

    def field_widget(self, name: str) -> QWidget:
        return self._fields[name].widget

    def value(self, name: str) -> FieldValue:
        return self._fields[name].read()

    def values(self) -> dict[str, FieldValue]:
        return {name: field.read() for name, field in self._fields.items()}

    def set_values(self, settings: BaseModel) -> None:
        for name, field in self._fields.items():
            field.write(getattr(settings, name))

    def set_gpu_choices(self, choices: tuple[str, ...], index: str) -> None:
        if self._gpu_combo is not None:
            _select_gpu(self._gpu_combo, choices, index)
