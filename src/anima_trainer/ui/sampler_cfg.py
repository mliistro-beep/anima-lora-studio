"""In a form with `sampler` and `cfg_scale` fields: a CFG still at the old sampler's default
follows the new sampler's default (preview.settings.cfg_after_sampler_change)."""

from PySide6.QtWidgets import QComboBox

from anima_trainer.preview.settings import cfg_after_sampler_change
from anima_trainer.ui.widgets.settings_form import SettingsForm


def link_cfg_to_sampler(form: SettingsForm) -> None:
    combo = form.field_widget("sampler")
    if not isinstance(combo, QComboBox):
        raise TypeError("the sampler field must be a combo box")
    previous = [combo.currentText()]  # the slot below sees the value before the change

    def changed(new: str) -> None:
        try:
            cfg = float(str(form.value("cfg_scale")))  # decimal fields hold text
        except ValueError:  # a value being typed that is not a number yet stays as it is
            cfg = None
        if cfg is not None:
            form.set_value("cfg_scale", cfg_after_sampler_change(previous[0], new, cfg))
        previous[0] = new

    combo.currentTextChanged.connect(changed)
