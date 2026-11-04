"""Turn a preset into training settings."""

from anima_trainer.presets.preset_file import KEPT_FIELDS, Preset
from anima_trainer.settings.schema import TrainingSettings


def apply_preset(preset: Preset, saved: TrainingSettings) -> TrainingSettings:
    """New settings from the defaults and the preset, so a preset always gives the same training;
    only what a preset cannot hold comes from `saved`."""
    kept = {field: getattr(saved, field) for field in KEPT_FIELDS}
    return TrainingSettings().with_changes(**preset.values, **kept)
