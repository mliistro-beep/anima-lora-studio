"""Filesystem layout of the app's data folder (venv, sd-scripts, models, configs, logs).

Same layout as a Citron installation, so generated TOMLs and logs look exactly like Citron's.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from anima_trainer.models.catalog import QWEN3_FILENAME, VAE_FILENAME, dit_filename

# Relative on purpose, like Citron: accelerate runs with cwd = the data folder.
ACCELERATE_CONFIG = Path("app_configs/accelerate_gpu.yaml")


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    platform: str = field(default=os.name)

    @property
    def config_file(self) -> Path:
        return self.root / "config.json"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def models_dir(self) -> Path:
        return self.root / "models" / "anima"

    @property
    def sd_scripts_dir(self) -> Path:
        return self.root / "sd-scripts"

    @property
    def qwen3_model(self) -> Path:
        return self.models_dir / "text_encoder" / QWEN3_FILENAME

    @property
    def vae_model(self) -> Path:
        return self.models_dir / "vae" / VAE_FILENAME

    @property
    def train_script(self) -> Path:
        return self.sd_scripts_dir / "anima_train_network.py"

    @property
    def inference_script(self) -> Path:
        return self.sd_scripts_dir / "anima_minimal_inference.py"

    @property
    def accelerate_config(self) -> Path:
        return ACCELERATE_CONFIG

    @property
    def accelerate_config_file(self) -> Path:
        return self.root / ACCELERATE_CONFIG

    @property
    def venv_dir(self) -> Path:
        return self.root / ".venv"

    @property
    def venv_python(self) -> Path:
        return self._venv_bin("python")

    @property
    def venv_accelerate(self) -> Path:
        return self._venv_bin("accelerate")

    def resolve(self, user_path: str) -> Path:
        """Resolve a user-entered path the way sd-scripts will (its cwd is the data folder)."""
        path = Path(user_path).expanduser()
        return path if path.is_absolute() else self.root / path

    def required_models(self) -> tuple[tuple[str, Path], ...]:
        """Models that must already exist; the DiT is downloaded on demand instead."""
        return (("Qwen3", self.qwen3_model), ("VAE", self.vae_model))

    def dit_model(self, base_model: str) -> Path:
        return self.models_dir / "dit" / dit_filename(base_model)

    def missing_required_models(self) -> tuple[str, ...]:
        return tuple(label for label, path in self.required_models() if not path.exists())

    def _venv_bin(self, name: str) -> Path:
        if self.platform == "nt":
            return self.venv_dir / "Scripts" / f"{name}.exe"
        return self.venv_dir / "bin" / name
