"""Filesystem layout of an existing Citron installation (venv, sd-scripts, models)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from anima_trainer.models.catalog import dit_filename

# Relative on purpose: Citron passes it as-is and runs accelerate with cwd = Citron root.
ACCELERATE_CONFIG = Path("app_configs/accelerate_gpu.yaml")


@dataclass(frozen=True)
class CitronPaths:
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
        return self.models_dir / "text_encoder" / "qwen_3_06b_base.safetensors"

    @property
    def vae_model(self) -> Path:
        return self.models_dir / "vae" / "qwen_image_vae.safetensors"

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
    def venv_python(self) -> Path:
        return self._venv_bin("python")

    @property
    def venv_accelerate(self) -> Path:
        return self._venv_bin("accelerate")

    def resolve(self, user_path: str) -> Path:
        """Resolve a user-entered path the way sd-scripts will (its cwd is the Citron root)."""
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
            return self.root / ".venv" / "Scripts" / f"{name}.exe"
        return self.root / ".venv" / "bin" / name
