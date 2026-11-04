"""Filesystem layout of the app's data folder (venv, sd-scripts, models, configs, logs).

Same layout as a Citron installation, so generated TOMLs and logs look exactly like Citron's.
A model file the user linked replaces the data folder copy while it is unchanged; otherwise the
data folder path is used again, so a download never writes into the user's own folder.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from anima_trainer.models.catalog import QWEN3_FILENAME, VAE_FILENAME, dit_filename
from anima_trainer.models.linked_models import LinkedModel, is_unchanged

DIT_SUFFIX = ".safetensors"

# Relative on purpose, like Citron: accelerate runs with cwd = the data folder.
ACCELERATE_CONFIG = Path("app_configs/accelerate_gpu.yaml")


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    platform: str = field(default=os.name)
    # Linked model files by model key (base model name, or file name for Qwen3 and the VAE).
    links: Mapping[str, LinkedModel] = field(
        default_factory=lambda: MappingProxyType({}), hash=False
    )

    def with_links(self, links: Mapping[str, LinkedModel]) -> "RuntimePaths":
        return replace(self, links=MappingProxyType(dict(links)))

    def is_linked(self, key: str) -> bool:
        """True while the linked file for `key` is in use."""
        link = self.links.get(key)
        return link is not None and is_unchanged(link)

    def has_broken_link(self, key: str) -> bool:
        """True when a file was linked for `key` but was moved or changed since."""
        return key in self.links and not self.is_linked(key)

    @property
    def config_file(self) -> Path:
        return self.root / "config.json"

    @property
    def preview_settings_file(self) -> Path:
        return self.root / "preview.json"

    @property
    def generator_settings_file(self) -> Path:
        return self.root / "generator.json"

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
        return self._model(QWEN3_FILENAME, self.models_dir / "text_encoder" / QWEN3_FILENAME)

    @property
    def vae_model(self) -> Path:
        return self._model(VAE_FILENAME, self.models_dir / "vae" / VAE_FILENAME)

    @property
    def train_script(self) -> Path:
        return self.sd_scripts_dir / "anima_train_network.py"

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
        filename = dit_filename(base_model)
        return self._model(filename.removesuffix(DIT_SUFFIX), self.models_dir / "dit" / filename)

    def missing_required_models(self) -> tuple[str, ...]:
        return tuple(label for label, path in self.required_models() if not path.exists())

    def _model(self, key: str, in_data_folder: Path) -> Path:
        return self.links[key].path if self.is_linked(key) else in_data_folder

    def _venv_bin(self, name: str) -> Path:
        if self.platform == "nt":
            return self.venv_dir / "Scripts" / f"{name}.exe"
        return self.venv_dir / "bin" / name
