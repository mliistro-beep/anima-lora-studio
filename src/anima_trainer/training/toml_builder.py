"""Training and dataset TOML generation, field-for-field identical to Citron.

Builders are pure (settings in, dict out); only write_configs() touches the filesystem.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import toml

from anima_trainer.preview.sample_prompts import (
    SampleRequest,
    build_sample_prompts,
    sample_config_keys,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings

FILE_TIMESTAMP_FORMAT = "%Y-%m-%d_%H%M%S"
CAPTION_EXTENSION = ".txt"
MAX_TOKEN_LENGTH = 512


@dataclass(frozen=True)
class GeneratedConfigs:
    training: Path
    dataset: Path
    samples: Path | None = None  # preview prompts, only when previews are on


def build_training_config(
    settings: TrainingSettings,
    paths: RuntimePaths,
    today: date,
    single_process_loader: bool = False,
    sample_keys: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Citron's training config; single_process_loader adds the Python 3.14 workaround and
    sample_keys the previews (without them the config is exactly Citron's)."""
    s = settings
    config = {
        "pretrained_model_name_or_path": str(paths.dit_model(s.base_model)),
        "qwen3": str(paths.qwen3_model),
        "vae": str(paths.vae_model),
        "network_module": "networks.lora_anima",
        "network_dim": s.network_dim,
        "network_alpha": s.network_alpha,
        "network_train_unet_only": True,
        "learning_rate": s.learning_rate,
        "optimizer_type": s.optimizer_type,
        "optimizer_args": ["weight_decay=0.1", "betas=[0.9, 0.99]"],
        "lr_scheduler": s.lr_scheduler,
        "lr_scheduler_num_cycles": s.lr_scheduler_num_cycles,
        "lr_warmup_steps": s.lr_warmup_steps,
        "max_train_epochs": s.max_train_epochs,
        "train_batch_size": s.train_batch_size,
        "gradient_accumulation_steps": s.gradient_accumulation_steps,
        "max_grad_norm": s.max_grad_norm,
        "seed": s.seed,
        "timestep_sampling": s.timestep_sampling,
        "discrete_flow_shift": s.discrete_flow_shift,
        "qwen3_max_token_length": MAX_TOKEN_LENGTH,
        "t5_max_token_length": MAX_TOKEN_LENGTH,
        "mixed_precision": s.mixed_precision,
        "gradient_checkpointing": s.gradient_checkpointing,
        "cache_latents": s.cache_latents,
        "cache_text_encoder_outputs": s.cache_text_encoder_outputs,
        "vae_chunk_size": s.vae_chunk_size,
        "vae_disable_cache": s.vae_disable_cache,
        "output_dir": s.output_directory,
        "output_name": s.project_name,
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "save_every_n_epochs": s.save_every_n_epochs,
        "save_last_n_epochs": s.save_last_n_epochs,
        "shuffle_caption": False,
        "caption_extension": CAPTION_EXTENSION,
        "noise_offset": s.noise_offset,
        "multires_noise_discount": s.multires_noise_discount,
        "training_comment": f"Anima LoRA - {today.isoformat()}",
        **(sample_keys or {}),
    }
    if single_process_loader:
        return {**config, "max_data_loader_n_workers": 0}
    return config


def build_dataset_config(settings: TrainingSettings) -> dict[str, Any]:
    return {
        "general": {
            "resolution": settings.resolution,
            "enable_bucket": True,
            "bucket_no_upscale": False,
            "bucket_reso_steps": 64,
            "min_bucket_reso": 256,
            "max_bucket_reso": 4096,
        },
        "datasets": [
            {
                "resolution": settings.resolution,
                "subsets": [
                    {
                        "num_repeats": settings.repeats,
                        "image_dir": settings.image_directory,
                        "caption_extension": CAPTION_EXTENSION,
                        "caption_dropout_rate": settings.caption_dropout,
                    }
                ],
            }
        ],
    }


def render_toml(config: dict[str, Any]) -> str:
    """Serialize with the same `toml` library Citron uses, so output matches byte for byte."""
    return toml.dumps(config)


def config_file_name(project_name: str, kind: str, now: datetime) -> str:
    return f"{project_name}_{kind}_{now.strftime(FILE_TIMESTAMP_FORMAT)}.toml"


def write_configs(
    settings: TrainingSettings,
    paths: RuntimePaths,
    now: datetime,
    single_process_loader: bool = False,
    samples: SampleRequest | None = None,
) -> GeneratedConfigs:
    """Write both TOML files (and the preview prompts, if any) into the configs dir of the
    data folder and create the output dir."""
    Path(settings.output_directory).mkdir(parents=True, exist_ok=True)
    paths.configs_dir.mkdir(parents=True, exist_ok=True)
    training = paths.configs_dir / config_file_name(settings.project_name, "training", now)
    dataset = paths.configs_dir / config_file_name(settings.project_name, "dataset", now)
    prompts_file, sample_keys = _write_sample_prompts(settings, paths, now, samples)
    config = build_training_config(settings, paths, now.date(), single_process_loader, sample_keys)
    training.write_text(render_toml(config), encoding="utf-8")
    dataset.write_text(render_toml(build_dataset_config(settings)), encoding="utf-8")
    return GeneratedConfigs(training, dataset, prompts_file)


def _write_sample_prompts(
    settings: TrainingSettings, paths: RuntimePaths, now: datetime, samples: SampleRequest | None
) -> tuple[Path | None, dict[str, Any] | None]:
    if samples is None:
        return None, None
    name = config_file_name(settings.project_name, "samples", now).removesuffix(".toml")
    prompts_file = paths.configs_dir / f"{name}.json"
    prompts = build_sample_prompts(samples.settings, samples.size)
    prompts_file.write_text(json.dumps(list(prompts), indent=2), encoding="utf-8")
    return prompts_file, sample_config_keys(samples.settings, prompts_file)
