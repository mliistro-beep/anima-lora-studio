"""Default settings, copied verbatim from Citron's DEFAULTS (app.py) in the same key order."""

from types import MappingProxyType

DEFAULTS = MappingProxyType(
    {
        # Basic
        "project_name": "my_lora",
        "base_model": "anima-base-v1.0",
        "image_directory": "",
        "output_directory": "",
        "network_dim": 20,
        "network_alpha": 20,
        "learning_rate": 0.0001,
        "max_train_epochs": 10,
        "resolution": 768,
        "repeats": 10,
        "caption_dropout": 0.1,
        "gpu_index": "0",
        # Advanced
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "cosine_with_restarts",
        "lr_scheduler_num_cycles": 1,
        "lr_warmup_steps": 100,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "save_every_n_epochs": 1,
        "save_last_n_epochs": 4,
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "seed": 42,
        "noise_offset": 0.03,
        "multires_noise_discount": 0.3,
        "timestep_sampling": "sigmoid",
        "discrete_flow_shift": 1.0,
        "cache_latents": True,
        "cache_text_encoder_outputs": True,
        "vae_chunk_size": 64,
        "vae_disable_cache": True,
        "num_cpu_threads_per_process": 1,
        # Internal
        "last_train_config": "",
        "last_dataset_config": "",
    }
)
