<!-- Generated: 2026-10-26 | Files scanned: 129 (62 src, 67 tests) | Token estimate: ~330 -->
# Data

No database. All state is files inside the app's data folder (<data>, default
~/.local/share/anima-trainer), plus QSettings (~/.config/anima-trainer/anima-trainer.conf): the data folder (key
runtime/data_dir) and the linked model files (group linked_models, one JSON value per model
key: path, size, mtime_ns).

| Data | Path | Written by | Format |
|------|------|------------|--------|
| Settings | <data>/config.json | settings/store.py save_settings | JSON, 35 keys, Citron order, indent 2 |
| Training config | <data>/configs/<proj>_training_<ts>.toml | toml_builder.write_configs | TOML (toml lib, byte-identical to Citron) |
| Dataset config | <data>/configs/<proj>_dataset_<ts>.toml | toml_builder.write_configs | TOML |
| Training log | <data>/logs/<proj>_<ts>.log | training/runner.py | "Command:", "Started:", blank, raw lines |
| DiT model | <data>/models/anima/dit/<base>.safetensors | models/downloader.py | via .part, renamed when complete and size + SHA-256 match |
| Qwen3 / VAE | <data>/models/anima/{text_encoder,vae}/<file>.safetensors | models/downloader.py | via .part, checked like the DiT |
| Linked model | anywhere chosen by the user | never written | used in place while size and mtime are unchanged |
| Install state | <data>/.install/{venv,sd-scripts,requirements,verified}.commit | runtime/installer.py | one-line stamp: "ready" (venv), pinned commit, commit + "+torchvision" (requirements) |
| Install log | <data>/logs/install_<ts>.log | workflow/install_session.py | step titles and output |
| Checkpoints | <output_dir>/<proj>-NNNNNN.safetensors, <proj>.safetensors | sd-scripts | read by log_parser (CheckpointSaved) |
| Samples (M5) | <output_dir>/sample/*.png | sd-scripts | planned gallery source |

<ts> = %Y-%m-%d_%H%M%S (toml_builder.FILE_TIMESTAMP_FORMAT)

## Test fixtures
tests/fixtures/citron/: config.json, golden_training.toml, golden_dataset.toml,
training_log_excerpt.log (real Citron run, used as golden reference only); tests/fixtures/fake_accelerate.py
