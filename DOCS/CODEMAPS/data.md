<!-- Generated: 2026-10-20 | Files scanned: 50 (19 src, 31 tests) | Token estimate: ~322 -->
# Data

No database. All state is files inside the Citron root, shared with Citron's web UI.

| Data | Path | Written by | Format |
|------|------|------------|--------|
| Settings | <citron>/config.json | settings/store.py save_settings | JSON, 35 keys, Citron order, indent 2 |
| Training config | <citron>/configs/<proj>_training_<ts>.toml | toml_builder.write_configs | TOML (toml lib, byte-identical to Citron) |
| Dataset config | <citron>/configs/<proj>_dataset_<ts>.toml | toml_builder.write_configs | TOML |
| Training log | <citron>/logs/<proj>_<ts>.log | training/runner.py | "Command:", "Started:", blank, raw lines |
| DiT model | <citron>/models/anima/dit/<base>.safetensors | models/downloader.py | via .part, renamed when complete |
| Checkpoints | <output_dir>/<proj>-NNNNNN.safetensors, <proj>.safetensors | sd-scripts | read by log_parser (CheckpointSaved) |
| Samples (M4) | <output_dir>/sample/*.png | sd-scripts | planned gallery source |

<ts> = %Y-%m-%d_%H%M%S (toml_builder.FILE_TIMESTAMP_FORMAT)

## Test fixtures
tests/fixtures/citron/: config.json, golden_training.toml, golden_dataset.toml,
training_log_excerpt.log (real Citron run); tests/fixtures/fake_accelerate.py
