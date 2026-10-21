<!-- Generated: 2026-10-21 | Files scanned: 97 (47 src, 50 tests) | Token estimate: ~330 -->
# Data

No database. All state is files inside the app's data folder (<data>, default
~/.local/share/anima-trainer), plus the data folder location in QSettings
(~/.config/anima-trainer/anima-trainer.conf, key runtime/data_dir).

| Data | Path | Written by | Format |
|------|------|------------|--------|
| Settings | <data>/config.json | settings/store.py save_settings | JSON, 35 keys, Citron order, indent 2 |
| Training config | <data>/configs/<proj>_training_<ts>.toml | toml_builder.write_configs | TOML (toml lib, byte-identical to Citron) |
| Dataset config | <data>/configs/<proj>_dataset_<ts>.toml | toml_builder.write_configs | TOML |
| Training log | <data>/logs/<proj>_<ts>.log | training/runner.py | "Command:", "Started:", blank, raw lines |
| DiT model | <data>/models/anima/dit/<base>.safetensors | models/downloader.py | via .part, renamed when complete |
| Install state | <data>/.install/{sd-scripts,requirements,verified}.commit | installer (M4) | pinned sd-scripts commit, one line |
| Checkpoints | <output_dir>/<proj>-NNNNNN.safetensors, <proj>.safetensors | sd-scripts | read by log_parser (CheckpointSaved) |
| Samples (M5) | <output_dir>/sample/*.png | sd-scripts | planned gallery source |

<ts> = %Y-%m-%d_%H%M%S (toml_builder.FILE_TIMESTAMP_FORMAT)

## Test fixtures
tests/fixtures/citron/: config.json, golden_training.toml, golden_dataset.toml,
training_log_excerpt.log (real Citron run, used as golden reference only); tests/fixtures/fake_accelerate.py
