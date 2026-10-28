<!-- Generated: 2026-10-28 | Files scanned: 189 (93 src, 96 tests) | Token estimate: ~280 -->
# Dependencies

## Runtime (pyproject.toml)
PySide6>=6.8 (6.12 installed)  GUI, QProcess, QThread, signals
pydantic>=2.7,<3 (2.14)        settings validation
toml>=0.10.2                   TOML writer, same lib as Citron for identical output

## Dev
pytest, pytest-qt, pytest-cov, ruff, mypy (+pydantic plugin), types-toml, bandit, pip-audit

## External, used through subprocesses (not imported)
Training runtime in the data folder (<data>): .venv (torch, accelerate), sd-scripts
  (installed by the Settings tab: venv, pinned sd-scripts commit, its requirements + torchvision;
  Citron is never used)
  accelerate launch ... sd-scripts/anima_train_network.py      training
  runtime_scripts/train_with_previews.py (app's, in place of the training script)  training with previews
  runtime_scripts/generate.py (app's, imports sd-scripts' anima_minimal_inference) generator
  .venv/bin/python -c "import torch..."                        GPU list
git                                                            fetches sd-scripts (installer)
nvidia-smi                                                     GPU list fallback
dmesg -T                                                       OOM hint (often not permitted)

## External services
Hugging Face circlestone-labs/Anima (https only)  DiT, Qwen3 and VAE downloads; expected SHA-256 and
                                                  sizes in models/catalog.py come from its LFS metadata
GitHub kohya-ss/sd-scripts (https, via git)       pinned commit fetched by the installer
