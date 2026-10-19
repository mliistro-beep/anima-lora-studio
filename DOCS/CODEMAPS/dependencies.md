<!-- Generated: 2026-10-20 | Files scanned: 50 (19 src, 31 tests) | Token estimate: ~267 -->
# Dependencies

## Runtime (pyproject.toml)
PySide6>=6.8 (6.12 installed)  GUI, QProcess, QThread, signals
pydantic>=2.7,<3 (2.14)        settings validation
toml>=0.10.2                   TOML writer, same lib as Citron for identical output

## Dev
pytest, pytest-qt, pytest-cov, ruff, mypy (+pydantic plugin), types-toml, bandit, pip-audit

## External, used through subprocesses (not imported)
Citron install (<citron>): .venv (Python 3.14, torch, accelerate), sd-scripts
  accelerate launch ... sd-scripts/anima_train_network.py      training
  sd-scripts/anima_minimal_inference.py                        inference (M4)
  .venv/bin/python -c "import torch..."                        GPU list
nvidia-smi                                                     GPU list fallback
dmesg -T                                                       OOM hint (often not permitted)

## External services
Hugging Face circlestone-labs/Anima (https only)  DiT downloads, models/catalog.py
