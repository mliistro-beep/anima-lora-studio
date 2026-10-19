# Anima LoRA Trainer (desktop)

Desktop (PySide6) version of Citron's Anima LoRA Trainer UI. It reuses an existing
Citron installation (venv, sd-scripts, models) and adds epoch previews and a simple
image generator.

Status and the per-component checklist: [DOCS/milestones.md](DOCS/milestones.md).
Contributor setup: [DOCS/CONTRIBUTING.md](DOCS/CONTRIBUTING.md).
Operations and troubleshooting: [DOCS/RUNBOOK.md](DOCS/RUNBOOK.md).
Architecture codemaps: [DOCS/CODEMAPS/](DOCS/CODEMAPS/architecture.md).

## Requirements

- Python 3.12 or newer (developed on 3.14)
- An existing Citron installation, for example `~/citron-anima-lora-trainer-ui`, with its
  `.venv`, `sd-scripts` and models under `models/anima/`
- An NVIDIA GPU for training (Citron's defaults target about 6 GB of VRAM)

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
QT_QPA_PLATFORM=offscreen .venv/bin/pytest
```

<!-- AUTO-GENERATED: commands (source: pyproject.toml) -->
| Command | Description |
|---------|-------------|
| `.venv/bin/anima-trainer` | Start the desktop app (`anima_trainer.main:main`) |
| `.venv/bin/pytest` | Run all tests (`tests/`), markers `integration` and `e2e` available |
| `.venv/bin/pytest --cov` | Run tests with coverage; fails under 80% (UI and `main.py` excluded) |
| `.venv/bin/ruff check src tests` | Lint (rules E, F, W, I, B, UP, SIM, T20; line length 100) |
| `.venv/bin/ruff format src tests` | Format the code |
| `.venv/bin/mypy` | Strict type check of `src/` with the pydantic plugin |
| `.venv/bin/bandit -c pyproject.toml -r src` | Security static analysis (tests excluded) |
| `.venv/bin/pip-audit` | Check installed dependencies for known vulnerabilities |
<!-- /AUTO-GENERATED -->
