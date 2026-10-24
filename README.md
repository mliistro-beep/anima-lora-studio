# Anima LoRA Trainer (desktop)

Desktop (PySide6) version of Citron's Anima LoRA Trainer UI, with the same settings and
identical TOML configs. It does not need Citron: it keeps its own training runtime (venv,
sd-scripts, models) in its data folder, and will add epoch previews and a simple image
generator.

Status and the per-component checklist: [DOCS/milestones.md](DOCS/milestones.md).
Contributor setup: [DOCS/CONTRIBUTING.md](DOCS/CONTRIBUTING.md).
Operations and troubleshooting: [DOCS/RUNBOOK.md](DOCS/RUNBOOK.md).
Architecture codemaps: [DOCS/CODEMAPS/](DOCS/CODEMAPS/architecture.md).

## Requirements

- Python 3.12 or newer (developed on 3.14)
- An internet connection for the first install of the training runtime and models
- git, used by the Install button of the Settings tab to fetch sd-scripts
- An NVIDIA GPU for training (the default settings use about 6 GB of VRAM)

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
