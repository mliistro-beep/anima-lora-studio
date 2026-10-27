# Anima LoRA Trainer (desktop)

Desktop (PySide6) version of Citron's Anima LoRA Trainer UI, with the same settings and
identical TOML configs. It does not need Citron: it keeps its own training runtime (venv,
sd-scripts, models) in its data folder, can use model files you already have (accepted
after a SHA-256 check), and will add epoch previews and a simple image generator.

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
scripts/dev-run.sh --setup            # first time: creates .venv with the dev tools, then starts the app
scripts/dev-run.sh                    # start the app from the checkout
scripts/dev-run.sh --scratch          # same, with data folder and settings in ~/anima-dev-scratch
QT_QPA_PLATFORM=offscreen .venv/bin/pytest
```

`scripts/dev-run.sh` is for development only; the release launcher is planned in M6.

<!-- AUTO-GENERATED: commands (source: pyproject.toml, scripts/) -->
| Command | Description |
|---------|-------------|
| `.venv/bin/anima-trainer` | Start the desktop app (`anima_trainer.main:main`) |
| `scripts/dev-run.sh` | Development only: start the app from the checkout (`--setup` creates `.venv`, `--scratch [DIR]` uses scratch XDG folders, default `~/anima-dev-scratch`) |
| `.venv/bin/pytest` | Run all tests (`tests/`), markers `integration` and `e2e` available |
| `.venv/bin/pytest --cov` | Run tests with coverage; fails under 80% (UI and `main.py` excluded) |
| `.venv/bin/ruff check src tests` | Lint (rules E, F, W, I, B, UP, SIM, T20; line length 100) |
| `.venv/bin/ruff format src tests` | Format the code |
| `.venv/bin/mypy` | Strict type check of `src/` with the pydantic plugin |
| `.venv/bin/bandit -c pyproject.toml -r src` | Security static analysis (tests excluded) |
| `.venv/bin/pip-audit` | Check installed dependencies for known vulnerabilities |
<!-- /AUTO-GENERATED -->

## License and credits

Copyright (C) 2026 Mauro Liistro. Licensed under the GNU General Public License, version 3
or later: see [LICENSE](LICENSE). The License and Credits buttons of the app show the same.

This app is based on [Citron Anima LoRA Trainer UI](https://github.com/citronlegacy/citron-anima-lora-trainer-ui)
by Citron Legacy (MIT): its features, settings and TOML configs come from Citron's work.
Training uses [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (Apache-2.0).
The models the app downloads keep their own licenses: the Anima models are for
non-commercial use only (CircleStone Labs Non-Commercial License and NVIDIA Open Model
License); Qwen3 0.6B and the Qwen-Image VAE are Apache-2.0. Details and the full notices:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
