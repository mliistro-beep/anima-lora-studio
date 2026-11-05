# Anima LoRA Studio

LoRA trainer for the Anima image model: a desktop (PySide6) version of Citron's Anima LoRA
Trainer UI, with the same settings and identical TOML configs. It does not need Citron: it
keeps its own training runtime (venv, sd-scripts, models) in its data folder, can use model
files you already have (accepted after a SHA-256 check), shows epoch previews during training
and has a simple image generator to try or compare the checkpoints.

Version 0.2.0, Linux. Changes per version: [CHANGELOG.md](CHANGELOG.md).
Status and the per-component checklist: [DOCS/milestones.md](DOCS/milestones.md).
Contributor setup: [DOCS/CONTRIBUTING.md](DOCS/CONTRIBUTING.md).
Operations and troubleshooting: [DOCS/RUNBOOK.md](DOCS/RUNBOOK.md).
Swap and earlyoom before the first training: [DOCS/SYSTEM_SETUP.md](DOCS/SYSTEM_SETUP.md).
Architecture codemaps: [DOCS/CODEMAPS/](DOCS/CODEMAPS/architecture.md).

## Requirements

- Python 3.12 or newer (developed on 3.14)
- An internet connection for the first install of the training runtime and models
- git, used by the Install button of the Settings tab to fetch sd-scripts
- An NVIDIA GPU for training (the default settings use about 6 GB of VRAM)
- 16 GB of system RAM, plus 8-16 GB of swap and an out-of-memory daemon (earlyoom or
  systemd-oomd). Without them, running out of RAM during a training or a generation can
  freeze the whole computer until a hard reset. Setup steps:
  [DOCS/SYSTEM_SETUP.md](DOCS/SYSTEM_SETUP.md).

## Install

Check the requirements above first, in particular the swap and the out-of-memory daemon
([DOCS/SYSTEM_SETUP.md](DOCS/SYSTEM_SETUP.md)). On Debian/Ubuntu the system packages are
`sudo apt install python3 python3-venv git`.

```bash
git clone https://github.com/mliistro-beep/anima-lora-studio.git
cd anima-lora-studio
./run_linux.sh                    # first start: creates .venv and installs the app (needs the network)
./run_linux.sh --install-desktop  # optional: add Anima LoRA Studio to the applications menu
```

Then, in the app's Settings tab, press **Install** to set up the training runtime (torch and
sd-scripts, several GB) and download the models, or point it to model files you already have
with **Use file...**. The app keeps all of it in its data folder,
`~/.local/share/anima-trainer`.

Later starts reinstall only when `pyproject.toml` changed (for example after `git pull`).
Do the first start from a terminal or with `--install-desktop`: from the menu it shows only a
notification while it installs, and its output is in `~/.local/state/anima-trainer/launcher.log`.
`./run_linux.sh --uninstall-desktop` removes the menu entry. More in
[DOCS/RUNBOOK.md](DOCS/RUNBOOK.md).

## Train from the terminal

After the runtime and the models are set up in the app, you can train without a window, also over
SSH on a machine with no display. Run these from the checkout (or with the full path of
`.venv/bin/anima-trainer`):

```bash
.venv/bin/anima-trainer tui                     # choose, approve or change a summary, follow progress
.venv/bin/anima-trainer train --list-presets    # the presets available
.venv/bin/anima-trainer train --preset Person --project my_lora --dataset ~/images \
    --output ~/loras --epochs 10 --repeats 10   # add --yes to skip the confirmation
```

In the interface, Browse picks a folder without writing its path. Project, folders, epochs and repeats are chosen for each training; the resolution starts at 768.
Epochs depend on how large the dataset is: the preset only suggests a value. **For remote use, run
it inside `tmux`**: quitting the interface or losing the connection without tmux stops the training.
Details, your own presets and exit codes are in [DOCS/RUNBOOK.md](DOCS/RUNBOOK.md).

## Development

```bash
scripts/dev-run.sh --setup            # first time: creates .venv with the dev tools, then starts the app
scripts/dev-run.sh                    # start the app from the checkout
scripts/dev-run.sh --scratch          # same, with data folder and settings in ~/anima-dev-scratch
QT_QPA_PLATFORM=offscreen .venv/bin/pytest
```

`scripts/dev-run.sh` is for development only; users start the app with `./run_linux.sh`.
To try a development build with `./run_linux.sh`, point `XDG_DATA_HOME` and `XDG_CONFIG_HOME`
at scratch folders first, or it uses your real data folder.

<!-- AUTO-GENERATED: commands (source: pyproject.toml, scripts/) -->
| Command | Description |
|---------|-------------|
| `./run_linux.sh` | Start the app, creating or updating `.venv` when needed; `--install-desktop` / `--uninstall-desktop` add or remove the menu entry |
| `.venv/bin/anima-trainer` | Start the desktop app (`anima_trainer.cli.entry:main`, then `anima_trainer.main:main`) |
| `.venv/bin/anima-trainer train --help` | Train from a preset, without a window (`--list-presets` shows them) |
| `scripts/dev-run.sh` | Development only: start the app from the checkout (`--setup` creates `.venv`, `--scratch [DIR]` uses scratch XDG folders, default `~/anima-dev-scratch`) |
| `.venv/bin/pytest` | Run all tests (`tests/`), markers `integration`, `e2e` and `runtime` available (`runtime` tests are skipped unless `ANIMA_TRAINER_TEST_RUNTIME` is set) |
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
