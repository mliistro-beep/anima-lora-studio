# Contributing

How to set up, test and submit changes for the Anima LoRA Studio desktop app.
Project rules come from [CLAUDE.md](../CLAUDE.md); progress is tracked in
[milestones.md](milestones.md).

## Development environment

Prerequisites:

- Python 3.12 or newer (the project is developed on 3.14)
- For real training or inference, the training runtime in the app's data folder (see
  [RUNBOOK.md](RUNBOOK.md)). Unit and integration tests do not need it: they use fixtures
  and a fake `accelerate`. Never point the app at a Citron installation: it would write
  `config.json`, `configs/` and `logs/` there.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

or `scripts/dev-run.sh --setup`, which does the same when `.venv` is missing and then starts
the app. The script is for development only: users start the app with `./run_linux.sh`
(see [RUNBOOK.md](RUNBOOK.md)), which installs without the dev tools and reuses this `.venv`
as it is. Do not try a development build with `./run_linux.sh` unless the XDG folders point
at scratch folders: it would use your real data folder.

The app's venv holds only the GUI and dev tools. Training runs in the training venv of
the data folder (torch, accelerate, sd-scripts), started as a separate process. To try the
real app without touching your own settings, start it with `XDG_CONFIG_HOME` and
`XDG_DATA_HOME` pointed at scratch folders: `scripts/dev-run.sh --scratch [DIR]` does it
(default `~/anima-dev-scratch`; not under `/tmp`, the runtime takes over 10 GB).

<!-- AUTO-GENERATED: commands (source: pyproject.toml, scripts/) -->
## Commands

| Command | Description |
|---------|-------------|
| `.venv/bin/anima-trainer` | Start the desktop app (`anima_trainer.main:main`) |
| `./run_linux.sh` | Release launcher: creates or updates `.venv` when needed and starts the app; `--install-desktop` / `--uninstall-desktop` add or remove the menu entry |
| `scripts/dev-run.sh` | Development only: start the app from the checkout (`--setup` creates `.venv`, `--scratch [DIR]` uses scratch XDG folders, default `~/anima-dev-scratch`) |
| `.venv/bin/pytest` | Run all tests (`tests/`), markers `integration`, `e2e` and `runtime` available (`runtime` tests are skipped unless `ANIMA_TRAINER_TEST_RUNTIME` is set) |
| `.venv/bin/pytest --cov` | Run tests with coverage; fails under 80% (UI and `main.py` excluded) |
| `.venv/bin/ruff check src tests` | Lint (rules E, F, W, I, B, UP, SIM, T20; line length 100) |
| `.venv/bin/ruff format src tests` | Format the code |
| `.venv/bin/mypy` | Strict type check of `src/` with the pydantic plugin |
| `.venv/bin/bandit -c pyproject.toml -r src` | Security static analysis (tests excluded) |
| `.venv/bin/pip-audit` | Check installed dependencies for known vulnerabilities |
<!-- /AUTO-GENERATED -->

<!-- AUTO-GENERATED: environment (source: src/, tests/) -->
## Environment variables

The app reads no environment variables of its own; Qt reads the XDG ones for it. It sets
the others for the processes it launches (training, generator and install steps), and tests
use two:

| Variable | Set by | Value | Purpose |
|----------|--------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `training/command_builder.py`, `preview/inference_command.py` | selected GPU index | Same as Citron: limits the training and the generator to the chosen GPU |
| `PYTHONUNBUFFERED` | `training/command_builder.py`, `preview/inference_command.py`, `runtime/installer.py` | `1` | Streams sd-scripts, generator, pip and GPU check output line by line |
| `ANIMA_TRAINER_SD_SCRIPTS` | `training/command_builder.py` (with previews), `preview/inference_command.py` | the data folder's `sd-scripts` | Tells `train_with_previews.py` and `generate.py` where sd-scripts is |
| `PYTORCH_CUDA_ALLOC_CONF` | `preview/inference_command.py` | `expandable_segments:True` | Less GPU memory held between the generator's text encoder, DiT and VAE phases |
| `XDG_DATA_HOME` | you (optional) | a folder | Read by Qt (`QStandardPaths`): parent of the default data folder `anima-trainer` |
| `XDG_CONFIG_HOME` | you (optional) | a folder | Read by Qt (`QSettings`): where the app keeps its settings (the chosen data folder, the linked model files) |
| `XDG_STATE_HOME` | you (optional) | a folder | Read by `run_linux.sh`: parent of `anima-trainer/launcher.log` (default `~/.local/state`) |
| `QT_QPA_PLATFORM` | you, when running tests | `offscreen` | Lets Qt tests run without a display |
| `ANIMA_TRAINER_TEST_RUNTIME` | you, when running tests (optional) | an installed data folder, e.g. `~/anima-dev-scratch/share/anima-trainer` | Enables the `runtime` tests (`tests/runtime_scripts/`), which need torch; skipped otherwise |
<!-- /AUTO-GENERATED -->

## Tests

- Layout: `tests/unit/` (pure logic), `tests/integration/` (subprocesses, local HTTP
  server, Qt signals), `tests/e2e/` (UI driven with pytest-qt), fixtures in
  `tests/fixtures/`. `tests/runtime_helpers.py` `mark_installed(paths)` makes a data
  folder look fully installed (files, install markers, accelerate config).
- Run a subset: `.venv/bin/pytest -m integration`, `.venv/bin/pytest tests/unit`.
- Workflow is TDD: write the failing test first, then the code. Coverage must stay at
  80% or more.
- Golden tests in `tests/unit/training/test_toml_builder.py` compare generated TOML
  byte for byte with real Citron files in `tests/fixtures/citron/`. If Citron's output
  format changes, refresh those fixtures from a real Citron run, not by hand.
- `tests/fixtures/fake_accelerate.py` stands in for `accelerate launch` (replay a log,
  exit codes, crash, process groups, and `train`: a run that writes checkpoints and preview
  PNGs from the real training TOML). Extend it rather than spawning real training.
- `tests/integration/launcher/` runs `run_linux.sh` copied into a temporary checkout, with
  a fake `python3` and a `PATH` that holds only the fakes and a few system tools, so it never
  touches the real `.venv`, Python, pip or menu entries.
- Tests that wait on Qt signals must not rely on timing: throttle the fake server
  (`block_delay_s`) instead of hoping a race goes your way, and make sure every thread
  or process a test starts has stopped before it ends.

## Real GPU and memory experiments

Simulations and measurements with the real runtime (trainings, `generate.py`, VRAM probes)
can use more system RAM than expected. Before running them, set up swap and earlyoom
([SYSTEM_SETUP.md](SYSTEM_SETUP.md)), and start long experiments with a memory cap so a
mistake stops the experiment, not the machine:

```bash
systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=4G <command>
```

Write their logs to disk line by line (unbuffered), so they survive a crash. Do not put
scripts or outputs of experiments under `/tmp` when it is a `tmpfs`: it lives in RAM.

## Code style

- ruff for lint and format, strict mypy; both must be clean.
- Many small files organized by feature (`runtime/`, `settings/`, `dataset/`,
  `training/`, `models/`, `preview/`, `system/`, `workflow/`, `ui/`), 200-400 lines
  typical, 800 max.
- Long jobs off the UI thread go through `system/background_task.py` `BackgroundTask`
  (see `ModelDownloader` and `ModelFileChecker`) instead of a new QThread class.
- Immutable data: frozen dataclasses and pydantic models; return new objects instead
  of changing existing ones.
- Validate user input at the boundary (`settings/schema.py`), handle errors explicitly,
  no `print`, no emoji.

There are no repository pre-commit hooks; run the checks below by hand.

## Submitting changes

1. Work on a feature branch, never on `main`.
2. Use conventional commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`.
3. Before opening a PR, all of these must pass:
   - `QT_QPA_PLATFORM=offscreen .venv/bin/pytest --cov`
   - `.venv/bin/ruff check src tests` and `.venv/bin/ruff format --check src tests`
   - `.venv/bin/mypy`
   - `.venv/bin/bandit -c pyproject.toml -r src`
4. Update the component checklist in [milestones.md](milestones.md) when a component
   is added, changed or finished.
5. PRs need a review before merge.
