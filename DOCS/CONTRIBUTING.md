# Contributing

How to set up, test and submit changes for the Anima LoRA Trainer desktop app.
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

The app's venv holds only the GUI and dev tools. Training runs in the training venv of
the data folder (torch, accelerate, sd-scripts), started as a separate process. To try the
real app without touching your own settings, start it with `XDG_CONFIG_HOME` and
`XDG_DATA_HOME` pointed at scratch folders.

<!-- AUTO-GENERATED: commands (source: pyproject.toml) -->
## Commands

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

<!-- AUTO-GENERATED: environment (source: src/, tests/) -->
## Environment variables

The app reads no environment variables of its own; Qt reads the XDG ones for it. It sets
the others for the processes it launches (training and install steps), and tests use one
for Qt:

| Variable | Set by | Value | Purpose |
|----------|--------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `training/command_builder.py` | selected GPU index | Same as Citron: limits the training to the chosen GPU |
| `PYTHONUNBUFFERED` | `training/command_builder.py`, `runtime/installer.py` | `1` | Streams sd-scripts, pip and GPU check output line by line |
| `XDG_DATA_HOME` | you (optional) | a folder | Read by Qt (`QStandardPaths`): parent of the default data folder `anima-trainer` |
| `XDG_CONFIG_HOME` | you (optional) | a folder | Read by Qt (`QSettings`): where the app keeps its settings, e.g. the chosen data folder |
| `QT_QPA_PLATFORM` | you, when running tests | `offscreen` | Lets Qt tests run without a display |
<!-- /AUTO-GENERATED -->

## Tests

- Layout: `tests/unit/` (pure logic), `tests/integration/` (subprocesses, local HTTP
  server, Qt signals), `tests/e2e/` (UI driven with pytest-qt), fixtures in
  `tests/fixtures/`.
- Run a subset: `.venv/bin/pytest -m integration`, `.venv/bin/pytest tests/unit`.
- Workflow is TDD: write the failing test first, then the code. Coverage must stay at
  80% or more.
- Golden tests in `tests/unit/training/test_toml_builder.py` compare generated TOML
  byte for byte with real Citron files in `tests/fixtures/citron/`. If Citron's output
  format changes, refresh those fixtures from a real Citron run, not by hand.
- `tests/fixtures/fake_accelerate.py` stands in for `accelerate launch` (replay a log,
  exit codes, crash, process groups). Extend it rather than spawning real training.
- Tests that wait on Qt signals must not rely on timing: throttle the fake server
  (`block_delay_s`) instead of hoping a race goes your way, and make sure every thread
  or process a test starts has stopped before it ends.

## Code style

- ruff for lint and format, strict mypy; both must be clean.
- Many small files organized by feature (`settings/`, `dataset/`, `training/`,
  `models/`, `preview/`, `system/`, `ui/`), 200-400 lines typical, 800 max.
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
