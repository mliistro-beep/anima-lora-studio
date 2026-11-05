<!-- Generated: 2026-10-28 | Files scanned: 199 (95 src, 104 tests) | Token estimate: ~600 -->
# Architecture

Single desktop app, Anima LoRA Studio (PySide6, src layout; package anima_trainer, version 0.2.0). The GUI venv runs Qt + logic; training and
inference run as subprocesses in a separate training venv inside the app's data folder
(same layout as a Citron install, but owned by the app; Citron is never used).
Status: M0-M4b done (UI with Citron parity; own runtime installed from the Settings tab;
existing model files linked after a SHA-256 check; each verified with a real run);
M4c: License and Credits dialogs, dev launcher scripts/dev-run.sh.
M5: previews during training and the image generator (memory profiles from free VRAM and RAM),
verified from the real UI on 2026-10-28.
M6 (release 0.2.0): run_linux.sh launcher and menu entry, app icon, swap / OOM daemon warning.

## Entry point
run_linux.sh  finds Python >= 3.12, creates <checkout>/.venv, pip install -e . when the stamp
              (sha256 of pyproject.toml + checkout path) changed, exec .venv/bin/anima-trainer;
              --install-desktop / --uninstall-desktop: anima-lora-studio.desktop + hicolor SVG icon
src/anima_trainer/main.py  main() -> int (window icon, setDesktopFileName("anima-lora-studio"));
                           create_window(store, detect) -> MainWindow
console script: anima-trainer = anima_trainer.cli.entry:main (train -> cli/train_command.py, QtCore only; otherwise main.main)

## Layers (feature packages)
cli/       anima-trainer entry (no graphical import), train (train_options, train_summary, train_command),
           tui/ (Textual; Qt events pumped from an interval), shared: train_preparation, terminal_run
presets/   preset file format, built-in TOMLs, catalog with the user's folder, apply on the defaults
runtime/   data folder (QSettings key runtime/data_dir, default ~/.local/share/anima-trainer), RuntimePaths,
           install plan and QProcess installer
settings/  validated immutable TrainingSettings, config.json store, field specs for the forms
dataset/   folder validation, step estimate
models/    catalog (URLs, expected size + SHA-256), model status, downloader and file check
           (+ background workers), linked model files (QSettings group linked_models)
training/  TOML builder, configure/start services, accelerate command, QProcess runner, log parser,
           progress state, outcome report, OOM hint
system/    GPU detection (+ QThread worker), GPU memory, dropdown parsing, process groups,
           BackgroundTask, memory safety (swap and OOM daemon from /proc)
workflow/  TrainingSession: Configure / model download / Start / Stop state machine;
           GeneratorSession: queue of generate.py runs (Generate, Compare epochs), Stop;
           InstallSession: plan, run and log the runtime install;
           LinkSession: check and save a model file the user already has
about/     license notice (GPL-3.0-or-later, app version), credits (author, Citron, models, software),
           rich text for the License and Credits dialogs
ui/        MainWindow, tabs (training, advanced, previews, generator, settings), PreviewController,
           widgets, dialogs (License / Credits, image viewer), app_icon (assets/anima-lora-studio.svg)
preview/   preview and generator settings (preview.json, generator.json), sample prompts JSON,
           preview plan (size from total VRAM), VRAM/RAM profiles, generate.py command,
           checkpoints, sample files and SampleWatcher (polls sample/, size + mtime), gallery layout
runtime_scripts/  run in the training venv, not imported by the app: train_with_previews.py,
           generate.py, profile_choice.py (pure), samplers.py (ER-SDE, Euler), sampler_self_check.py

## Configure flow
TrainingSettings -> configure_training(settings, RuntimePaths, now)
  -> resolve relative dirs against the data folder
  -> validate_dataset -> estimate_steps -> model check (Qwen3, VAE; DiT auto-download)
  -> write_configs -> <data>/configs/*_training_*.toml, *_dataset_*.toml
  -> save_settings -> <data>/config.json
  -> ConfigureResult(success, messages, settings, configs, dataset, estimate)

## Train flow
TrainingSession.start(StartRequest) -> dit_model_download? -> ModelDownloader -> plan_start -> LaunchSpec
TrainingRunner.start(spec, log_path)
  QProcess (new session, cwd <data>) -> <data>/.venv/bin/accelerate launch ... anima_train_network.py
  stdout -> split_lines -> line_received(str) + log file
                        -> parse_line -> event_received -> apply_event -> ProgressState
  exit -> finished(RunOutcome) -> outcome_messages + oom_hint(exit_code, log_tail, dmesg, crashed)
stop()/shutdown(): SIGTERM process group, SIGKILL after grace; window close and SIGINT/SIGTERM -> shutdown

## Preview flow
Configure with previews on -> plan_previews (size from total VRAM) -> write_configs also writes
  <data>/configs/<proj>_samples_<ts>.json and the sample_* TOML keys
Start -> accelerate launch ... runtime_scripts/train_with_previews.py (same arguments; sd-scripts
  sampling with ER-SDE/Euler, VAE decode in tiles, a preview in OOM is skipped)
SampleWatcher (2 s poll of <output_dir>/sample/) -> samples_changed -> PreviewsTab gallery (epoch x prompt)

## Generator flow
GeneratorTab Generate / Compare Epochs -> PreviewController -> GenerationRequest per checkpoint
  -> GeneratorSession -> build_generation_launch -> <data>/.venv/bin/python generate.py
  generate.py reads free VRAM (torch) and MemAvailable once, usable_profiles skips the ones that do
  not fit (larger margins while a training runs), falls back on CUDA OOM, exit 3 when nothing fits
  -> INFO/OK lines, step progress, image_ready -> <output_dir>/generated/<proj>_<label>_<time>_<seed>.png

## Download flow
ModelDownloader.start(url, dest) -> BackgroundTask thread -> download_file (.part, Range resume)
  -> progress(done, total) ... then one of finished|failed|cancelled (after thread exit)

## Linked model flow
Settings tab Use file... -> LinkSession.check(key, path) -> ModelFileChecker (BackgroundTask)
  -> check_model_file: size, then SHA-256 in chunks vs EXPECTED_FILES[key] -> LinkedModel(path, size, mtime_ns)
  -> save_link (QSettings) -> RuntimePaths.with_links(load_links) -> session.set_runtime
RuntimePaths.qwen3_model / vae_model / dit_model -> the linked file while size and mtime are unchanged,
  else the data folder path (downloads never write into the user's folder)
