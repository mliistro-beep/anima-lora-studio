<!-- Generated: 2026-10-27 | Files scanned: 138 (67 src, 71 tests) | Token estimate: ~600 -->
# Architecture

Single desktop app (PySide6, src layout). The GUI venv runs Qt + logic; training and
inference run as subprocesses in a separate training venv inside the app's data folder
(same layout as a Citron install, but owned by the app; Citron is never used).
Status: M0-M4b done (UI with Citron parity; own runtime installed from the Settings tab;
existing model files linked after a SHA-256 check; each verified with a real run);
M4c: License and Credits dialogs, dev launcher scripts/dev-run.sh.
Previews and generator (M5) not built.

## Entry point
src/anima_trainer/main.py  main() -> int; create_window(store, detect) -> MainWindow
console script: anima-trainer = anima_trainer.main:main

## Layers (feature packages)
runtime/   data folder (QSettings key runtime/data_dir, default ~/.local/share/anima-trainer), RuntimePaths,
           install plan and QProcess installer
settings/  validated immutable TrainingSettings, config.json store, field specs for the forms
dataset/   folder validation, step estimate
models/    catalog (URLs, expected size + SHA-256), model status, downloader and file check
           (+ background workers), linked model files (QSettings group linked_models)
training/  TOML builder, configure/start services, accelerate command, QProcess runner, log parser,
           progress state, outcome report, OOM hint
system/    GPU detection (+ QThread worker), dropdown parsing, process groups, BackgroundTask
workflow/  TrainingSession: Configure / model download / Start / Stop state machine;
           InstallSession: plan, run and log the runtime install;
           LinkSession: check and save a model file the user already has
about/     license notice (GPL-3.0-or-later, app version), credits (author, Citron, models, software),
           rich text for the License and Credits dialogs
ui/        MainWindow, tabs (training, advanced, settings), widgets, dialogs (License / Credits)
preview/   (M5, empty)

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

## Download flow
ModelDownloader.start(url, dest) -> BackgroundTask thread -> download_file (.part, Range resume)
  -> progress(done, total) ... then one of finished|failed|cancelled (after thread exit)

## Linked model flow
Settings tab Use file... -> LinkSession.check(key, path) -> ModelFileChecker (BackgroundTask)
  -> check_model_file: size, then SHA-256 in chunks vs EXPECTED_FILES[key] -> LinkedModel(path, size, mtime_ns)
  -> save_link (QSettings) -> RuntimePaths.with_links(load_links) -> session.set_runtime
RuntimePaths.qwen3_model / vae_model / dit_model -> the linked file while size and mtime are unchanged,
  else the data folder path (downloads never write into the user's folder)
