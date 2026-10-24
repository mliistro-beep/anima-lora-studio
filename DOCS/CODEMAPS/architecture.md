<!-- Generated: 2026-10-21 | Files scanned: 97 (47 src, 50 tests) | Token estimate: ~560 -->
# Architecture

Single desktop app (PySide6, src layout). The GUI venv runs Qt + logic; training and
inference run as subprocesses in a separate training venv inside the app's data folder
(same layout as a Citron install, but owned by the app; Citron is never used).
Status: M0-M3 done (UI with Citron parity, verified with a real run). M4 (independent
runtime) in progress: data folder done, installer not built. Previews (M5) not built.

## Entry point
src/anima_trainer/main.py  main() -> int; create_window(store, detect) -> MainWindow
console script: anima-trainer = anima_trainer.main:main

## Layers (feature packages)
runtime/   data folder (QSettings key runtime/data_dir, default ~/.local/share/anima-trainer), RuntimePaths,
           install plan and QProcess installer
settings/  validated immutable TrainingSettings, config.json store, field specs for the forms
dataset/   folder validation, step estimate
models/    base model catalog, model status, downloader (+ QThread worker)
training/  TOML builder, configure/start services, accelerate command, QProcess runner, log parser,
           progress state, outcome report, OOM hint
system/    GPU detection (+ QThread worker), dropdown parsing, process groups
workflow/  TrainingSession: Configure / model download / Start / Stop state machine;
           InstallSession: plan, run and log the runtime install
ui/        MainWindow, tabs (training, advanced, settings), widgets
preview/   (M5, empty)

## Configure flow
TrainingSettings -> configure_training(settings, RuntimePaths, now)
  -> resolve relative dirs against the data folder
  -> validate_dataset -> estimate_steps -> model check (Qwen3, VAE; DiT auto-download)
  -> write_configs -> <data>/configs/*_training_*.toml, *_dataset_*.toml
  -> save_settings -> <data>/config.json
  -> ConfigureResult(success, messages, settings, configs, dataset, estimate)

## Train flow
TrainingSession.start(StartRequest) -> dit_download? -> ModelDownloader -> plan_start -> LaunchSpec
TrainingRunner.start(spec, log_path)
  QProcess (new session, cwd <data>) -> <data>/.venv/bin/accelerate launch ... anima_train_network.py
  stdout -> split_lines -> line_received(str) + log file
                        -> parse_line -> event_received -> apply_event -> ProgressState
  exit -> finished(RunOutcome) -> outcome_messages + oom_hint(exit_code, log_tail, dmesg, crashed)
stop()/shutdown(): SIGTERM process group, SIGKILL after grace; window close and SIGINT/SIGTERM -> shutdown

## Download flow
ModelDownloader.start(url, dest) -> _DownloadThread -> download_file (.part, Range resume)
  -> progress(done, total) ... then one of finished|failed|cancelled (after thread exit)
