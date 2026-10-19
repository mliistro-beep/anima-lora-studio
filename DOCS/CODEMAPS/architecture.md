<!-- Generated: 2026-10-20 | Files scanned: 50 (19 src, 31 tests) | Token estimate: ~517 -->
# Architecture

Single desktop app (PySide6, src layout). Wraps an external Citron install: the GUI venv
runs Qt + logic; training/inference run as subprocesses in the Citron venv.
Status: core + services done (M0-M2); UI (M3) and previews (M4) not built.

## Entry point
src/anima_trainer/main.py  main() -> int, create_window() (empty QMainWindow for now)
console script: anima-trainer = anima_trainer.main:main

## Layers (feature packages)
settings/  validated immutable TrainingSettings, config.json store
dataset/   folder validation, step estimate
models/    Citron layout (CitronPaths), base model catalog, downloader (+ QThread worker)
training/  TOML builder, configure flow, accelerate command, QProcess runner, log parser, OOM hint
system/    GPU detection and dropdown parsing
preview/   (M4, empty)   ui/ tabs/ widgets/  (M3, empty)

## Configure flow
TrainingSettings -> configure_training(settings, CitronPaths, now)
  -> resolve relative dirs against Citron root
  -> validate_dataset -> estimate_steps -> model check (Qwen3, VAE; DiT auto-download)
  -> write_configs -> <citron>/configs/*_training_*.toml, *_dataset_*.toml
  -> save_settings -> <citron>/config.json
  -> ConfigureResult(success, messages, settings, configs, dataset, estimate)

## Train flow
build_training_launch(paths, toml, toml, gpu, threads) -> LaunchSpec
TrainingRunner.start(spec, log_path)
  QProcess (new session) -> <citron>/.venv/bin/accelerate launch ... anima_train_network.py
  stdout -> split_lines -> line_received(str) + log file
                        -> parse_line -> event_received(EpochStarted|StepProgress|CheckpointSaved|...)
  exit -> finished(RunOutcome) -> oom_hint(exit_code, log_tail, dmesg, crashed)
stop()/shutdown(): SIGTERM process group, SIGKILL after grace

## Download flow
ModelDownloader.start(url, dest) -> _DownloadThread -> download_file (.part, Range resume)
  -> progress(done, total) ... then one of finished|failed|cancelled (after thread exit)
