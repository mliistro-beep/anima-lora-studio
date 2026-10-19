<!-- Generated: 2026-10-20 | Files scanned: 50 (19 src, 31 tests) | Token estimate: ~607 -->
# Backend (core and services)

No HTTP API: "backend" = non-UI modules called by the Qt UI. Public surface only.

## settings/
schema.py   TrainingSettings(BaseModel, frozen) 35 fields; with_changes(**) -> new; missing_required_fields()
            validators: project name ^[\w.-]+$ (fullmatch), gpu_index digits, no bool ints, no inf/nan
defaults.py DEFAULTS: CitronDefaults (read-only TypedDict over MappingProxyType)
choices.py  Literal types + tuples: BASE_MODELS, OPTIMIZERS, LR_SCHEDULERS, MIXED_PRECISIONS, TIMESTEP_SAMPLINGS
store.py    load_settings(path) -> LoadResult(settings, warnings); save_settings(path, settings) atomic

## dataset/
validator.py      validate_dataset(dir) -> DatasetReport(image_count, missing_captions, warnings); DatasetError
step_estimator.py estimate_steps(images, repeats, batch, accum, epochs) -> StepEstimate

## models/
paths.py           CitronPaths(root, platform): config/configs/logs/models/scripts/venv paths,
                   dit_model(base), required_models(), missing_required_models(), resolve(user_path)
catalog.py         BASE_MODEL_URLS, dit_filename(base)
downloader.py      download_file(url, dest, on_progress, is_cancelled) -> Path; DownloadError, DownloadCancelled
download_worker.py ModelDownloader(QObject): start/cancel/shutdown; signals progress, finished, failed, cancelled

## training/
toml_builder.py      build_training_config, build_dataset_config, render_toml, config_file_name, write_configs -> GeneratedConfigs
configure_service.py configure_training(settings, paths, now) -> ConfigureResult
command_builder.py   build_training_launch(...) -> LaunchSpec(program, arguments, env, working_dir)
runner.py            TrainingRunner(QObject): start/stop/shutdown/log_tail; signals line_received,
                     event_received, finished(RunOutcome), failed_to_start, warning; split_lines, log_file_path
log_parser.py        parse_line(line) -> TrainingEvent|None; parse_output(text); parse_duration
oom_hint.py          oom_hint(exit_code, log_tail, kernel_log, crashed) -> str|None; read_kernel_log()

## system/
gpu_info.py   detect_gpus(paths, run) -> tuple[str] (Citron torch -> nvidia-smi -> ("0","1"))
gpu_choice.py gpu_index_from_choice(label) -> "N"

## Largest files
training/runner.py 230 | training/configure_service.py 168 | models/downloader.py 140
