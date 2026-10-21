<!-- Generated: 2026-10-21 | Files scanned: 97 (47 src, 50 tests) | Token estimate: ~700 -->
# Backend (core and services)

No HTTP API: "backend" = non-UI modules called by the Qt UI. Public surface only.

## runtime/
paths.py     RuntimePaths(root, platform): config/configs/logs/models/scripts/venv paths,
             accelerate_config (relative) / accelerate_config_file, dit_model(base),
             required_models(), missing_required_models(), resolve(user_path)
data_dir.py  default_data_dir(), load_data_dir(store, default), save_data_dir(store, dir) (OSError),
             runtime_problems(paths), open_data_dir(dir) -> OpenedRuntime(paths, settings, warnings, problems)

## settings/
schema.py      TrainingSettings(BaseModel, frozen) 35 fields; with_changes(**) -> new; missing_required_fields()
defaults.py    DEFAULTS: CitronDefaults (read-only TypedDict over MappingProxyType)
choices.py     Literal types + tuples: BASE_MODELS, OPTIMIZERS, LR_SCHEDULERS, MIXED_PRECISIONS, TIMESTEP_SAMPLINGS
store.py       load_settings(path) -> LoadResult(settings, warnings); save_settings(path, settings) atomic
field_specs.py FieldSpec per field (Citron labels, groups, limits); describe_validation_error(error)

## dataset/
validator.py      validate_dataset(dir) -> DatasetReport(image_count, missing_captions, warnings); DatasetError
step_estimator.py estimate_steps(images, repeats, batch, accum, epochs) -> StepEstimate

## models/
catalog.py         BASE_MODEL_URLS, dit_filename(base)
status.py          model_statuses(paths) -> tuple[ModelStatus(label, path, present, base_model)]
downloader.py      download_file(url, dest, on_progress, is_cancelled) -> Path; DownloadError, DownloadCancelled
download_worker.py ModelDownloader(QObject): start/cancel/shutdown; signals progress, finished, failed, cancelled

## training/
toml_builder.py      build_training_config, build_dataset_config, render_toml, config_file_name, write_configs
configure_service.py configure_training(settings, paths, now) -> ConfigureResult
start_service.py     StartRequest, dit_download(paths, base) -> DitDownload|None, plan_start(...) -> StartPlan
command_builder.py   build_training_launch(...) -> LaunchSpec(program, arguments, env, working_dir)
runner.py            TrainingRunner(QObject): start/stop/shutdown/log_tail; signals line_received,
                     event_received, finished(RunOutcome), failed_to_start, warning; split_lines, log_file_path
log_parser.py        parse_line(line) -> TrainingEvent|None; parse_output(text); parse_duration
progress.py          ProgressState (frozen), apply_event(state, event), format_duration(seconds)
outcome_report.py    completed(outcome), run_failed(outcome), outcome_messages(outcome, output_dir, hint)
oom_hint.py          oom_hint(exit_code, log_tail, kernel_log, crashed) -> str|None; read_kernel_log()

## system/
gpu_info.py   detect_gpus(paths, run) -> tuple[str] (training venv torch -> nvidia-smi -> ("0","1"))
gpu_worker.py GpuDetector(detect): start(paths)/shutdown; signals detected(choices), warning
gpu_choice.py gpu_index_from_choice(label) -> "N"; choice_for_index(choices, saved)

## workflow/
training_session.py TrainingSession(paths, settings, runner, downloader): configure, start, download_model,
                    stop, shutdown, set_runtime; signals state_changed, log_line, progress_changed, ...

## Largest files
workflow/training_session.py 245 | training/runner.py 230 | ui/main_window.py 172
