<!-- Generated: 2026-10-26 | Files scanned: 129 (62 src, 67 tests) | Token estimate: ~700 -->
# Backend (core and services)

No HTTP API: "backend" = non-UI modules called by the Qt UI. Public surface only.

## runtime/
paths.py     RuntimePaths(root, platform, links): config/configs/logs/models/scripts/venv paths,
             accelerate_config (relative) / accelerate_config_file, dit_model(base),
             required_models(), missing_required_models(), resolve(user_path);
             with_links(links) -> new; is_linked(key), has_broken_link(key); model paths use a
             linked file only while it is unchanged
data_dir.py  default_data_dir(), load_data_dir(store, default), save_data_dir(store, dir) (OSError),
             resolve_data_dir(dir) (absolute), data_dir_problem(dir) -> str|None (a file, or a path
             below a file), runtime_problems(paths) (REPAIR_MESSAGE when a step is pending),
             open_data_dir(dir) -> OpenedRuntime(paths, settings, warnings, problems)
accelerate_config.py ACCELERATE_GPU_YAML (Citron's file), write_accelerate_config(paths) -> Path
python_finder.py find_python(which, run, resolve) -> PythonInterpreter(path, version)|None (>=3.10, venv+ensurepip;
                 >=3.14 only as fallback; one probe per resolved interpreter)
venv_info.py     read_venv_version(venv_dir) -> (major, minor)|None; needs_single_process_loader(version, platform)
install_plan.py  plan_install(paths, python, git) -> InstallPlan(steps, problems, ready); InstallStep(key, title,
                 command, working_dir, marker, stamp, action); SD_SCRIPTS_COMMIT pinned, EXTRA_PACKAGES
                 (torchvision); markers in <data>/.install: venv_marker, sd_scripts_marker,
                 requirements_marker, verified_marker; needs_venv(paths), install_pending(paths)
installer.py     RuntimeInstaller(QObject): start(steps)/cancel/shutdown/is_running; signals step_started(i, n, step),
                 line_received, finished(ok, message); writes the step stamp to its marker after success,
                 stops at first failure

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
catalog.py         BASE_MODEL_URLS, QWEN3_FILENAME, VAE_FILENAME, SUPPORT_MODEL_URLS (by file name), dit_filename(base),
                   EXPECTED_FILES[key] -> ExpectedFile(sha256, size) (Hugging Face LFS), model_label(key)
status.py          model_statuses(paths) -> tuple[ModelStatus(label, path, present, download_key, linked, link_broken)]
model_downloads.py model_download(paths, key) / dit_model_download(paths, base) -> ModelDownload(description,
                   url, destination)|None (key: base model name for a DiT, file name for Qwen3 / VAE)
downloader.py      download_file(url, dest, on_progress, is_cancelled) -> Path; DownloadError, DownloadCancelled
download_worker.py ModelDownloader(BackgroundTask): start(url, dest, expected)/cancel/shutdown; finished(path)
linked_models.py   LinkedModel(path, size, mtime_ns); link_for(path); is_unchanged(link)
file_check.py      check_model_file(path, expected, on_progress, is_cancelled) -> LinkedModel; ModelFileRejected,
                   CheckCancelled (size first, then SHA-256; read only)
linked_store.py    load_links(store) -> Mapping[key, LinkedModel]; save_link / remove_link (OSError)
check_worker.py    ModelFileChecker(BackgroundTask): start(path, expected); finished(LinkedModel)

## training/
toml_builder.py      build_training_config, build_dataset_config, render_toml, config_file_name, write_configs
configure_service.py configure_training(settings, paths, now) -> ConfigureResult (3.14+ venv: max_data_loader_n_workers = 0)
start_service.py     StartRequest, plan_start(...) -> StartPlan
command_builder.py   build_training_launch(...) -> LaunchSpec(program, arguments, env, working_dir)
runner.py            TrainingRunner(QObject): start/stop/shutdown/log_tail; signals line_received,
                     event_received, finished(RunOutcome), failed_to_start, warning; split_lines, log_file_path
log_parser.py        parse_line(line) -> TrainingEvent|None; parse_duration
progress.py          ProgressState (frozen), apply_event(state, event), format_duration(seconds)
outcome_report.py    completed(outcome), run_failed(outcome), outcome_messages(outcome, output_dir, hint)
oom_hint.py          oom_hint(exit_code, log_tail, kernel_log, crashed) -> str|None; read_kernel_log()

## system/
gpu_info.py   detect_gpus(paths, run) -> tuple[str] (training venv torch -> nvidia-smi -> ("0","1"))
gpu_worker.py GpuDetector(detect): start(paths)/shutdown; signals detected(choices), warning
gpu_choice.py gpu_index_from_choice(label) -> "N"; choice_for_index(choices, saved)
process_group.py use_own_session(process), terminate_group(process), kill_group(process),
                 KILL_GRACE_MS, SHUTDOWN_KILL_WAIT_MS (runner + installer)
process_output.py split_lines(pending, data) -> (lines, rest)
background_task.py BackgroundTask(QObject): one job at a time on a QThread; progress, then one of
                   finished(value) / failed(message) / cancelled after the thread stopped; cancel, shutdown

## workflow/
training_session.py TrainingSession(paths, settings, runner, downloader): configure, start, download_model,
                    stop, shutdown, set_runtime; signals state_changed, log_line, progress_changed, ...;
                    download_model(key) also fetches Qwen3 / VAE
install_session.py InstallSession(installer, plan=plan_for): start(paths) -> problems, cancel, shutdown;
                   signals log_line, step_changed, finished(ok, message); log <data>/logs/install_<ts>.log;
                   Python is looked for only when the venv must be created
link_session.py    LinkSession(store, checker, expected): check(key, path) -> problem|None, unlink(key), links(),
                   cancel, shutdown, is_running; signals progress, finished(linked, message)

## Largest files
ui/main_window.py 316 | workflow/training_session.py 242 | ui/tabs/settings_tab.py 238 | runtime/install_plan.py 226
