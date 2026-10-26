<!-- Generated: 2026-10-26 | Files scanned: 129 (62 src, 67 tests) | Token estimate: ~300 -->
# Frontend (Qt UI)

main.py create_window(store, detect) -> MainWindow(session, GpuDetector, store, opened,
                                                  InstallSession(RuntimeInstaller()))
                                       (LinkSession(store, ModelFileChecker()) created by default)

## MainWindow (ui/main_window.py)
  tabs/training_tab.py   basic settings form, Configure / Start / Stop, custom TOML, status, progress, log
  tabs/advanced_tab.py   advanced settings form
  tabs/settings_tab.py   data folder (Apply), runtime problems, Install / Repair / Cancel install,
                         install log, model status table, Qwen3 / VAE / DiT download and cancel,
                         Use file... / Unlink per model, file check progress and Cancel check
  tabs/previews_tab.py   epoch x prompt grid (M5, not built)
  tabs/generator_tab.py  manual inference on a checkpoint (M5, not built)
  widgets/ path_picker, log_view (tqdm redraw collapse, max 5000 lines), progress_panel, settings_form

## State flow
form values -> TrainingSettings.with_changes() (new object each time) -> TrainingSession.configure
TrainingSession signals -> progress_panel, log_view, tab button states (SessionState)
Settings tab Apply -> open_data_dir -> with saved links -> session.set_runtime -> forms reloaded, save_data_dir
Settings tab Use file... -> LinkSession.check -> (training, install, downloads disabled) -> finished
  -> result shown; if linked: session.set_runtime(paths.with_links(...)), model table refreshed
Settings tab shown -> model table re-read (a linked file may have been moved or changed)
Settings tab Install -> (apply the picker folder if changed) -> InstallSession.start(session.paths)
  -> problems shown, or RuntimeInstaller runs;
  training and downloads disabled until finished -> runtime_problems, models, save_data_dir, GPU detection
window close / aboutToQuit / SIGINT / SIGTERM -> MainWindow.shutdown() (install + file check + session + GPU detector)
