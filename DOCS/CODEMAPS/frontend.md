<!-- Generated: 2026-10-24 | Files scanned: 117 (56 src, 61 tests) | Token estimate: ~300 -->
# Frontend (Qt UI)

main.py create_window(store, detect) -> MainWindow(session, GpuDetector, store, opened,
                                                  InstallSession(RuntimeInstaller()))

## MainWindow (ui/main_window.py)
  tabs/training_tab.py   basic settings form, Configure / Start / Stop, custom TOML, status, progress, log
  tabs/advanced_tab.py   advanced settings form
  tabs/settings_tab.py   data folder (Apply), runtime problems, Install / Repair / Cancel install,
                         install log, model status table, Qwen3 / VAE / DiT download and cancel
  tabs/previews_tab.py   epoch x prompt grid (M5, not built)
  tabs/generator_tab.py  manual inference on a checkpoint (M5, not built)
  widgets/ path_picker, log_view (tqdm redraw collapse, max 5000 lines), progress_panel, settings_form

## State flow
form values -> TrainingSettings.with_changes() (new object each time) -> TrainingSession.configure
TrainingSession signals -> progress_panel, log_view, tab button states (SessionState)
Settings tab Apply -> open_data_dir -> session.set_runtime -> forms reloaded, save_data_dir
Settings tab Install -> (apply the picker folder if changed) -> InstallSession.start(session.paths)
  -> problems shown, or RuntimeInstaller runs;
  training and downloads disabled until finished -> runtime_problems, models, save_data_dir, GPU detection
window close / aboutToQuit / SIGINT / SIGTERM -> MainWindow.shutdown() (install + session + GPU detector)
