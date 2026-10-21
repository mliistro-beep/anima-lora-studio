<!-- Generated: 2026-10-21 | Files scanned: 97 (47 src, 50 tests) | Token estimate: ~300 -->
# Frontend (Qt UI)

main.py create_window(store, detect) -> MainWindow(session, GpuDetector, store, opened)

## MainWindow (ui/main_window.py)
  tabs/training_tab.py   basic settings form, Configure / Start / Stop, custom TOML, status, progress, log
  tabs/advanced_tab.py   advanced settings form
  tabs/settings_tab.py   data folder (Apply), runtime problems, model status table, DiT download/cancel
  tabs/previews_tab.py   epoch x prompt grid (M5, not built)
  tabs/generator_tab.py  manual inference on a checkpoint (M5, not built)
  widgets/ path_picker, log_view (tqdm redraw collapse, max 5000 lines), progress_panel, settings_form

## State flow
form values -> TrainingSettings.with_changes() (new object each time) -> TrainingSession.configure
TrainingSession signals -> progress_panel, log_view, tab button states (SessionState)
Settings tab Apply -> open_data_dir -> session.set_runtime -> forms reloaded, save_data_dir
window close / aboutToQuit / SIGINT / SIGTERM -> MainWindow.shutdown() (session + GPU detector)
