<!-- Generated: 2026-10-28 | Files scanned: 199 (95 src, 104 tests) | Token estimate: ~360 -->
# Frontend (Qt UI)

main.py main(): display name APP_NAME "Anima LoRA Studio", window icon ui/app_icon.py app_icon(),
                setDesktopFileName(DESKTOP_FILE_NAME) (the menu entry run_linux.sh installs)
main.py create_window(store, detect) -> MainWindow(session, GpuDetector, store, opened,
                                                  InstallSession(RuntimeInstaller()),
                                                  memory_check=read_memory_safety)
                                       (LinkSession(store, ModelFileChecker()) created by default)

## MainWindow (ui/main_window.py)
  tabs/training_tab.py   basic settings form, Configure / Start / Stop, custom TOML, status, progress, log
  tabs/advanced_tab.py   advanced settings form
  tabs/settings_tab.py   data folder (Apply), runtime problems, swap / OOM daemon lines, Install / Repair / Cancel install,
                         install log, model status table, Qwen3 / VAE / DiT download and cancel,
                         Use file... / Unlink per model, file check progress and Cancel check
  tabs/previews_tab.py   preview settings and prompts (used by Configure), epoch x prompt gallery
                         (read-only while training)
  tabs/generator_tab.py  checkpoint or no LoRA, prompt and parameters, Generate / Compare Epochs /
                         Stop, progress, result and Enlarge, generator log
  preview_controller.py  wires both tabs to SampleWatcher, GeneratorSession, preview.json and
                         generator.json; lighter profile note while a training runs
  corner of the tab bar  License / Credits buttons -> dialogs/text_dialog.py TextDialog (QTextBrowser,
                         external links) with about/render.py license_html / credits_html
  widgets/ path_picker, log_view (tqdm redraw collapse, max 5000 lines), progress_panel, settings_form,
           image_grid (thumbnails cached by path + mtime, never a failed load)
  dialogs/ text_dialog, image_viewer

## State flow
form values -> TrainingSettings.with_changes() (new object each time) -> TrainingSession.configure
TrainingSession signals -> progress_panel, log_view, tab button states (SessionState)
Settings tab Apply -> open_data_dir -> with saved links -> session.set_runtime -> forms reloaded, save_data_dir
start and Apply -> memory_check() -> all OK/WARN lines in the Settings tab, only the WARN lines in the
  Training status
Settings tab Use file... -> LinkSession.check -> (training, install, downloads disabled) -> finished
  -> result shown; if linked: session.set_runtime(paths.with_links(...)), model table refreshed
Settings tab shown -> model table re-read (a linked file may have been moved or changed)
Settings tab Install -> (apply the picker folder if changed) -> InstallSession.start(session.paths)
  -> problems shown, or RuntimeInstaller runs;
  training and downloads disabled until finished -> runtime_problems, models, save_data_dir, GPU detection
Training state TRAINING -> PreviewController.training_started (watch sample/, previews tab read-only);
  back to idle -> training_finished (refresh gallery and checkpoints)
Generator running -> Configure / Start disabled (a training may not start during a generation;
  a generation may join a running training)
window close / aboutToQuit / SIGINT / SIGTERM -> MainWindow.shutdown() (install + file check + previews
  and generator + session + GPU detector)
