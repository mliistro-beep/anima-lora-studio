<!-- Generated: 2026-10-20 | Files scanned: 50 (19 src, 31 tests) | Token estimate: ~210 -->
# Frontend (Qt UI)

Current: main.py create_window() -> empty QMainWindow "Anima LoRA Trainer". ui/ is empty.

## Planned (M3, M4; see DOCS/milestones.md)
MainWindow
  tabs/training_tab.py   basic settings, Configure / Start / Stop
  tabs/advanced_tab.py   advanced settings
  tabs/settings_tab.py   Citron root, model status, download
  tabs/previews_tab.py   epoch x prompt grid (M4)
  tabs/generator_tab.py  manual inference on a checkpoint (M4)
  widgets/ path_picker, log_view, progress_panel, image_gallery

## State flow (planned)
widgets edit -> TrainingSettings.with_changes() (new object each time)
TrainingRunner / ModelDownloader signals -> progress_panel, log_view
window close -> TrainingRunner.shutdown(), ModelDownloader.shutdown()
