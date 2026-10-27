"""Main window: the three tabs wired to the training, install and model link sessions, GPU
detection and QSettings, plus the License and Credits buttons."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError
from PySide6.QtCore import QSettings
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QWidget,
)

from anima_trainer.about.credits import CREDIT_SECTIONS
from anima_trainer.about.license import app_version
from anima_trainer.about.render import credits_html, license_html
from anima_trainer.models.check_worker import ModelFileChecker
from anima_trainer.models.status import model_statuses
from anima_trainer.runtime.data_dir import (
    OpenedRuntime,
    data_dir_problem,
    open_data_dir,
    resolve_data_dir,
    runtime_problems,
    save_data_dir,
)
from anima_trainer.settings.field_specs import describe_validation_error
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.system.gpu_worker import GpuDetector
from anima_trainer.training.start_service import StartRequest
from anima_trainer.ui.dialogs.text_dialog import TextDialog
from anima_trainer.ui.tabs.advanced_tab import AdvancedTab
from anima_trainer.ui.tabs.settings_tab import SettingsTab
from anima_trainer.ui.tabs.training_tab import TrainingTab
from anima_trainer.workflow.install_session import InstallSession
from anima_trainer.workflow.link_session import LinkSession
from anima_trainer.workflow.training_session import SessionState, TrainingSession

APP_NAME = "Anima LoRA Trainer"
DEFAULT_SIZE = (1200, 800)
QUIT_QUESTION = (
    "Training, a model download, a model file check or the install is still running. "
    "Stop it and quit?"
)


class MainWindow(QMainWindow):
    def __init__(
        self,
        session: TrainingSession,
        detector: GpuDetector,
        store: QSettings,
        opened: OpenedRuntime,
        install: InstallSession,
        open_root: Callable[[Path], OpenedRuntime] = open_data_dir,
        links: LinkSession | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.resize(*DEFAULT_SIZE)
        self.session = session
        self._detector = detector
        self._store = store
        self._install = install
        self._installing = False
        self._links = links if links is not None else LinkSession(store, ModelFileChecker())
        self._checking = False
        self._open_root = open_root
        self._problems = opened.problems
        self._root_exists = opened.paths.root.is_dir()
        self._shut_down = False
        self.training_tab = TrainingTab()
        self.advanced_tab = AdvancedTab()
        self.settings_tab = SettingsTab()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.advanced_tab, "Advanced Settings")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.license_button = QPushButton("License")
        self.credits_button = QPushButton("Credits")
        self.about_dialog: TextDialog | None = None
        self.tabs.setCornerWidget(self._about_buttons())
        self.setCentralWidget(self.tabs)
        self._connect()
        opened = self._with_links(opened)
        self.session.set_runtime(opened.paths, opened.settings)
        self._show_runtime(opened)
        self._detector.start(opened.paths)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        if self.session.state is not SessionState.IDLE or self._installing or self._checking:
            answer = QMessageBox.question(
                self,
                APP_NAME,
                QUIT_QUESTION,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        # Hidden first: the waits below can take seconds (runner kill grace, GPU probe).
        self.hide()
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        """Stop training, downloads, file checks, the install and GPU detection; safe to call
        more than once.

        Also connected to QApplication.aboutToQuit, so exits that skip closeEvent
        (Ctrl+C, SIGTERM, logout) do not leave the training child running on the GPU.
        """
        if self._shut_down:
            return
        self._shut_down = True
        self._install.shutdown()
        self._links.shutdown()
        self.session.shutdown()
        self._detector.shutdown()

    def _about_buttons(self) -> QWidget:
        corner = QWidget()
        layout = QHBoxLayout(corner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.license_button)
        layout.addWidget(self.credits_button)
        return corner

    def _show_about(self, title: str, html: str) -> None:
        # open(), not exec(): window-modal without a nested event loop.
        if self.about_dialog is not None:
            self.about_dialog.deleteLater()
        self.about_dialog = TextDialog(title, html, self)
        self.about_dialog.open()

    def _connect(self) -> None:
        self.license_button.clicked.connect(
            lambda: self._show_about("License", license_html(APP_NAME, app_version()))
        )
        self.credits_button.clicked.connect(
            lambda: self._show_about("Credits", credits_html(CREDIT_SECTIONS))
        )
        tab = self.training_tab
        tab.configure_button.clicked.connect(self._configure)
        tab.start_button.clicked.connect(self._start)
        tab.stop_button.clicked.connect(self.session.stop)
        # Linked model files may be moved or edited while the app runs.
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.settings_tab.root_applied.connect(self._apply_root)
        self.settings_tab.download_requested.connect(self.session.download_model)
        self.settings_tab.cancel_requested.connect(self.session.stop)
        self.settings_tab.install_requested.connect(self._start_install)
        self.settings_tab.cancel_install_requested.connect(self._install.cancel)
        self.settings_tab.link_requested.connect(self._link_model)
        self.settings_tab.unlink_requested.connect(self._unlink_model)
        self.settings_tab.cancel_check_requested.connect(self._links.cancel)
        self._links.progress.connect(self.settings_tab.show_check)
        self._links.finished.connect(self._on_link_finished)
        self._install.log_line.connect(self.settings_tab.install_log.append_line)
        self._install.step_changed.connect(self.settings_tab.show_install_step)
        self._install.finished.connect(self._on_install_finished)
        self.session.log_line.connect(tab.log.append_line)
        self.session.log_cleared.connect(tab.log.clear)
        self.session.progress_changed.connect(tab.progress.show_progress)
        self.session.download_progress.connect(tab.progress.show_download)
        self.session.download_progress.connect(self.settings_tab.show_download)
        self.session.state_changed.connect(self._apply_state)
        self.session.models_changed.connect(self._refresh_models)
        self._detector.detected.connect(self._on_gpus_detected)
        self._detector.warning.connect(lambda message: tab.status.append_line(f"WARN: {message}"))

    def _show_runtime(self, opened: OpenedRuntime) -> None:
        self._problems = opened.problems
        self._root_exists = opened.paths.root.is_dir()
        for form in (self.training_tab.form, self.advanced_tab.form):
            form.set_values(opened.settings)
        self.settings_tab.set_root(str(opened.paths.root))
        self.settings_tab.set_problems(opened.problems)
        self._refresh_models()
        notes = (
            *(f"WARN: {warning}" for warning in opened.warnings),
            *(f"ERROR: {problem}" for problem in opened.problems),
        )
        self.training_tab.status.set_lines(notes)
        if opened.problems:
            self.tabs.setCurrentWidget(self.settings_tab)
        self._apply_state(self.session.state)

    def _collect_settings(self) -> TrainingSettings | None:
        values = {**self.training_tab.form.values(), **self.advanced_tab.form.values()}
        try:
            return self.session.settings.with_changes(**values)
        except ValidationError as error:
            self.training_tab.status.set_lines(describe_validation_error(error))
            return None

    def _configure(self) -> None:
        settings = self._collect_settings()
        if settings is not None:
            self.training_tab.status.set_lines(self.session.configure(settings).messages)

    def _start(self) -> None:
        # Like Citron, Start reads only these fields; the rest comes from the saved TOMLs,
        # so an invalid value elsewhere in the form does not block it.
        basic, advanced = self.training_tab.form, self.advanced_tab.form
        request = StartRequest(
            gpu_index=str(basic.value("gpu_index")),
            cpu_threads=int(advanced.value("num_cpu_threads_per_process")),
            base_model=str(basic.value("base_model")),
            custom_config=self.training_tab.custom_config_path(),
        )
        self.session.start(request)

    def _apply_root(self, text: str) -> None:
        if not text.strip():
            self.settings_tab.set_problems(("Choose the data folder first.",))
            return
        opened = self._with_links(self._open_root(Path(text.strip())))
        self.session.set_runtime(opened.paths, opened.settings)
        self._show_runtime(opened)
        self._detector.start(opened.paths)
        if not opened.paths.root.is_dir():
            return
        try:
            save_data_dir(self._store, opened.paths.root)
        except OSError as error:
            self.training_tab.status.append_line(f"WARN: {error}")

    def _start_install(self) -> None:
        if self._busy():
            return
        if not self._use_typed_folder():
            return
        # Finding Python (only for a new venv) probes a few interpreters on this thread.
        problems = self._install.start(self.session.paths)
        for problem in problems:
            self.settings_tab.install_log.append_line(f"ERROR: {problem}")
        if not problems:
            self._set_installing(True)

    def _use_typed_folder(self) -> bool:
        """Apply the picker folder if it changed; False when it cannot hold the runtime.

        Checked before applying, so a mistyped path leaves the current folder active.
        """
        typed = self.settings_tab.root_picker.path().strip()
        if not typed:
            self._apply_root(typed)  # shows "Choose the data folder first."
            return False
        folder = resolve_data_dir(Path(typed))
        problem = data_dir_problem(folder)
        if problem is not None:
            self.settings_tab.install_log.append_line(f"ERROR: {problem}")
            return False
        if folder != self.session.paths.root:
            self._apply_root(typed)
        return True

    def _on_install_finished(self, ok: bool, message: str) -> None:
        """Re-check the runtime, also after a failure: a partial install changes the folder."""
        if self._shut_down:
            # Cancelled by shutdown(): a new GPU detection would delay the exit (torch probe).
            return
        paths = self.session.paths
        self._problems = runtime_problems(paths)
        self._root_exists = paths.root.is_dir()
        self.settings_tab.set_problems(self._problems)
        self._refresh_models()
        # Appended: the status may hold validation or Configure messages worth keeping.
        notes = tuple(f"ERROR: {problem}" for problem in self._problems)
        outcome = "OK: Training runtime installed." if ok else f"ERROR: {message}"
        for note in notes or (outcome,):
            self.training_tab.status.append_line(note)
        if self._root_exists:
            try:
                save_data_dir(self._store, paths.root)
            except OSError as error:
                self.training_tab.status.append_line(f"WARN: {error}")
        self._detector.start(paths)  # the training venv may have torch now
        self._set_installing(False)

    def _link_model(self, key: str, path: str) -> None:
        if self._busy():
            return
        error = self._links.check(key, Path(path))
        if error is not None:
            self.settings_tab.show_check_result(f"ERROR: {error}")
            return
        self._set_checking(True)

    def _on_link_finished(self, linked: bool, message: str) -> None:
        if self._shut_down:
            return
        self._set_checking(False)
        self.settings_tab.show_check_result(message)
        if linked:
            self._reload_links()

    def _unlink_model(self, key: str) -> None:
        if self._busy():
            return
        error = self._links.unlink(key)
        if error is not None:
            self.settings_tab.show_check_result(f"ERROR: {error}")
        self._reload_links()

    def _reload_links(self) -> None:
        """Use the saved links; called only while idle, as set_runtime requires."""
        self.session.set_runtime(
            self.session.paths.with_links(self._links.links()), self.session.settings
        )
        self._refresh_models()

    def _with_links(self, opened: OpenedRuntime) -> OpenedRuntime:
        return replace(opened, paths=opened.paths.with_links(self._links.links()))

    def _busy(self) -> bool:
        return self.session.state is not SessionState.IDLE or self._installing or self._checking

    def _set_checking(self, checking: bool) -> None:
        self._checking = checking
        self.settings_tab.set_checking(checking)
        self._apply_state(self.session.state)

    def _set_installing(self, installing: bool) -> None:
        self._installing = installing
        self.settings_tab.set_installing(installing)
        self._apply_state(self.session.state)

    def _apply_state(self, state: SessionState) -> None:
        ready = not self._problems and not self._installing and not self._checking
        self.training_tab.set_state(state, ready=ready)
        self.settings_tab.set_state(state, can_download=self._root_exists)

    def _on_tab_changed(self, _index: int) -> None:
        if self.tabs.currentWidget() is self.settings_tab:
            self._refresh_models()

    def _refresh_models(self) -> None:
        self.settings_tab.set_models(model_statuses(self.session.paths))

    def _on_gpus_detected(self, choices: tuple[str, ...]) -> None:
        current = str(self.training_tab.form.value("gpu_index"))
        self.training_tab.form.set_gpu_choices(choices, current)
