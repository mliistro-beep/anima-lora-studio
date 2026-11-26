"""Choose a folder or a model file by browsing, for whoever prefers it to writing the path; a new
folder can be made from the folder browser too (handy on a small remote screen)."""

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Static

HIDDEN_PREFIX = "."
MODEL_SUFFIX = ".safetensors"
PICKER_CSS = """
FolderPicker { align: center middle; }
#folder-box { width: 90%; height: 90%; padding: 1 2; border: thick $primary; background: $surface; }
#folder-tree { height: 1fr; margin: 1 0; }
#folder-box .buttons { height: auto; }
#folder-box .buttons Button { margin-right: 2; }
#folder-new-row { height: auto; }
#folder-name { width: 1fr; }
#folder-errors { height: auto; color: $error; }
"""


class NewFolderError(Exception):
    """A folder that cannot be made; the message is the ERROR: line to show."""


def create_folder(parent: Path, name: str) -> Path:
    """Make `name` inside `parent` and return it. Raises NewFolderError for a name that is not a
    plain, visible folder name, an existing folder, or a folder the system refuses."""
    name = name.strip()
    if not name:
        raise NewFolderError("ERROR: Write a folder name.")
    if "/" in name:
        raise NewFolderError("ERROR: A folder name cannot contain '/'.")
    if "\0" in name:
        raise NewFolderError("ERROR: A folder name cannot contain a NUL character.")
    if name in {".", ".."}:
        raise NewFolderError(f"ERROR: '{name}' is not a folder name.")
    if name.startswith(HIDDEN_PREFIX):
        raise NewFolderError(
            "ERROR: A name starting with '.' makes a hidden folder, not listed here."
        )
    folder = parent / name
    if folder.exists():
        raise NewFolderError(f"ERROR: '{name}' already exists here.")
    try:
        folder.mkdir()
    except OSError as error:
        raise NewFolderError(f"ERROR: Cannot create {folder}: {error}") from error
    return folder


def start_folder(text: str, home: Path) -> Path:
    """Where to start: the nearest folder of `text` that exists, or `home` for a blank field."""
    if not text.strip():
        return home
    path = Path(text.strip()).expanduser().absolute()
    return next((part for part in (path, *path.parents) if part.is_dir()), home)


class FoldersOnly(DirectoryTree):
    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [p for p in paths if p.is_dir() and not p.name.startswith(HIDDEN_PREFIX)]


class FolderPicker(ModalScreen[str | None]):
    """Dismissed with the chosen folder, or None when cancelled."""

    DEFAULT_CSS = PICKER_CSS
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, start: Path) -> None:
        super().__init__()
        self._current = start

    def compose(self) -> ComposeResult:
        with Vertical(id="folder-box"):
            yield Static(str(self._current), id="folder-current", markup=False)
            yield FoldersOnly(self._current, id="folder-tree")
            with Horizontal(id="folder-new-row"):
                yield Input(placeholder="new folder name", id="folder-name", compact=True)
                yield Button("Create", id="folder-create", compact=True)
            yield Static(id="folder-errors", markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Use this folder", variant="primary", id="folder-use")
                yield Button("Up", id="folder-up")
                yield Button("New folder", id="folder-new")
                yield Button("Cancel", id="folder-cancel")

    def on_mount(self) -> None:
        self._show_new_folder(False)
        self.query_one(FoldersOnly).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._create()

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        event.stop()
        self._show(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "folder-use":
            self.dismiss(str(self._current))
        elif event.button.id == "folder-up":
            self._show(self._current.parent)
            self.query_one(FoldersOnly).path = self._current
        elif event.button.id == "folder-new":
            self._show_new_folder(True)
            self.query_one("#folder-name", Input).focus()
        elif event.button.id == "folder-create":
            self._create()
        elif event.button.id == "folder-cancel":
            self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _show(self, folder: Path) -> None:
        self._current = folder
        self.query_one("#folder-current", Static).update(str(folder))
        self.query_one("#folder-errors", Static).update("")

    def _show_new_folder(self, shown: bool) -> None:
        self.query_one("#folder-name", Input).display = shown
        self.query_one("#folder-create", Button).display = shown
        if not shown:
            self.query_one("#folder-errors", Static).update("")

    def _create(self) -> None:
        """Make the folder inside the one shown and go into it, ready for Use this folder."""
        field = self.query_one("#folder-name", Input)
        errors = self.query_one("#folder-errors", Static)
        try:
            folder = create_folder(self._current, field.value)
        except NewFolderError as error:
            errors.update(str(error))
            return
        field.value = ""
        self._show_new_folder(False)
        self._show(folder)
        tree = self.query_one(FoldersOnly)
        tree.path = folder
        tree.focus()  # the hidden name field had it


class ModelFilesOnly(DirectoryTree):
    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [
            p
            for p in paths
            if not p.name.startswith(HIDDEN_PREFIX)
            and (p.is_dir() or p.suffix.lower() == MODEL_SUFFIX)
        ]


class FilePicker(ModalScreen[str | None]):
    """Choose a .safetensors file; dismissed with its path, or None when cancelled."""

    DEFAULT_CSS = PICKER_CSS.replace("FolderPicker", "FilePicker")
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, start: Path) -> None:
        super().__init__()
        self._start = start

    def compose(self) -> ComposeResult:
        with Vertical(id="folder-box"):
            yield Static(f"Choose a {MODEL_SUFFIX} file: Enter on it.", markup=False)
            yield ModelFilesOnly(self._start, id="folder-tree")
            with Horizontal(classes="buttons"):
                yield Button("Up", id="file-up")
                yield Button("Cancel", id="file-cancel")

    def on_mount(self) -> None:
        self.query_one(ModelFilesOnly).focus()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        event.stop()
        self.dismiss(str(event.path))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "file-up":
            tree = self.query_one(ModelFilesOnly)
            tree.path = Path(tree.path).parent
        elif event.button.id == "file-cancel":
            self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)
