"""Grid of image thumbnails with row and column labels; double-click opens the image.

Thumbnails are cached by path and modification time, so polling the preview folder does not
reload every image each time; the cache keeps at most `max_thumbnails`, dropping the oldest.
"""

from collections.abc import Mapping
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QTableWidget, QTableWidgetItem, QWidget

THUMBNAIL_SIDE = 192
MAX_THUMBNAILS = 256  # about 256 * 192 * 192 * 4 bytes, under 40 MB
CELL_MARGIN = 8
PATH_ROLE = Qt.ItemDataRole.UserRole


class ImageGrid(QTableWidget):
    image_activated = Signal(object)  # Path

    def __init__(
        self,
        thumbnail_side: int = THUMBNAIL_SIDE,
        parent: QWidget | None = None,
        max_thumbnails: int = MAX_THUMBNAILS,
    ) -> None:
        super().__init__(parent)
        self._side = thumbnail_side
        self._max_thumbnails = max_thumbnails
        # Insertion order is age: the first key is the oldest (widget state, mutated in place).
        self._thumbnails: dict[tuple[Path, int], QIcon] = {}
        self.setIconSize(QSize(thumbnail_side, thumbnail_side))
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.horizontalHeader().setDefaultSectionSize(thumbnail_side + CELL_MARGIN)
        self.verticalHeader().setDefaultSectionSize(thumbnail_side + CELL_MARGIN)
        self.cellDoubleClicked.connect(self._on_double_click)

    def set_grid(
        self,
        rows: tuple[str, ...],
        columns: tuple[str, ...],
        cells: Mapping[tuple[int, int], Path],
    ) -> None:
        self.clear()
        self.setRowCount(len(rows))
        self.setColumnCount(len(columns))
        self.setVerticalHeaderLabels(list(rows))
        self.setHorizontalHeaderLabels(list(columns))
        for (row, column), path in cells.items():
            self.setItem(row, column, self._item(path))

    def set_cell(self, row: int, column: int, path: Path) -> None:
        self.setItem(row, column, self._item(path))

    def cached_thumbnails(self) -> int:
        return len(self._thumbnails)

    def is_cached(self, path: Path) -> bool:
        return any(cached == path for cached, _ in self._thumbnails)

    def image_at(self, row: int, column: int) -> Path | None:
        item = self.item(row, column)
        return None if item is None else item.data(PATH_ROLE)

    def _item(self, path: Path) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setIcon(self._thumbnail(path))
        item.setToolTip(str(path))
        item.setData(PATH_ROLE, path)
        return item

    def _thumbnail(self, path: Path) -> QIcon:
        try:
            key = (path, path.stat().st_mtime_ns)
        except OSError:
            return QIcon()
        cached = self._thumbnails.get(key)
        if cached is None:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                # Half written (sd-scripts saves in place) or unreadable: not cached, so the
                # next set_grid tries again.
                return QIcon()
            scaled = pixmap.scaled(
                self._side,
                self._side,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            cached = QIcon(scaled)
            self._thumbnails[key] = cached
            while len(self._thumbnails) > self._max_thumbnails:
                del self._thumbnails[next(iter(self._thumbnails))]
        return cached

    def _on_double_click(self, row: int, column: int) -> None:
        path = self.image_at(row, column)
        if path is not None:
            self.image_activated.emit(path)
