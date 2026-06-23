"""
TakesPanel - right panel showing list of takes for the current audio file.

Row 0 is always a synthetic "LIVE" entry that returns the app to idle/live mode.
Clicking any take row (always, even the same take) emits take_selected so the
main window can restart playback from the beginning.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_LIVE_LABEL = "LIVE"


class _TakeList(QListWidget):
    """QListWidget that records, per mouse click, whether the click changed the
    current row. Lets the panel tell a row-changing click (which already emits
    via currentRowChanged) apart from a re-click on the already-selected row
    (which must emit explicitly to restart the take), without double-emitting."""

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._row_changed_this_click = False
        super().mousePressEvent(event)


def _take_label(take: dict) -> str:
    """Format a display label for a take dict."""
    take_id = take.get("take_id", "?")
    name = take.get("name", "").strip()
    is_default = take.get("is_default", False)
    ts = take.get("timestamp_utc", "")
    display_ts = ts.replace("T", " ").rstrip("Z")[:16] if ts else ""
    star = " *" if is_default else ""
    display_name = name if name else take_id
    if display_ts:
        return f"{display_name}{star}  ({display_ts})"
    return f"{display_name}{star}"


class TakesPanel(QWidget):
    """Right panel: list of recorded takes for the current audio file.

    Row 0 is always "LIVE". Clicking it emits *live_selected*. Clicking any
    other row emits *take_selected* - even re-clicking the already-selected
    row - so the main window can restart the take from the beginning.
    """

    take_selected = Signal(str)        # take_id
    live_selected = Signal()           # user chose the LIVE entry
    take_set_default = Signal(str)     # take_id
    take_deleted = Signal(str)         # take_id
    take_renamed = Signal(str, str)    # take_id, new_name

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._takes: list[dict] = []
        self._loading = False  # suppress signals while repopulating the list

        header = QLabel("Takes")
        header.setStyleSheet("font-weight: bold; padding: 4px;")

        self._list = _TakeList(self)
        self._list._row_changed_this_click = False
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        # currentRowChanged fires for keyboard Up/Down navigation and for the
        # first selection of a new row by mouse.
        self._list.currentRowChanged.connect(self._on_row_changed)
        # itemClicked fires even when the same row is clicked again (re-click to
        # restart a take); only that case is handled here.
        self._list.itemClicked.connect(self._on_item_clicked)

        self._btn_rename = QPushButton("Rename...", self)
        self._btn_default = QPushButton("Set Default", self)
        self._btn_delete = QPushButton("Delete", self)
        self._btn_rename.clicked.connect(self._on_rename)
        self._btn_default.clicked.connect(self._on_set_default)
        self._btn_delete.clicked.connect(self._on_delete)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_rename)
        btn_row.addWidget(self._btn_default)
        btn_row.addWidget(self._btn_delete)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(header)
        layout.addWidget(self._list)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_takes(self, data: dict | None) -> None:
        """Repopulate the list from a synclip data dict (or just LIVE if None)."""
        self._loading = True
        try:
            self._takes = []
            self._list.clear()
            # Row 0: synthetic LIVE entry
            self._list.addItem(QListWidgetItem(_LIVE_LABEL))
            if data is not None:
                self._takes = list(data.get("takes", []))
                for take in self._takes:
                    self._list.addItem(QListWidgetItem(_take_label(take)))
            # Default selection: last take, or LIVE if no takes
            if self._list.count() > 1:
                self._list.setCurrentRow(self._list.count() - 1)
            else:
                self._list.setCurrentRow(0)
        finally:
            self._loading = False
        self._update_buttons()

    def current_take_id(self) -> str | None:
        row = self._list.currentRow()
        if row <= 0 or row - 1 >= len(self._takes):
            return None
        return self._takes[row - 1].get("take_id")

    def select_live(self) -> None:
        """Programmatically select the LIVE row (does NOT emit live_selected)."""
        self._loading = True
        try:
            self._list.setCurrentRow(0)
        finally:
            self._loading = False

    def select_take(self, take_id: str) -> None:
        """Programmatically highlight a take's row (does NOT emit a signal)."""
        for i, t in enumerate(self._takes):
            if t.get("take_id") == take_id:
                self._loading = True
                try:
                    self._list.setCurrentRow(i + 1)  # +1 for the LIVE row
                finally:
                    self._loading = False
                return

    def select_adjacent(self, delta: int) -> None:
        count = self._list.count()
        if count == 0:
            return
        row = self._list.currentRow()
        new_row = max(0, min(row + delta, count - 1))
        if new_row != row:
            self._list.setCurrentRow(new_row)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit_for_row(self, row: int) -> None:
        if self._loading:
            return
        if row == 0:
            self.live_selected.emit()
        elif 0 < row <= len(self._takes):
            take_id = self._takes[row - 1].get("take_id", "")
            if take_id:
                self.take_selected.emit(take_id)

    def _update_buttons(self) -> None:
        # Set Default / Delete only make sense for a real take (not the LIVE row).
        has_take = self.current_take_id() is not None
        self._btn_rename.setEnabled(has_take)
        self._btn_default.setEnabled(has_take)
        self._btn_delete.setEnabled(has_take)

    def _on_row_changed(self, row: int) -> None:
        # Mark that this mouse click (if any) changed the row, so the following
        # itemClicked won't emit a second time for the same click.
        self._list._row_changed_this_click = True
        self._update_buttons()
        self._emit_for_row(row)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        # Only emit here for a re-click on the already-selected row: in that case
        # currentRowChanged did NOT fire, so _row_changed_this_click stays False.
        if not self._list._row_changed_this_click:
            self._emit_for_row(self._list.row(item))

    def _on_rename(self) -> None:
        take_id = self.current_take_id()
        if not take_id:
            return
        row = self._list.currentRow()
        current_name = self._takes[row - 1].get("name", "") if row > 0 else ""
        name, ok = QInputDialog.getText(
            self, "Rename Take", "Name:", text=current_name
        )
        if ok:
            self.take_renamed.emit(take_id, name)

    def _on_set_default(self) -> None:
        take_id = self.current_take_id()
        if take_id:
            self.take_set_default.emit(take_id)

    def _on_delete(self) -> None:
        take_id = self.current_take_id()
        if take_id:
            self.take_deleted.emit(take_id)

    def _show_context_menu(self, pos) -> None:
        take_id = self.current_take_id()
        if not take_id:
            return
        menu = QMenu(self)
        act_rename = menu.addAction("Rename...")
        act_default = menu.addAction("Set Default")
        act_delete = menu.addAction("Delete")
        action = menu.exec(self._list.mapToGlobal(pos))
        if action == act_rename:
            self._on_rename()
        elif action == act_default:
            self.take_set_default.emit(take_id)
        elif action == act_delete:
            self.take_deleted.emit(take_id)
