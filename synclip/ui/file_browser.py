"""
FileBrowser - left panel showing directories and audio files.
"""

from __future__ import annotations

import os

from PySide6.QtCore import (
    QDir,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
# QFileSystemModel moved between modules across Qt/PySide6 versions
# (QtWidgets in Qt5/older PySide6, QtGui in Qt6). Import from wherever it lives.
try:
    from PySide6.QtGui import QFileSystemModel
except ImportError:
    from PySide6.QtWidgets import QFileSystemModel
from PySide6.QtWidgets import QMenu, QTreeView, QVBoxLayout, QWidget

_AUDIO_EXTS = {".wav", ".ogg", ".mp3"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_MEDIA_EXTS = _AUDIO_EXTS | _VIDEO_EXTS


class _AudioFilterProxy(QSortFilterProxyModel):
    """Show only directories and audio files."""

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        source_model = self.sourceModel()
        index = source_model.index(source_row, 0, source_parent)
        if source_model.isDir(index):
            return True
        name = source_model.fileName(index)
        _, ext = os.path.splitext(name)
        return ext.lower() in _MEDIA_EXTS


class FileBrowser(QWidget):
    """Left panel: directory tree filtered to audio files."""

    file_selected = Signal(str)  # full path of selected audio or video file
    use_as_video = Signal(str)   # right-click -> "Use as Video Input"
    use_as_audio = Signal(str)   # right-click -> "Use as Audio Input"

    def __init__(self, root_dir: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._root_dir = root_dir

        self._fs_model = QFileSystemModel(self)
        self._fs_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self._fs_model.setRootPath(root_dir)

        self._proxy = _AudioFilterProxy(self)
        self._proxy.setSourceModel(self._fs_model)

        self._tree = QTreeView(self)
        self._tree.setModel(self._proxy)
        self._tree.setRootIndex(self._proxy.mapFromSource(self._fs_model.index(root_dir)))

        # Hide size / type / modified columns - only Name
        for col in range(1, self._fs_model.columnCount()):
            self._tree.hideColumn(col)
        self._tree.setHeaderHidden(True)
        self._tree.setAnimated(False)
        self._tree.setIndentation(16)

        self._tree.clicked.connect(self._on_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_root(self, path: str) -> None:
        self._root_dir = path
        self._fs_model.setRootPath(path)
        self._tree.setRootIndex(
            self._proxy.mapFromSource(self._fs_model.index(path))
        )

    def current_media_path(self) -> str | None:
        """Return full path of the currently selected audio or video file."""
        idx = self._tree.currentIndex()
        if not idx.isValid():
            return None
        src_idx = self._proxy.mapToSource(idx)
        if self._fs_model.isDir(src_idx):
            return None
        path = os.path.normpath(self._fs_model.filePath(src_idx))
        _, ext = os.path.splitext(path)
        if ext.lower() in _MEDIA_EXTS:
            return path
        return None

    # Keep old name as alias for callers that still use it.
    def current_audio_path(self) -> str | None:
        return self.current_media_path()

    def select_file(self, path: str) -> None:
        """Highlight *path* in the tree without emitting file_selected."""
        src_idx = self._fs_model.index(path)
        if not src_idx.isValid():
            return
        proxy_idx = self._proxy.mapFromSource(src_idx)
        if proxy_idx.isValid():
            self._tree.setCurrentIndex(proxy_idx)
            self._tree.scrollTo(proxy_idx)

    def select_adjacent(self, delta: int) -> None:
        """Select the next (+1) or previous (-1) audio file in the current directory."""
        current_path = self.current_audio_path()

        # Determine current directory
        cur_idx = self._tree.currentIndex()
        if not cur_idx.isValid():
            # Try root
            cur_dir_src = self._fs_model.index(self._root_dir)
        else:
            src_idx = self._proxy.mapToSource(cur_idx)
            if self._fs_model.isDir(src_idx):
                cur_dir_src = src_idx
            else:
                cur_dir_src = self._fs_model.parent(src_idx)

        # Collect audio files in the directory (via proxy)
        cur_dir_proxy = self._proxy.mapFromSource(cur_dir_src)
        audio_proxy_indices: list[QModelIndex] = []
        row_count = self._proxy.rowCount(cur_dir_proxy)
        for row in range(row_count):
            child_proxy = self._proxy.index(row, 0, cur_dir_proxy)
            child_src = self._proxy.mapToSource(child_proxy)
            if not self._fs_model.isDir(child_src):
                _, ext = os.path.splitext(self._fs_model.fileName(child_src))
                if ext.lower() in _MEDIA_EXTS:
                    audio_proxy_indices.append(child_proxy)

        if not audio_proxy_indices:
            return

        # Find current position
        current_row = -1
        if current_path:
            for i, proxy_idx in enumerate(audio_proxy_indices):
                src_idx = self._proxy.mapToSource(proxy_idx)
                if self._fs_model.filePath(src_idx) == current_path:
                    current_row = i
                    break

        new_row = current_row + delta
        new_row = max(0, min(new_row, len(audio_proxy_indices) - 1))
        new_index = audio_proxy_indices[new_row]
        self._tree.setCurrentIndex(new_index)
        self._tree.scrollTo(new_index)
        self._on_clicked(new_index)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_clicked(self, proxy_index: QModelIndex) -> None:
        src_idx = self._proxy.mapToSource(proxy_index)
        if self._fs_model.isDir(src_idx):
            return
        path = self._fs_model.filePath(src_idx)
        _, ext = os.path.splitext(path)
        if ext.lower() in _MEDIA_EXTS:
            self.file_selected.emit(path)

    def _on_context_menu(self, pos) -> None:
        """Right-click a media file: offer it as a video or audio input."""
        idx = self._tree.indexAt(pos)
        if not idx.isValid():
            return
        src_idx = self._proxy.mapToSource(idx)
        if self._fs_model.isDir(src_idx):
            return
        path = self._fs_model.filePath(src_idx)
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in _MEDIA_EXTS:
            return

        menu = QMenu(self)
        if ext in _VIDEO_EXTS:
            act_v = menu.addAction("Use as Video Input")
            act_v.triggered.connect(lambda: self.use_as_video.emit(path))
        # Both video (has audio track) and audio files can be audio input.
        act_a = menu.addAction("Use as Audio Input")
        act_a.triggered.connect(lambda: self.use_as_audio.emit(path))
        menu.exec(self._tree.viewport().mapToGlobal(pos))
