"""
capmo main window: RECORD (live preview + record) plus a lightweight live
**board check**. All solving / processing is done offline by the shared library
(:mod:`capmo.core.pipeline`, the same code as the ``capmo/tools`` CLIs).

* A **dynamic** grid of camera tiles (one per connected phone, reflowing as
  phones connect / disconnect) shows the decoded live video.
* **● Record** writes a session directory (raw H.264 + per-frame timing +
  clock samples + ``manifest.json``).
* **Board check** (toggle) runs the ChArUco detector on the live frames and
  feeds :class:`capmo.core.board_coverage.LiveBoardTracker`: each tile gets an
  amber border while scanning and a green one once that camera has "good
  coverage" for ``calibrate_board`` (image coverage, distinct views, tilt,
  sharpness, simultaneous views with the other cameras), with a hint such as
  "cover the top-left" or "show to camC together with camA". Recording while
  it is on stores the board in the manifest (``board``) plus the tracker's
  verdict (``board_check``), which tells the offline pipeline to calibrate
  from the board segment. Nothing is solved live.
* **Process…** runs :func:`capmo.core.pipeline.process_session` on the last
  recording (pose keypoints -> board ``rig.json`` or body self-calibration ->
  33-joint track) in a worker thread, with progress in the status bar.
* **View** plays the processed track on the vendored character (processing
  it first if needed).

Threading: the capture server runs its own threads. Its only UI-facing
callback (a phone connected) is re-emitted as a Qt signal, so the slot runs
on the GUI thread; frames reach the GUI through the preview hub's
latest-frame cache, polled by a QTimer on the GUI thread. The ChArUco
detection of the board check runs in one scan thread (latest frame per phone,
stale ones dropped) that hands its results back through a queued signal; the
GUI thread only draws. The pipeline worker lives in a QThread and talks back
only through queued signals.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time

import numpy as np

log = logging.getLogger("capmo.ui")

try:  # cv2 is used only for the board-check overlay; degrade if absent.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)

from ..core.board import BoardConfig
from ..core.board_coverage import GOOD, IDLE, LiveBoardTracker
from ..core.presets import DEFAULT_PRESET, PRESETS

# Single resolution@fps dropdown presets: (label, width, height, fps).
QUALITY_PRESETS = [(label, w, h, f) for _, label, w, h, f in PRESETS]
DEFAULT_QUALITY = [p[0] for p in PRESETS].index(DEFAULT_PRESET)

from ..server import CaptureServer
from ..recorder import RecorderController
from ..decode import LivePreviewHub
from .. import protocol
from .character_view import CharacterView
from .workers import LatestWorker

# Vendored rig asset driven by the View flow.
_RIG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "characters",
                         "superhero_male", "superhero_male.gltf")

class CameraTile(QFrame):
    """One phone's live preview + stats + board-check readout."""

    # Persistent border colours of the board-check state per camera (kept
    # separate from the transient green flash drawn when a new view is kept).
    _BOARD_BORDERS = {
        "off": "",                        # board check off: default frame
        "idle": "#caa64a",                # amber: no usable board view yet
        "scanning": "#caa64a",            # amber: collecting coverage / views
        "good": "#46c846",                # green: good coverage for calibrate_board
    }

    def __init__(self, phone_id: str, parent=None) -> None:
        super().__init__(parent)
        self.phone_id = phone_id
        self.setFrameShape(QFrame.Shape.Box)
        self.setMinimumSize(240, 180)
        self._board_state = "off"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self._title = QLabel(phone_id, self)
        self._title.setStyleSheet("color:#9cf; font-weight:bold;")
        lay.addWidget(self._title)
        # The decoded frame is drawn here (scaled to fit).
        self._video = QLabel("waiting for video...", self)
        self._video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video.setStyleSheet("background:#16161a; color:#667;")
        self._video.setMinimumSize(160, 120)
        lay.addWidget(self._video, stretch=1)
        self._stats = QLabel("", self)
        self._stats.setStyleSheet("color:#888; font-size:10px;")
        lay.addWidget(self._stats)
        # Shown only during board check: coverage / views / pairs / hint.
        self._board = QLabel("", self)
        self._board.setWordWrap(True)
        self._board.setStyleSheet("color:#fc9; font-size:11px; font-weight:bold;")
        self._board.setVisible(False)
        lay.addWidget(self._board)

    @property
    def board_state(self) -> str:
        return self._board_state

    def board_text(self) -> str:
        return "" if self._board.isHidden() else self._board.text()

    def set_board_state(self, state: str, text: str = "") -> None:
        """Board-check state ``off`` / ``idle`` / ``scanning`` / ``good`` -> the
        tile border (amber while scanning, green when good) and the readout."""
        self._board_state = state
        color = self._BOARD_BORDERS.get(state, "")
        if color:
            tint = "#bf8" if state == GOOD else "#fc9"
            self._board.setStyleSheet(f"color:{tint}; font-size:11px; font-weight:bold;")
            self.setStyleSheet(f"CameraTile {{ border: 3px solid {color}; }}")
        else:
            self.setStyleSheet("")
        if text and state != "off":
            self._board.setText(text)
            self._board.setVisible(True)
        else:
            self._board.setVisible(False)

    def video_size(self) -> tuple[int, int]:
        sz = self._video.size()
        return sz.width(), sz.height()

    def set_frame(self, bgr: np.ndarray, overlay_text: str | None = None,
                  captured: bool = False) -> None:
        """Show one decoded frame (HxWx3 BGR uint8), scaled to the tile.

        *bgr* is never modified (it is usually the preview hub's cached frame,
        which the board detector reads too): the frame is downscaled to the tile
        first (a new array, which also makes the QImage/QPixmap work cheap) and
        the optional board-check feedback is drawn on that. *overlay_text* is
        burned into the top-left, and *captured* tints that readout green and
        draws a green border for the frame a new distinct view was kept.
        """
        h, w = bgr.shape[:2]
        tw, th = self.video_size()
        scale = min(tw / w, th / h, 1.0) if w and h else 1.0
        if cv2 is not None and scale < 1.0:
            bgr = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        elif overlay_text and cv2 is not None:
            bgr = bgr.copy()                  # about to draw: not on the caller's array
        if overlay_text and cv2 is not None:
            color = (80, 230, 80) if captured else (255, 200, 60)  # BGR
            label = ("view  " if captured else "") + overlay_text
            cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)        # outline for contrast
            cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
            if captured:
                h0, w0 = bgr.shape[:2]
                cv2.rectangle(bgr, (1, 1), (w0 - 2, h0 - 2), (80, 230, 80), 3)
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        img = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(img)
        if cv2 is None:                       # no cv2: let Qt do the scaling
            pix = pix.scaled(self._video.size(), Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        self._video.setPixmap(pix)

    def update_stats(self, label: str, fps: float, offset: float, rtt: float,
                     frames: int, ready: bool) -> None:
        self._title.setText(label)
        sync = f"off={offset:+.0f}ms rtt={rtt:.0f}ms" if ready else "syncing..."
        self._stats.setText(f"{self.phone_id} | {fps:.1f} fps | {sync} | {frames} frames")


class _ServerBridge(QObject):
    """Re-emits capture-server callbacks (server threads) as Qt signals, so the
    connected slots run on the GUI thread (queued connection)."""
    phone_connected = Signal(str)


class PipelineWorker(QObject):
    """Runs one ``process_session`` call in a QThread; reports via signals.

    ``progress(stage, detail, fraction)`` is the pipeline's callback, re-emitted
    as :attr:`progressed`; :meth:`cancel` makes the next callback raise
    :class:`capmo.core.pipeline.Cancelled`.
    """
    progressed = Signal(str)
    finished = Signal(object)          # PipelineResult
    failed = Signal(str)

    def __init__(self, fn, kwargs: dict) -> None:
        super().__init__()
        self._fn = fn
        self._kwargs = kwargs
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _progress(self, stage, detail="", fraction=None) -> None:
        from ..core.pipeline import Cancelled
        if self._cancel:
            raise Cancelled("cancelled")
        pct = f" {fraction:.0%}" if isinstance(fraction, (int, float)) else ""
        self.progressed.emit(f"{stage}: {detail}{pct}")

    @Slot()
    def run(self) -> None:
        try:
            res = self._fn(progress=self._progress, **self._kwargs)
        except Exception as exc:          # reported in the status bar, never raised
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(res)


class CapmoMainWindow(QWidget):
    closed = Signal()                  # the window really closed (see closeEvent)

    # Board detection runs at most this often per phone, so the 100 ms refresh
    # tick doesn't pay for a full ChArUco detect every time (GUI stays snappy).
    _SCAN_INTERVAL = 0.18          # s  (~5-6 Hz per phone)

    def __init__(self, host: str = "0.0.0.0", port: int = 12345,
                 record_dir: str = ".",
                 board_cfg: BoardConfig | None = None,
                 width: int = 1280, height: int = 720, fps: int = 60,
                 pipeline_fn=None) -> None:
        super().__init__()
        self.setWindowTitle("capmo")
        self.resize(1100, 720)

        self._record_dir = record_dir
        self._board_cfg = board_cfg or BoardConfig()
        # -- board check (GUI thread only) -----------------------------------
        self._board_check = False
        self._board_in_recording = False       # board check was on during the take
        self._tracker = LiveBoardTracker(self._board_cfg)
        self._scan_ts: dict[str, float] = {}   # last detection submit time per phone
        self._seen_seq: dict[str, int] = {}    # last preview frame submitted
        self._last_det: dict = {}              # newest detection result per phone
        self._det_ver: dict[str, int] = {}     # bumped with every new result
        self._drawn: dict[str, tuple] = {}     # what each tile currently shows
        self._flash_until: dict[str, float] = {}
        # The scan thread runs detector + tracker.ingest; the lock makes the
        # GUI's tracker calls (reset / statuses / summary ...) atomic against it.
        self._tracker_lock = threading.RLock()
        self._scan_gen = 0                     # bumped on reset: drops in-flight scans
        self._scanner = LatestWorker(self._scan_job, "capmo-board-scan", self)
        self._scanner.done.connect(self._on_scan_done)

        # -- offline processing (worker thread) ------------------------------
        self._pipeline_fn = pipeline_fn        # injectable (tests); default: core.pipeline
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._processed: dict[str, object] = {}   # session dir -> PipelineResult
        self._view_after_process = False
        self.last_result = None

        self._stream = (width, height, fps)    # one tuple: read atomically by the accept thread
        self._closing = False
        self._closed = False
        self._viewing = False                  # the character view replaces the grid
        self._server = CaptureServer(host=host, port=port)
        self._bridge = _ServerBridge(self)
        self._bridge.phone_connected.connect(self._on_phone_connected_gui)
        # The GUI has no separate "start streaming" step, so tell each phone to
        # START (and set its framerate) the moment it connects -- otherwise it
        # connects, answers PINGs, but never sends video.
        self._server.on_connect = self._on_phone_connect
        self._recorder = RecorderController(self._server, record_dir)
        self._preview = LivePreviewHub()
        # Feed every frame (with its host arrival time) to the live decoder.
        self._recorder.add_frame_listener(
            lambda conn, payload, recv_ms, *_: self._preview.feed(conn.phone_id, payload, recv_ms))
        self._server.start()

        outer = QVBoxLayout(self)
        bar = QHBoxLayout()
        from ..server import local_ip
        self._status = QLabel(
            f"Listening on {host}:{self._server.port} — point phones at "
            f"{local_ip()}:{self._server.port}", self)
        bar.addWidget(self._status, stretch=1)
        # One dropdown for resolution + fps; changing it re-applies to phones.
        self._quality = QComboBox(self)
        for label, *_ in QUALITY_PRESETS:
            self._quality.addItem(label)
        idx = self._match_quality(width, height, fps)
        self._quality.setCurrentIndex(idx)       # before connect -> no premature fire
        self._quality.setToolTip("Capture resolution and frame rate")
        self._quality.currentIndexChanged.connect(self._on_quality_changed)
        bar.addWidget(self._quality)
        self._apply_quality(idx, broadcast=False)
        self._board_btn = QPushButton("Board check", self)
        self._board_btn.setCheckable(True)
        self._board_btn.setToolTip(
            "Live ChArUco coverage check for the board segment (nothing is solved live); "
            "recording with it on stores the board in the session for offline calibration")
        self._board_btn.toggled.connect(self._on_board_check_toggled)
        bar.addWidget(self._board_btn)
        self._rec_btn = QPushButton("● Record", self)
        self._rec_btn.setCheckable(True)
        self._rec_btn.toggled.connect(self._on_record_toggled)
        bar.addWidget(self._rec_btn)
        self._process_btn = QPushButton("Process…", self)
        self._process_btn.setToolTip("Run the offline pipeline on the last recording")
        self._process_btn.clicked.connect(self._on_process_clicked)
        bar.addWidget(self._process_btn)
        self._view_btn = QPushButton("View", self)
        self._view_btn.setCheckable(True)
        self._view_btn.toggled.connect(self._on_view_toggled)
        bar.addWidget(self._view_btn)
        outer.addLayout(bar)

        self._grid_host = QWidget(self)
        self._grid = QGridLayout(self._grid_host)
        outer.addWidget(self._grid_host, stretch=1)
        self._tiles: dict[str, CameraTile] = {}

        # Character playback view (hidden until View runs). Software-rendered,
        # GL-free; degrades to a placeholder if the rig is missing.
        self._character = CharacterView(self)
        self._character.hide()
        outer.addWidget(self._character, stretch=1)
        self._skin = None                         # lazily loaded rig SkinData

        # Poll the (thread-owned) connection stats from the GUI thread only.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)

    # -- server callbacks ----------------------------------------------------

    def _on_phone_connect(self, conn) -> None:
        """Kick a freshly-connected phone into streaming (accept-thread context:
        only touches the socket, then hands over to the GUI thread)."""
        w, h, fps = self._stream
        conn.send(protocol.cmd_start(w, h))
        conn.send(protocol.cmd_framerate(fps))
        self._bridge.phone_connected.emit(conn.phone_id)

    @Slot(str)
    def _on_phone_connected_gui(self, phone_id: str) -> None:
        log.info("phone connected: %s", phone_id)
        if not self._viewing:
            self._refresh()                           # reflow the grid right away

    # -- quality / record ----------------------------------------------------

    @staticmethod
    def _match_quality(width: int, height: int, fps: int) -> int:
        """Index of the preset matching (w, h, fps), else the default."""
        for i, (_, w, h, f) in enumerate(QUALITY_PRESETS):
            if (w, h, f) == (width, height, fps):
                return i
        log.warning("no quality preset for %dx%d@%d; using %s", width, height, fps,
                    QUALITY_PRESETS[DEFAULT_QUALITY][0])
        return DEFAULT_QUALITY

    def _apply_quality(self, index: int, *, broadcast: bool) -> None:
        """Adopt the selected resolution@fps; optionally push it to live phones."""
        _, w, h, fps = QUALITY_PRESETS[index]
        self._stream = (w, h, fps)
        if broadcast:
            self._server.broadcast(protocol.cmd_start(w, h))
            self._server.broadcast(protocol.cmd_framerate(fps))

    def _on_quality_changed(self, index: int) -> None:
        self._apply_quality(index, broadcast=True)

    def _on_record_toggled(self, on: bool) -> None:
        if on:
            w, h, fps = self._stream
            self._board_in_recording = self._board_check
            with self._tracker_lock:
                board = self._tracker.board_cfg.to_dict() if self._board_check else None
            info = {"nominal_fps": fps, "requested_size": [w, h],
                    "preset": PRESETS[self._quality.currentIndex()][0],
                    "board": board, "recorder": {"tool": "capmo.ui"}}
            try:
                d = self._recorder.start(info=info)
            except Exception as exc:              # e.g. unwritable record dir
                log.warning("could not start recording: %s", exc)
                self._board_in_recording = False
                self._rec_btn.blockSignals(True)
                self._rec_btn.setChecked(False)
                self._rec_btn.blockSignals(False)
                self._status.setText(f"Could not start recording: {exc}")
                return
            self._rec_btn.setText("■ Stop")
            self._quality.setEnabled(False)        # a preset change would restart the encoders
            self._status.setText(f"Recording → {d}")
        else:
            if self._board_in_recording:
                # the board as finally matched (layout auto-resolve) + the verdict
                with self._tracker_lock:
                    self._recorder.update_info(board=self._tracker.board_cfg.to_dict(),
                                               board_check=self._tracker.summary())
            d = self._recorder.stop()
            self._board_in_recording = False
            self._rec_btn.setText("● Record")
            self._quality.setEnabled(True)
            self._status.setText(f"Stopped — {d}" if d else "Stopped")

    # -- board check ---------------------------------------------------------

    def _on_board_check_toggled(self, on: bool) -> None:
        self._board_check = on
        self._scan_gen += 1                    # in-flight scans belong to the old state
        if on:
            with self._tracker_lock:
                self._tracker.reset()
                self._tracker.set_cameras(list(self._tiles))
            self._scan_ts.clear()
            self._seen_seq.clear()
            self._last_det.clear()
            self._flash_until.clear()
            if self._recorder.recording:
                self._board_in_recording = True
            self._board_btn.setText("Stop board check")
            self._status.setText("Board check: walk the board through the volume — "
                                 "tiles turn green when coverage is good")
        else:
            self._board_btn.setText("Board check")
            self._status.setText("Board check off")
        self._drawn.clear()
        self._update_board_overlay()

    def _submit_scan(self, pid: str, img: np.ndarray, t_ms, seq: int) -> None:
        """Hand one new preview frame to the scan thread (throttled per phone;
        a frame still waiting there is replaced by this newer one)."""
        now = time.monotonic()
        if seq == self._seen_seq.get(pid) or now - self._scan_ts.get(pid, 0.0) < self._SCAN_INTERVAL:
            return
        self._scan_ts[pid] = now
        self._seen_seq[pid] = seq
        t = (t_ms / 1000.0) if t_ms is not None else time.time()
        self._scanner.submit(pid, (self._scan_gen, img, t, seq))

    def _scan_job(self, pid: str, job):
        """Scan thread: detect + ingest one frame. Returns ``(gen, det, accepted)``
        (None if the detection failed or was superseded by a board-check reset)."""
        gen, img, t, _seq = job
        if gen != self._scan_gen:
            return None
        with self._tracker_lock:
            detector = self._tracker.detector      # lazily built, replaced by reset()
        det = detector(img)                        # the slow part, outside the lock
        h, w = img.shape[:2]
        with self._tracker_lock:
            if gen != self._scan_gen:
                return None
            res = self._tracker.ingest(pid, t, det, (w, h), image=img)
        return gen, det, bool(res.accepted)

    @Slot(object, object)
    def _on_scan_done(self, pid: str, result) -> None:
        gen, det, accepted = result
        if gen != self._scan_gen or not self._board_check:
            return
        self._last_det[pid] = det
        self._det_ver[pid] = self._det_ver.get(pid, 0) + 1
        if accepted:
            self._flash_until[pid] = time.monotonic() + 0.4
        self._update_board_overlay()

    def _update_board_overlay(self) -> None:
        if not self._board_check:
            for tile in self._tiles.values():
                tile.set_board_state("off", "")
            return
        with self._tracker_lock:
            sts = self._tracker.statuses()
            good_views = self._tracker.th.good_views
            pairs = self._tracker.pair_counts()
        for pid, tile in self._tiles.items():
            st = sts.get(pid)
            if st is None:
                tile.set_board_state(IDLE, "board check: no board yet")
            else:
                tile.set_board_state(st.state, st.readout(good_views))
        if sts:
            good = sum(1 for s in sts.values() if s.state == GOOD)
            self._status.setText(
                f"Board check — {good}/{len(sts)} cameras good"
                + (" · " + ", ".join(f"{k}:{v}" for k, v in pairs.items()) if pairs else ""))

    def _refresh(self) -> None:
        conns = {c.phone_id: c for c in self._server.connections()}
        # Reflow the grid if the set of phones changed.
        if set(conns) != set(self._tiles):
            self._rebuild_grid(list(conns))
        for pid, conn in conns.items():
            tile = self._tiles.get(pid)
            if tile is None:
                continue
            tile.update_stats(conn.label, conn.fps, conn.clock.offset_ms,
                              conn.clock.rtt_ms, conn.frames, conn.clock.ready)
            stamped = self._preview.latest_stamped(pid)
            if stamped is None:
                continue
            img, t_ms, seq = stamped
            if not self._board_check:
                key = (seq, tile.video_size())
                if self._drawn.get(pid) != key:     # nothing new to show: skip the work
                    self._drawn[pid] = key
                    tile.set_frame(img)             # raw pass-through
                continue
            self._submit_scan(pid, img, t_ms, seq)
            shown = self._last_det.get(pid)
            flash = time.monotonic() < self._flash_until.get(pid, 0.0)
            key = (seq, tile.video_size(), self._det_ver.get(pid, 0), flash)
            if self._drawn.get(pid) == key:
                continue
            self._drawn[pid] = key
            frame = img                             # shared with the hub + scan thread:
            if shown is not None and shown.corners is not None and cv2 is not None:
                try:
                    frame = img.copy()              # ...so draw on a private copy
                    cv2.aruco.drawDetectedCornersCharuco(frame, shown.corners, shown.ids)
                except Exception:
                    frame = img
            n_markers = shown.n_markers if shown is not None else 0
            n_corners = shown.count if shown is not None else 0
            text = f"{n_markers} markers · {n_corners} corners"
            tile.set_frame(frame, overlay_text=text, captured=flash)

    def _rebuild_grid(self, phone_ids: list[str]) -> None:
        for pid in set(self._tiles) - set(phone_ids):
            self._preview.remove(pid)   # free the decoder for departed phones
        for tile in self._tiles.values():
            tile.hide()
            tile.setParent(None)
            tile.deleteLater()
        self._tiles = {}
        self._drawn.clear()
        cols = max(1, math.ceil(math.sqrt(len(phone_ids)))) if phone_ids else 1
        for i, pid in enumerate(phone_ids):
            tile = CameraTile(pid, self._grid_host)
            self._grid.addWidget(tile, i // cols, i % cols)
            self._tiles[pid] = tile
        with self._tracker_lock:
            self._tracker.set_cameras(phone_ids)
        self._update_board_overlay()

    # -- offline processing (Process… / View) --------------------------------

    def _session_for_processing(self) -> str | None:
        return self._recorder.last_session_dir

    @property
    def processing(self) -> bool:
        return self._thread is not None

    def _on_process_clicked(self) -> None:
        self.start_processing()

    def start_processing(self, session_dir: str | None = None) -> bool:
        """Run the offline pipeline on *session_dir* (default: the last
        recording) in a worker thread. False (status explains) if not possible."""
        if self.processing:
            self._status.setText("Process: already running")
            return False
        session_dir = session_dir or self._session_for_processing()
        if not session_dir:
            self._status.setText("Process: no recorded session yet — record first")
            return False
        fn = self._pipeline_fn
        if fn is None:
            from ..core.pipeline import process_session as fn
        self._worker = PipelineWorker(fn, {"session_dir": session_dir})
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(self._on_pipeline_progress)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.failed.connect(self._on_pipeline_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._process_btn.setEnabled(False)
        self._status.setText(f"Processing {session_dir} …")
        self._thread.start()
        return True

    @Slot(str)
    def _on_pipeline_progress(self, msg: str) -> None:
        self._status.setText(f"Processing — {msg}")

    @Slot(object)
    def _on_pipeline_finished(self, res) -> None:
        self.last_result = res
        self._processed[res.session_dir] = res
        self._status.setText(f"Processed ({res.rig_source} rig) → {res.joints3d_path}")
        if self._view_after_process:
            self._view_after_process = False
            self._play_result(res)

    @Slot(str)
    def _on_pipeline_failed(self, msg: str) -> None:
        self._status.setText(f"Process failed: {msg}")
        if self._view_after_process:
            self._view_after_process = False
            self._reset_view_btn()

    @Slot()
    def _on_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.wait()                # already at its end; makes deleteLater safe
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None
        self._process_btn.setEnabled(True)
        if self._closing:                      # the close was waiting for the worker
            self.close()

    # -- character view ------------------------------------------------------

    def _load_skin(self):
        """Lazily load the rig SkinData; None (with a status note) if missing."""
        if self._skin is None:
            try:
                from .. import skinning
                self._skin = skinning.load_skin(_RIG_PATH)
            except Exception as exc:              # asset/parse failure
                self._status.setText(f"View: rig asset unavailable ({exc})")
                self._skin = None
        return self._skin

    def _on_view_toggled(self, on: bool) -> None:
        """Play the processed track of the last recording (processing it first
        in the worker if needed). Toggling off returns to the camera grid."""
        if not on:
            self._view_btn.setText("View")
            self._view_after_process = False
            self._character.stop()
            self._character.hide()
            self._grid_host.show()
            self._viewing = False
            self._timer.start(100)             # live tiles again
            self._refresh()
            self._status.setText("View stopped")
            return
        session_dir = self._session_for_processing()
        res = self._processed.get(session_dir) if session_dir else None
        if res is None and self.last_result is not None and session_dir is None:
            res = self.last_result
        if res is not None:
            self._play_result(res)
            return
        if not session_dir:
            self._status.setText("View: no recorded session yet — record first")
            self._reset_view_btn()
            return
        if self._load_skin() is None:
            self._reset_view_btn()
            return
        self._view_after_process = True
        self._view_btn.setText("Processing…")
        if not self.processing and not self.start_processing(session_dir):
            self._view_after_process = False
            self._reset_view_btn()

    def _play_result(self, res) -> None:
        skin = self._load_skin()
        if skin is None:
            self._reset_view_btn()
            return
        try:
            from .. import animate
            j = res.joints3d().to_y_up()
            track = animate.animate_joints(skin, j.joints, j.valid, alpha=0.3)
        except Exception as exc:
            self._status.setText(f"View: could not load the processed track ({exc})")
            self._reset_view_btn()
            return
        if not track:
            self._status.setText("View: the processed track is empty")
            self._reset_view_btn()
            return
        if not self._view_btn.isChecked():
            self._view_btn.blockSignals(True)
            self._view_btn.setChecked(True)
            self._view_btn.blockSignals(False)
        self._view_btn.setText("Stop view")
        self._grid_host.hide()
        self._viewing = True
        self._timer.stop()                     # nothing live is drawn meanwhile
        self._character.show()
        self._character.set_track(skin, track)
        self._character.play()
        self._status.setText(f"Viewing {len(track)} frame(s) ({res.rig_source} rig)")

    def _reset_view_btn(self) -> None:
        """Pop the View button back off without re-running the toggle handler
        (so a failed-start status message survives)."""
        self._view_btn.blockSignals(True)
        self._view_btn.setChecked(False)
        self._view_btn.setText("View")
        self._view_btn.blockSignals(False)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Close at once -- unless the Process worker is still running: then
        cancel it, hide the window and finish the close from the worker's
        ``finished`` (:meth:`_on_thread_finished`), so the GUI never blocks on
        it and no running QThread is ever destroyed."""
        if self._closed:
            event.accept()
            return
        if not self._closing:
            self._closing = True
            self._timer.stop()
            self._character.shutdown()
            if self._rec_btn.isChecked():
                self._rec_btn.setChecked(False)   # same stop path as the button (board verdict)
            self._scanner.stop()
            self._server.stop()
            self._preview.stop()
        if self._thread is not None:
            if self._worker is not None:
                self._worker.cancel()
            self.hide()
            event.ignore()
            return
        self._closed = True
        super().closeEvent(event)
        self.closed.emit()
