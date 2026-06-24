"""
MainWindow - main application window driven by a single AppStateMachine.

The window owns one AppStateMachine (the authoritative state) and one bridge
method, _apply(plan), which is the ONLY place that pushes state into the
subsystems (capture worker, audio player, IPC broadcast, UI widgets). Every UI
handler asks the machine for a Plan and applies it.

Four modes (see app_state.Mode and DESIGN.md):
    LIVE          live preview; MediaPipe + broadcast iff Broadcast is on.
    RECORD_VIDEO  write webcam frames to a file (no analysis).
    PROCESS_VIDEO offline-analyse a video file into a take, streaming live.
    REVIEW        play back a finished take.
"""

from __future__ import annotations

import bisect
import collections
import os
import tempfile
import time

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# QAction lives in QtGui on Qt6 but QtWidgets on older PySide6 builds.
try:
    from PySide6.QtGui import QAction, QActionGroup, QShortcut, QKeySequence
except ImportError:
    from PySide6.QtWidgets import QAction, QActionGroup
    from PySide6.QtGui import QShortcut, QKeySequence

from ..app_state import (
    AppStateMachine,
    AudioKind,
    AudioSource,
    Mode,
    Plan,
    VideoSource,
)
from ..audio_player import AudioPlayer
from ..capture_worker import CaptureWorker
from ..ipc_server import IPCServer, MODE_LIVE, MODE_REVIEW, MODE_EDIT
from .. import data as data_mod
from .file_browser import FileBrowser
from .webcam_view import WebcamView
from .takes_panel import TakesPanel
from .value_editor import ValueEditor
from .quad_view import QuadView, MESH_VIEW_COUNT
from ..view_pipeline import ViewConfig, ViewPipeline
from .. import ui_config as ui_config_mod
from ..head_mesh import load_head_mesh
from .. import modifiers as mod_mod
from .modifier_stack import ModifierStackWidget
from ..retarget import RetargetConfig, retarget_stream
from ..data import Stream, StreamStore, interp_blendshapes, interp_head_pose

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac"}
_MESH_EXTS = {".glb", ".gltf"}

# Preview view index for the broadcast output.
OUTPUT_VIEW = 1


def _default_head_mesh() -> str | None:
    """Locate the bundled head.glb (override with SYNCLIP_HEAD_MESH)."""
    env = os.environ.get("SYNCLIP_HEAD_MESH")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, "..", "..", "godot", "data", "head.glb"))
    return cand if os.path.isfile(cand) else None


# ---------------------------------------------------------------------------
# Interpolation helpers - thin aliases for the shared implementations in
# data so call sites in this file don't need renaming.
# ---------------------------------------------------------------------------

_interp_blendshapes = interp_blendshapes
_interp_head_pose = interp_head_pose


# ---------------------------------------------------------------------------
# Camera Settings dialog (replaces probing)
# ---------------------------------------------------------------------------


class CameraSettingsDialog(QDialog):
    """Pick a capture resolution + frame rate. We only *request* the mode; the
    driver clamps to what it supports (no enumeration / preview stall)."""

    _RESOLUTIONS = [("720p", 1280, 720), ("1080p", 1920, 1080), ("2160p", 3840, 2160)]
    _RATES = [30, 60, 120]

    def __init__(self, current: tuple[int, int, int] | None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera Settings")
        layout = QVBoxLayout(self)

        cur_w, cur_h, cur_fps = current or (1920, 1080, 60)

        res_box = QGroupBox("Resolution", self)
        res_layout = QHBoxLayout(res_box)
        self._res_buttons: list[tuple[QRadioButton, int, int]] = []
        for label, w, h in self._RESOLUTIONS:
            rb = QRadioButton(label, res_box)
            rb.setChecked(w == cur_w and h == cur_h)
            res_layout.addWidget(rb)
            self._res_buttons.append((rb, w, h))
        if not any(rb.isChecked() for rb, _, _ in self._res_buttons):
            self._res_buttons[1][0].setChecked(True)  # default 1080p
        layout.addWidget(res_box)

        rate_box = QGroupBox("Frame rate", self)
        rate_layout = QHBoxLayout(rate_box)
        self._rate_buttons: list[tuple[QRadioButton, int]] = []
        for fps in self._RATES:
            rb = QRadioButton(f"{fps} fps", rate_box)
            rb.setChecked(fps == cur_fps)
            rate_layout.addWidget(rb)
            self._rate_buttons.append((rb, fps))
        if not any(rb.isChecked() for rb, _ in self._rate_buttons):
            self._rate_buttons[1][0].setChecked(True)  # default 60
        layout.addWidget(rate_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_mode(self) -> tuple[int, int, int]:
        w = h = 0
        for rb, rw, rh in self._res_buttons:
            if rb.isChecked():
                w, h = rw, rh
        fps = 60
        for rb, rfps in self._rate_buttons:
            if rb.isChecked():
                fps = rfps
        return (w, h, fps)


# ---------------------------------------------------------------------------
# Retarget Settings dialog
# ---------------------------------------------------------------------------


class RetargetSettingsDialog(QDialog):
    """Configure analysis-by-synthesis retargeting parameters."""

    def __init__(self, cfg: RetargetConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Retarget Settings")
        lay = QVBoxLayout(self)

        def _row(label, widget):
            r = QHBoxLayout()
            r.addWidget(QLabel(label))
            r.addWidget(widget, stretch=1)
            return r

        # Max iterations
        self._iter_slider = QSlider(Qt.Orientation.Horizontal)
        self._iter_slider.setRange(1, 20)
        self._iter_slider.setValue(cfg.max_iter)
        self._iter_label = QLabel(str(cfg.max_iter))
        self._iter_slider.valueChanged.connect(lambda v: self._iter_label.setText(str(v)))
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Max iterations:"))
        iter_row.addWidget(self._iter_slider, stretch=1)
        iter_row.addWidget(self._iter_label)
        lay.addLayout(iter_row)

        # Tolerance
        self._tol_slider = QSlider(Qt.Orientation.Horizontal)
        self._tol_slider.setRange(1, 20)   # 0.005 .. 0.10 (step 0.005)
        self._tol_slider.setValue(max(1, int(round(cfg.tolerance / 0.005))))
        self._tol_label = QLabel(f"{cfg.tolerance:.3f}")
        self._tol_slider.valueChanged.connect(
            lambda v: self._tol_label.setText(f"{v * 0.005:.3f}")
        )
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Tolerance:"))
        tol_row.addWidget(self._tol_slider, stretch=1)
        tol_row.addWidget(self._tol_label)
        lay.addLayout(tol_row)

        # Gain
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(1, 10)   # 0.1 .. 1.0
        self._gain_slider.setValue(max(1, int(round(cfg.gain * 10))))
        self._gain_label = QLabel(f"{cfg.gain:.1f}")
        self._gain_slider.valueChanged.connect(
            lambda v: self._gain_label.setText(f"{v * 0.1:.1f}")
        )
        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Gain:"))
        gain_row.addWidget(self._gain_slider, stretch=1)
        gain_row.addWidget(self._gain_label)
        lay.addLayout(gain_row)

        # Scope
        scope_box = QGroupBox("Channel scope")
        sb_lay = QHBoxLayout(scope_box)
        self._scope_mouth = QRadioButton("Mouth only")
        self._scope_all = QRadioButton("All channels")
        if cfg.scope == "all":
            self._scope_all.setChecked(True)
        else:
            self._scope_mouth.setChecked(True)
        sb_lay.addWidget(self._scope_mouth)
        sb_lay.addWidget(self._scope_all)
        lay.addWidget(scope_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def result_config(self) -> RetargetConfig:
        cfg = RetargetConfig()
        cfg.max_iter = self._iter_slider.value()
        cfg.tolerance = self._tol_slider.value() * 0.005
        cfg.gain = self._gain_slider.value() * 0.1
        cfg.scope = "all" if self._scope_all.isChecked() else "mouth"
        return cfg


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Top-level application window, driven by AppStateMachine + _apply(plan)."""

    def __init__(self, root_dir: str, video_source: int | str = 0,
                 ipc_port: int = 9876, ipc_host: str = "127.0.0.1") -> None:
        super().__init__()

        self._root_dir = root_dir
        self._ipc_host = ipc_host
        self._ipc_port = ipc_port
        self._init_complete = False  # guards _save_ui_config during startup

        # --- Authoritative state ---
        self._sm = AppStateMachine()

        # Window-level mirrors of the chosen sources (kept in sync with the SM).
        self._video_source: int | str = video_source   # camera index or file path
        self._video_path: str | None = None             # set when source is a file
        self._video_audio_path: str | None = None       # extracted track (temp)
        self._audio_path: str | None = None             # user-chosen audio file

        # Broadcast routing, set by _apply() from the Plan.
        self._broadcast_enabled: bool = True
        self._broadcast_mode: int = MODE_LIVE
        self._current_plan: Plan | None = None

        # Audio bookkeeping.
        self._current_audio_path: str | None = None
        self._audio_loaded_path: str | None = None

        # Take / REVIEW state.
        self._current_synclip: dict | None = None
        self._current_take_id: str | None = None
        # Last values broadcast during REVIEW playback, so we only stream on an
        # actual change (and never while paused).
        self._last_review_values: list[float] | None = None
        # Modifier presets saved to ui.config.
        self._presets: dict[str, list[dict]] = {}

        # All blendshape streams (mediapipe, ai, retarget, ...).
        self._streams = StreamStore()
        self._ai_gen_for_audio: str | None = None

        # Per-view preview pipelines (the 3 mesh quads). Each has an independent
        # AI-mix / curves / smoothing config; exactly one is the broadcast output.
        self._setup_views()

        # PROCESS_VIDEO accumulation.
        self._recorded_frames: list[dict] = []
        self._record_frame_index: int = 0
        self._process_frac: float = 0.0

        self._extract_worker = None

        self._last_raw_blendshapes: list[float] | None = None
        self._last_head_pose: dict | None = None

        # FPS measurement.
        self._frame_times: collections.deque[float] = collections.deque(maxlen=30)
        self._current_fps: float = 0.0

        self._live_settings: dict | None = None
        self._ipc_label_cache: str = ""

        # Transient drag flag (set/cleared only via begin/end scrub).
        self._scrubbing = False
        # User toggle for the landmark overlay (AND-ed with the plan).
        self._landmarks_user_visible = True
        # Camera mode (w, h, fps) chosen via Camera Settings.
        self._current_mode: tuple[int, int, int] | None = None
        # Webcam recording temp paths.
        self._webcam_raw_path: str | None = None
        self._webcam_final_path: str | None = None
        self._webcam_audio_path: str | None = None
        self._webcam_audio_start_ms: float = 0.0

        # Analysis-by-synthesis retargeting (video only).
        self._retarget_cfg = RetargetConfig()

        self._setup_backend()
        self._setup_ui()
        self._setup_toolbar()
        self._setup_tuning_toolbar()
        self._setup_menus()
        self._setup_shortcuts()
        self._setup_timer()

        # Seed the state machine with the initial source and apply the LIVE plan.
        self._sm.set_video_source(self._make_video_source())
        self._sm.set_audio_source(self._make_audio_source())
        self._apply(self._sm.plan())

        # Default the video input to the first video file in the directory (if
        # any) rather than the webcam, so opening a project lands on its footage.
        # Skip this when ui.config already names a source to restore, so the
        # auto-default doesn't load a video the restore then has to undo.
        saved = ui_config_mod.load(self._root_dir)
        has_saved_source = bool(saved.get("audio_path")) or isinstance(
            saved.get("video_source"), str)
        videos, _audios = self._list_dir_media()
        if videos and not isinstance(self._video_source, str) and not has_saved_source:
            self._load_video_file(videos[0])
        else:
            self._rebuild_input_dropdowns()

        self._update_title()
        self._update_status()

        # Restore persisted UI state (after widgets and sources are ready).
        self._restore_ui_config()
        self._init_complete = True

    # ==================================================================
    # Per-view preview pipelines
    # ==================================================================

    def _setup_views(self) -> None:
        """Build the 3 preview ViewConfigs/pipelines and the shared mesh cache."""
        default_mesh = _default_head_mesh()
        self._views: list[ViewConfig] = [
            ViewConfig(label="Raw results", mesh_path=default_mesh),
            ViewConfig(label="Output (sent)", mesh_path=default_mesh, is_output=True),
            ViewConfig(label="Model B", mesh_path=default_mesh),
        ]
        self._pipelines = [ViewPipeline(cfg) for cfg in self._views]
        self._output_index = OUTPUT_VIEW
        self._selected_view = OUTPUT_VIEW
        self._mesh_cache: dict[str, object] = {}

    def _get_mesh(self, path: str | None):
        """Load (and cache) a head mesh; None on any failure."""
        if not path:
            return None
        if path not in self._mesh_cache:
            try:
                self._mesh_cache[path] = load_head_mesh(path)
            except Exception as exc:
                print(f"[main_window] failed to load mesh {path}: {exc}")
                self._mesh_cache[path] = None
        return self._mesh_cache[path]

    def _on_camera_changed(self, index: int) -> None:
        self._views[index].camera = self._quad.get_camera(index)
        self._save_ui_config()

    def _view_title(self, cfg) -> tuple[str, str]:
        star = " *" if cfg.is_output else ""
        names = []
        for mc in cfg.modifiers:
            cls = mod_mod.get_class(mc.type)
            if cls is None:
                continue
            tag = cls.display_name
            if not mc.enabled:
                tag = f"({tag})"
            names.append(tag)
        chain = " -> ".join(names) if names else "mediapipe"
        return cfg.label + star, chain

    def _load_view_meshes(self) -> None:
        """Push each view's mesh into its renderer (called once the quad exists)."""
        for i, cfg in enumerate(self._views):
            self._quad.set_mesh(i, self._get_mesh(cfg.mesh_path))
            self._quad.set_title(i, *self._view_title(cfg))
            if cfg.camera:
                self._quad.set_camera(i, cfg.camera)

    def _track_names(self) -> list[str]:
        """Every available track, mediapipe first (the always-present base),
        then each other loaded stream. The single source of truth for the input
        modifier's stream picker and the bottom editor panel."""
        rest = [n for n in self._streams.names() if n != "mediapipe"]
        return ["mediapipe"] + rest

    def _build_streams(self, raw: list[float], ai_vals: list[float] | None,
                       pos_ms: float) -> dict:
        """The named sample streams available to the modifier stacks at *pos_ms*.

        Every loaded stream is sampled dynamically; mediapipe and ai are then
        overridden with the live/edited values the caller passes in."""
        streams = self._streams.sample_all(pos_ms)
        streams["mediapipe"] = raw
        if ai_vals is not None:
            streams["ai"] = ai_vals
        return streams

    def _update_pipeline_streams(self) -> None:
        """Push the full take/AI/retarget streams into every pipeline so each
        modifier can run its whole-timeline preparation (e.g. closure
        detection)."""
        if not hasattr(self, "_pipelines"):
            return
        for pipe in self._pipelines:
            pipe.set_streams(self._streams.prepare_all())
        # Keep the modifier editors' stream pickers in sync with the live tracks.
        if hasattr(self, "_modifier_stack"):
            self._modifier_stack.set_available_streams(self._track_names())
        self._refresh_modifier_status()

    def _drive_previews(self, raw: list[float], ai_vals: list[float] | None,
                        pose: dict | None = None, pos_ms: float = 0.0) -> list[float]:
        """Run every view pipeline over the streams, update the quad and bars,
        return the output view's weights."""
        streams = self._build_streams(raw, ai_vals, pos_ms)
        duration_ms = self._audio.duration_ms or 0.0
        out = list(raw)
        selected_mix: list[float] = list(raw)
        for i, pipe in enumerate(self._pipelines):
            w, view_pose = pipe.process(streams, pose, pos_ms, duration_ms)
            if hasattr(self, "_quad"):
                self._quad.set_weights(i, w)
                self._quad.set_head_pose(i, view_pose)
            if i == self._output_index:
                out = w
            if i == self._selected_view:
                selected_mix = w
                self._update_selected_influence(pipe, streams, pos_ms, duration_ms)
        # Keep the docked influence curve's playhead aligned with the clip.
        self._update_influence_playhead(pos_ms)
        # Update bars according to the selected source. "mix" shows the selected
        # view's pipeline output; any other entry is a track read straight from
        # the (dynamically sampled) stream set -- no per-backend branches.
        if hasattr(self, "_bars_source_combo"):
            src = self._bars_source_combo.currentData()
            if src == "mix":
                self._bars.set_values(selected_mix)
            else:
                vals = streams.get(src)
                self._bars.set_values(list(vals) if vals is not None else [0.0] * 52)
        return out

    def _update_selected_influence(self, pipe, streams, pos_ms: float,
                                   duration_ms: float) -> None:
        """Push each modifier's live effective influence to the bound stack rows
        (shown as the absolute-mode readout)."""
        if not hasattr(self, "_modifier_stack"):
            return
        ctx = mod_mod.ModifierContext(streams=streams, pos_ms=pos_ms,
                                      duration_ms=duration_ms)
        for ri, m in enumerate(pipe._modifiers):
            self._modifier_stack.set_row_influence(ri, m.effective_influence(ctx))

    def _reset_pipelines(self) -> None:
        for pipe in self._pipelines:
            pipe.reset()

    # ==================================================================
    # The single bridge: Plan -> subsystems
    # ==================================================================

    def _apply(self, plan: Plan) -> None:
        """The ONLY place that pushes state into subsystems."""
        self._current_plan = plan

        # 1. Capture worker.
        self._worker.configure(
            source=self._video_source,
            run_mediapipe=plan.run_mediapipe,
            throttle_fps=plan.throttle_fps,
            video_looping=plan.video_looping,
            paused=plan.paused,
            record_path=plan.record_path,
            audio_path=self._playback_audio_path(),
        )

        # 2. Broadcast routing (read by _emit_frame / the poll loop).
        self._broadcast_enabled = plan.broadcast
        self._broadcast_mode = plan.broadcast_mode

        # 3. Audio.
        self._sync_audio(plan)

        # 4. Landmark overlay (mode gate AND user toggle).
        self._webcam_view.set_overlay_visible(
            plan.show_landmarks and self._landmarks_user_visible
        )

        # 5. UI widgets.
        self._sync_widgets(plan)

    def _sync_audio(self, plan: Plan) -> None:
        """Bring the audio player in line with the Plan."""
        if plan.audio_playing:
            path = self._playback_audio_path()
            if not path:
                return
            if self._audio_loaded_path != path:
                try:
                    self._audio.load(path)
                    self._audio_loaded_path = path
                    self._audio.play_loop()
                except Exception:
                    pass
            elif self._audio.is_paused():
                self._audio.unpause()
            elif not self._audio.is_playing():
                self._audio.play_loop()
        else:
            if plan.paused:
                self._audio.pause()
            elif plan.mode == Mode.PROCESS_VIDEO:
                self._audio.stop()
            else:
                # MIC/NONE in LIVE, or scrubbing in REVIEW: hold quietly.
                if self._audio.is_playing():
                    self._audio.pause()

    def _sync_widgets(self, plan: Plan) -> None:
        """Enable/disable + relabel widgets for the current Plan."""
        mode = plan.mode
        labels = {
            Mode.LIVE: "LIVE",
            Mode.RECORD_VIDEO: "REC VIDEO",
            Mode.PROCESS_VIDEO: "PROCESS",
            Mode.REVIEW: "REVIEW",
        }
        self._webcam_view.set_state_indicator(labels[mode])

        # Bottom panel: read-only except in REVIEW mode.
        src = self._bars_source_combo.currentData() if hasattr(self, "_bars_source_combo") else "mix"
        editable_src = src != "mix"
        self._value_editor.set_read_only(mode != Mode.REVIEW or not editable_src)

        # Pause button: legal in LIVE and REVIEW.
        can_pause = mode in (Mode.LIVE, Mode.REVIEW)
        self._pause_btn.setEnabled(can_pause)
        self._pause_btn.setText("Resume" if plan.paused else "Pause")

        # Timeline only meaningful in REVIEW.
        self._timeline.setEnabled(mode == Mode.REVIEW)

        # Broadcast checkbox reflects the SM.
        self._broadcast_check.blockSignals(True)
        self._broadcast_check.setChecked(self._sm.broadcast)
        self._broadcast_check.blockSignals(False)

        # Progress bar only during PROCESS.
        self._progress.setVisible(mode == Mode.PROCESS_VIDEO)

        self._update_controls()

    def _emit_frame(self, pos_ms: float, values: list[float],
                    pose: dict | None, mode: int | None = None) -> None:
        """Broadcast a frame iff broadcasting is enabled for the current plan."""
        if not self._broadcast_enabled:
            return
        filtered_pose = self._pipelines[self._output_index].filter_pose(pose)
        try:
            self._ipc.send_frame(
                pos_ms, values, filtered_pose,
                mode=mode if mode is not None else self._broadcast_mode,
            )
        except Exception:
            pass

    # ==================================================================
    # Source helpers
    # ==================================================================

    def _make_video_source(self) -> VideoSource:
        if isinstance(self._video_source, str):
            return VideoSource(kind="file", path=self._video_source)
        return VideoSource(kind="camera", camera_index=int(self._video_source))

    def _make_audio_source(self) -> AudioSource:
        if self._video_path and self._video_audio_path:
            return AudioSource(kind=AudioKind.FILE, path=self._video_audio_path)
        if self._audio_path:
            return AudioSource(kind=AudioKind.FILE, path=self._audio_path)
        return AudioSource(kind=AudioKind.NONE)

    def _playback_audio_path(self) -> str | None:
        if self._video_audio_path:
            return self._video_audio_path
        return self._audio_path or self._current_audio_path

    # ==================================================================
    # Backend / UI setup
    # ==================================================================

    def _setup_backend(self) -> None:
        self._audio = AudioPlayer()

        self._worker = CaptureWorker(source=self._video_source)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error.connect(self._on_worker_error)
        self._worker.webcam_record_finished.connect(self._on_webcam_record_finished)
        self._worker.process_progress.connect(self._on_process_progress)
        self._worker.process_finished.connect(self._on_process_finished)
        self._worker.start()

        self._ipc = IPCServer(host=self._ipc_host, port=self._ipc_port)
        self._ipc.start()

        # Enumerate cameras ONCE at startup (opening indices 0-9 is slow and must
        # not run every time the input dropdowns rebuild on a source switch).
        try:
            self._cameras = self._worker.list_cameras()
        except Exception:
            self._cameras = []
        if not self._cameras:
            self._cameras = [(0, "Camera 0")]

    def _setup_ui(self) -> None:
        self.setWindowTitle("SynClip Capture")
        self.resize(1280, 800)

        central = QWidget(self)
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.setSpacing(4)

        self._h_splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter = self._h_splitter

        # Left column: Inputs strip, file browser, takes panel.
        left_col = QWidget(splitter)
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_layout.addWidget(self._build_input_strip())

        self._file_browser = FileBrowser(self._root_dir, left_col)
        self._file_browser.file_selected.connect(self._on_file_selected)
        self._file_browser.use_as_video.connect(self._on_use_as_video)
        self._file_browser.use_as_audio.connect(self._on_use_as_audio)
        left_layout.addWidget(self._file_browser, stretch=1)

        self._takes_panel = TakesPanel(left_col)
        self._takes_panel.take_selected.connect(self._on_take_selected)
        self._takes_panel.live_selected.connect(self._go_to_live)
        self._takes_panel.take_set_default.connect(self._on_take_set_default)
        self._takes_panel.take_deleted.connect(self._on_take_deleted)
        self._takes_panel.take_renamed.connect(self._on_take_renamed)
        left_layout.addWidget(self._takes_panel, stretch=1)

        self._webcam_view = WebcamView()
        self._quad = QuadView(self._webcam_view, [c.label for c in self._views], splitter)
        self._quad.view_selected.connect(self._on_view_selected)
        for i in range(len(self._views)):
            r = self._quad.renderer(i)
            if hasattr(r, "camera_changed"):
                r.camera_changed.connect(lambda idx=i: self._on_camera_changed(idx))
        splitter.addWidget(left_col)
        splitter.addWidget(self._quad)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        self._quad.select(self._selected_view)
        self._load_view_meshes()

        transport = self._build_transport()

        # Single editor widget for both LIVE (read-only) and REVIEW (editable).
        # The source combo selects which data stream is shown and edited.
        self._value_editor = ValueEditor()
        self._value_editor.value_changed.connect(self._on_value_edited)
        editor_scroll = QScrollArea()
        editor_scroll.setWidget(self._value_editor)
        editor_scroll.setWidgetResizable(True)

        # Keep _bars as an alias so existing references still compile.
        self._bars = self._value_editor

        # The bottom "stack" is now just the single scroll area (no switching).
        self._bottom_stack = editor_scroll
        self._bottom_stack.setMinimumHeight(120)

        # Channel source selector: which stream the editor shows / edits.
        bars_header = QWidget(central)
        bars_header_lay = QHBoxLayout(bars_header)
        bars_header_lay.setContentsMargins(0, 0, 0, 0)
        bars_header_lay.addWidget(QLabel("Show:"))
        self._bars_source_combo = QComboBox(central)
        # "mix" (the read-only pipeline output) is the only fixed entry; every
        # track is added dynamically by _update_bars_source_combo, by name.
        self._bars_source_combo.addItem("mix", "mix")
        self._bars_source_combo.addItem("mediapipe", "mediapipe")
        self._bars_source_combo.currentIndexChanged.connect(self._on_bars_source_changed)
        bars_header_lay.addWidget(self._bars_source_combo)
        bars_header_lay.addStretch(1)

        bottom_container = QWidget(central)
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)
        bottom_layout.addWidget(transport)
        bottom_layout.addWidget(self._build_influence_editor(central))
        bottom_layout.addWidget(bars_header)
        bottom_layout.addWidget(self._bottom_stack)

        self._v_splitter = QSplitter(Qt.Orientation.Vertical, central)
        v_split = self._v_splitter
        v_split.addWidget(splitter)
        v_split.addWidget(bottom_container)
        v_split.setStretchFactor(0, 1)
        v_split.setStretchFactor(1, 0)
        v_split.setSizes([560, 240])
        outer_layout.addWidget(v_split, stretch=1)
        self._h_splitter.splitterMoved.connect(lambda *_: self._save_ui_config())
        self._v_splitter.splitterMoved.connect(lambda *_: self._save_ui_config())

        self._statusbar = QStatusBar(self)
        self.setStatusBar(self._statusbar)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self._progress.setFixedWidth(180)
        self._statusbar.addPermanentWidget(self._progress)

        self._ipc_label = QLabel(self)
        self._ipc_label.setStyleSheet("color: #8ad; padding: 0 8px;")
        self._statusbar.addPermanentWidget(self._ipc_label)
        self._update_ipc_label()

    def _build_input_strip(self) -> QWidget:
        """Top-left strip: Video / Audio dropdowns, Broadcast, Camera Settings."""
        box = QGroupBox("Inputs", self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)

        v_row = QHBoxLayout()
        v_row.addWidget(QLabel("Video:"))
        self._video_combo = QComboBox(box)
        self._video_combo.activated.connect(self._on_video_combo)
        v_row.addWidget(self._video_combo, stretch=1)
        layout.addLayout(v_row)

        a_row = QHBoxLayout()
        a_row.addWidget(QLabel("Audio:"))
        self._audio_combo = QComboBox(box)
        self._audio_combo.activated.connect(self._on_audio_combo)
        a_row.addWidget(self._audio_combo, stretch=1)
        layout.addLayout(a_row)

        ctrl_row = QHBoxLayout()
        self._broadcast_check = QCheckBox("Broadcast", box)
        # Reflect the actual default from the state machine, not a literal.
        self._broadcast_check.setChecked(self._sm.broadcast)
        self._broadcast_check.toggled.connect(self._on_broadcast_toggled)
        ctrl_row.addWidget(self._broadcast_check)
        cam_btn = QPushButton("Camera Settings", box)
        cam_btn.clicked.connect(self._open_camera_settings)
        ctrl_row.addWidget(cam_btn)
        layout.addLayout(ctrl_row)

        return box

    # ==================================================================
    # Input dropdowns
    # ==================================================================

    def _list_dir_media(self) -> tuple[list[str], list[str]]:
        """Return (video_files, audio_files) in the current root directory."""
        videos, audios = [], []
        try:
            for name in sorted(os.listdir(self._root_dir)):
                ext = os.path.splitext(name)[1].lower()
                full = os.path.join(self._root_dir, name)
                if not os.path.isfile(full):
                    continue
                if ext in _VIDEO_EXTS:
                    videos.append(full)
                elif ext in _AUDIO_EXTS:
                    audios.append(full)
        except OSError:
            pass
        return videos, audios

    def _rebuild_input_dropdowns(self) -> None:
        """Rebuild Video/Audio dropdowns from cameras + current-directory files."""
        videos, audios = self._list_dir_media()

        # Video: cameras then video files. Data = int index or str path.
        # Cameras come from the cached startup enumeration (never re-probed here).
        self._video_combo.blockSignals(True)
        self._video_combo.clear()
        for cam in self._cameras:
            if isinstance(cam, (tuple, list)):
                idx, label = cam[0], (str(cam[1]) if len(cam) > 1 else f"Camera {cam[0]}")
            else:
                idx, label = int(cam), f"Camera {cam}"
            self._video_combo.addItem(label, idx)
        for path in videos:
            self._video_combo.addItem(os.path.basename(path), path)
        self._select_combo_data(self._video_combo, self._video_source)
        self._video_combo.blockSignals(False)

        # Audio: "From video" (file video only), "Webcam mic" (camera), files.
        self._audio_combo.blockSignals(True)
        self._audio_combo.clear()
        if self._video_path:
            self._audio_combo.addItem("From video", ("video", None))
        if not isinstance(self._video_source, str):
            self._audio_combo.addItem("Webcam mic", ("mic", None))
        for path in audios:
            self._audio_combo.addItem(os.path.basename(path), ("file", path))
        self._select_current_audio_combo()
        self._audio_combo.blockSignals(False)

    def _select_combo_data(self, combo: QComboBox, data) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _select_current_audio_combo(self) -> None:
        target = None
        if self._video_path:
            target = ("video", None)
        elif self._audio_path:
            target = ("file", self._audio_path)
        if target is not None:
            self._select_combo_data(self._audio_combo, target)

    def _on_video_combo(self, _index: int) -> None:
        data = self._video_combo.currentData()
        if isinstance(data, str):
            self._load_video_file(data)
        else:
            self._select_camera(int(data))

    def _on_audio_combo(self, _index: int) -> None:
        data = self._audio_combo.currentData()
        if not isinstance(data, tuple):
            return
        kind, path = data
        if kind == "video":
            return  # already using the video's track
        if kind == "mic":
            self._audio_path = None
            self._sm.set_audio_source(AudioSource(kind=AudioKind.MIC))
            self._apply(self._sm.plan())
        elif kind == "file":
            self._load_audio_file(path)

    def _on_broadcast_toggled(self, on: bool) -> None:
        self._apply(self._sm.set_broadcast(on))

    def _on_use_as_video(self, path: str) -> None:
        self._load_video_file(path)
        self._rebuild_input_dropdowns()

    def _on_use_as_audio(self, path: str) -> None:
        self._load_audio_file(path)
        self._rebuild_input_dropdowns()

    def _open_camera_settings(self) -> None:
        dlg = CameraSettingsDialog(self._current_mode, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            w, h, fps = dlg.selected_mode()
            self._current_mode = (w, h, fps)
            self._worker.set_capture_mode(w, h, fps)
            self._statusbar.showMessage(
                f"Camera mode requested: {w} x {h} @ {fps} fps.", 4000
            )

    # ==================================================================
    # Toolbar / actions
    # ==================================================================

    def _setup_toolbar(self) -> None:
        # Alt+letter shortcuts (per spec) so single letters don't clash with
        # typing in sliders/fields.
        self._act_record = QAction("Rec Video", self)
        self._act_record.setShortcut("Alt+R")
        self._act_record.setToolTip("Record webcam to a video file (Alt+R)")
        self._act_record.triggered.connect(self._toggle_webcam_record)

        self._act_process = QAction("Process", self)
        self._act_process.setShortcut("Alt+P")
        self._act_process.setToolTip("Analyse the selected video file into a take (Alt+P)")
        self._act_process.triggered.connect(self._start_process)

        self._act_live = QAction("Live", self)
        self._act_live.setShortcut("Alt+L")
        self._act_live.setToolTip("Return to live mode (Alt+L)")
        self._act_live.triggered.connect(self._go_to_live)

        self._act_set_default = QAction("Set Default Take", self)
        self._act_set_default.setShortcut("Alt+D")
        self._act_set_default.triggered.connect(self._set_default_current_take)

        self._act_delete_take = QAction("Delete Take", self)
        self._act_delete_take.setShortcut("Delete")
        self._act_delete_take.triggered.connect(self._delete_current_take)

        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("toolbar_main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(self._act_record)
        toolbar.addAction(self._act_process)
        toolbar.addAction(self._act_live)
        toolbar.addSeparator()
        toolbar.addAction(self._act_set_default)
        toolbar.addAction(self._act_delete_take)

    def _setup_shortcuts(self) -> None:
        def sc(seq: str, slot) -> None:
            s = QShortcut(QKeySequence(seq), self)
            s.activated.connect(slot)

        sc("Left", lambda: self._step_frame(-1))
        sc("Right", lambda: self._step_frame(+1))
        sc("Up", lambda: self._takes_panel.select_adjacent(-1))
        sc("Down", lambda: self._takes_panel.select_adjacent(+1))
        sc("Space", self._on_space)
        sc("Escape", self._go_to_live)

    def _on_space(self) -> None:
        if self._sm.mode in (Mode.REVIEW, Mode.LIVE):
            self._on_toggle_pause()

    # Transport row geometry, reused to line the influence-curve editor up with
    # the timeline slider (same width, same x-extent -> playhead matches scrub).
    _TRANSPORT_MARGIN = 4
    _TRANSPORT_SPACING = 6
    _PAUSE_BTN_W = 90
    _TIME_LABEL_W = 110

    def _build_transport(self) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(self._TRANSPORT_MARGIN, 0, self._TRANSPORT_MARGIN, 0)
        layout.setSpacing(self._TRANSPORT_SPACING)

        self._pause_btn = QPushButton("Pause", row)
        self._pause_btn.setFixedWidth(self._PAUSE_BTN_W)
        self._pause_btn.clicked.connect(self._on_toggle_pause)
        layout.addWidget(self._pause_btn)

        self._timeline = QSlider(Qt.Orientation.Horizontal, row)
        self._timeline.setRange(0, 1000)
        self._timeline.setValue(0)
        self._timeline.sliderPressed.connect(self._on_scrub_start)
        self._timeline.sliderReleased.connect(self._on_scrub_end)
        self._timeline.valueChanged.connect(self._on_scrub_value)
        layout.addWidget(self._timeline, stretch=1)

        self._time_label = QLabel("0.0 / 0.0 s", row)
        self._time_label.setFixedWidth(self._TIME_LABEL_W)
        layout.addWidget(self._time_label)
        return row

    # ==================================================================
    # Influence-curve editor (docked under the timeline, scrub-synced)
    # ==================================================================

    def _build_influence_editor(self, parent: QWidget) -> QWidget:
        """The shared influence-curve editor shown directly under the timeline so
        its x-axis lines up with the scrubber. Hidden until a modifier row's
        curve button selects it; one curve edited at a time."""
        panel = QWidget(parent)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        head = QHBoxLayout()
        self._infl_curve_title = QLabel("Influence curve", panel)
        self._infl_curve_title.setStyleSheet("color:#9cf;")
        head.addWidget(self._infl_curve_title)
        head.addStretch(1)
        close_btn = QPushButton("×", panel)
        close_btn.setFixedWidth(26)
        close_btn.setToolTip("Close the influence-curve editor")
        close_btn.clicked.connect(self._close_influence_editor)
        head.addWidget(close_btn)
        lay.addLayout(head)

        # The CurveEditor is rebuilt per binding (its y-range/reference depend on
        # the mode), so it lives in a swappable host. Inset the host so the curve
        # lines up horizontally with the timeline slider above it (the slider is
        # flanked by the Pause button and the time label).
        self._infl_curve_host = QWidget(panel)
        self._infl_curve_host_lay = QVBoxLayout(self._infl_curve_host)
        left_inset = self._TRANSPORT_MARGIN + self._PAUSE_BTN_W + self._TRANSPORT_SPACING
        right_inset = self._TRANSPORT_MARGIN + self._TIME_LABEL_W + self._TRANSPORT_SPACING
        self._infl_curve_host_lay.setContentsMargins(left_inset, 0, right_inset, 0)
        self._infl_curve_host.setMinimumHeight(120)
        lay.addWidget(self._infl_curve_host)

        self._influence_editor = None          # current CurveEditor or None
        self._influence_curve_mc = None         # bound ModifierConfig or None
        self._influence_editor_panel = panel
        panel.setVisible(False)
        return panel

    def _on_influence_curve_selected(self, mc) -> None:
        if mc is None:
            self._close_influence_editor()
        else:
            self._bind_influence_editor(mc)

    def _bind_influence_editor(self, mc) -> None:
        from .curve_editor import CurveEditor
        if self._influence_editor is not None:
            self._influence_editor.setParent(None)
            self._influence_editor.deleteLater()
            self._influence_editor = None

        if mc.influence_anim == "relative":
            y_range, ref = (-1.0, 1.0), 0.0          # signed offset, neutral 0
            default = [(0.0, 0.0), (1.0, 0.0)]
        else:  # absolute
            y_range, ref = (0.0, 1.0), float(mc.influence)
            default = [(0.0, mc.influence), (1.0, mc.influence)]

        ed = CurveEditor(self._infl_curve_host, eval_mode="animation",
                         y_range=y_range, reference=ref, default_points=default)
        ed.set_points([(p[0], p[1]) for p in mc.influence_curve])
        ed.curve_changed.connect(self._on_influence_curve_changed)
        self._infl_curve_host_lay.addWidget(ed)
        self._influence_editor = ed
        self._influence_curve_mc = mc

        cls = mod_mod.get_class(mc.type)
        name = cls.display_name if cls else mc.type
        self._infl_curve_title.setText(f"Influence curve — {name} ({mc.influence_anim})")
        self._influence_editor_panel.setVisible(True)
        self._update_influence_playhead(self._audio.get_position_ms())

    def _on_influence_curve_changed(self, points) -> None:
        if self._influence_curve_mc is None:
            return
        self._influence_curve_mc.influence_curve = [list(p) for p in points]
        self._on_modifier_stack_changed()

    def _close_influence_editor(self) -> None:
        self._influence_curve_mc = None
        if hasattr(self, "_modifier_stack"):
            self._modifier_stack.clear_curve_editing()
        if self._influence_editor is not None:
            self._influence_editor.setParent(None)
            self._influence_editor.deleteLater()
            self._influence_editor = None
        self._influence_editor_panel.setVisible(False)

    def _update_influence_playhead(self, pos_ms: float) -> None:
        if getattr(self, "_influence_editor", None) is None:
            return
        dur = self._audio.duration_ms or 0.0
        self._influence_editor.set_playhead(pos_ms / dur if dur > 0 else None)

    def _on_toggle_pause(self) -> None:
        plan = self._sm.plan()
        if plan.paused:
            new = self._sm.resume()
        else:
            new = self._sm.pause()
        self._apply(new)
        # On resume of a video source, release any scrub/step hold (otherwise the
        # picture stays frozen on the held frame while audio/blendshapes advance)
        # and realign the picture to the audio clock.
        if not new.paused and self._video_path:
            self._worker.set_scrub(False)
            self._worker.set_audio_position(self._audio.get_position_ms(), True)
            self._worker.request_resync()

    # ==================================================================
    # Scrubbing (REVIEW)
    # ==================================================================

    def _on_scrub_start(self) -> None:
        if self._sm.mode != Mode.REVIEW:
            return
        self._scrubbing = True
        self._apply(self._sm.begin_scrub())
        if self._video_path:
            self._worker.set_scrub(True, self._scrub_target_ms())

    def _on_scrub_value(self, _value: int) -> None:
        if not self._scrubbing:
            return
        pos_ms = self._scrub_target_ms()
        self._audio.seek(pos_ms)
        if self._video_path:
            self._worker.set_scrub(True, pos_ms)
        self._update_time_label(pos_ms)
        if self._streams.has("mediapipe"):
            values = self._review_blendshapes(pos_ms, reset_first=True)
            if values:
                mp = self._streams.get("mediapipe")
                pose = mp.head_pose_at(pos_ms) if mp else None
                self._emit_frame(pos_ms, values, pose, mode=MODE_REVIEW)

    def _on_scrub_end(self) -> None:
        self._scrubbing = False
        pos_ms = self._scrub_target_ms()
        self._audio.seek(pos_ms)
        self._apply(self._sm.end_scrub())
        if self._video_path and not self._audio.is_paused():
            self._worker.set_scrub(False)

    def _scrub_target_ms(self) -> float:
        dur = self._audio.duration_ms or 0.0
        return (self._timeline.value() / 1000.0) * dur

    def _update_time_label(self, pos_ms: float) -> None:
        dur = self._audio.duration_ms or 0.0
        self._time_label.setText(f"{pos_ms / 1000.0:.1f} / {dur / 1000.0:.1f} s")

    # ==================================================================
    # AI second-source mixer (REVIEW)
    # ==================================================================

    # ---- Modifier stack (the right-dock per-view modifier list) ----------

    def _current_display_pose(self) -> dict | None:
        """The head pose currently shown (REVIEW interpolates the take; otherwise
        the last live/processed pose)."""
        if self._sm.mode == Mode.REVIEW and self._streams.has("mediapipe"):
            mp = self._streams.get("mediapipe")
            return mp.head_pose_at(self._audio.get_position_ms()) if mp else None
        return self._last_head_pose

    def _refresh_modifier_status(self) -> None:
        """Show each modifier's live status in the bound stack widget."""
        if not hasattr(self, "_modifier_stack"):
            return
        pipe = self._pipelines[self._selected_view]
        for i, m in enumerate(pipe._modifiers):
            self._modifier_stack.set_row_status(i, m.status_text())

    def _push_filtered_pose(self) -> None:
        """Re-apply each view's filtered head pose to its renderer immediately,
        so a pose-axis change resets that axis even when nothing is playing (the
        normal refresh paths no-op when idle)."""
        if not hasattr(self, "_quad"):
            return
        pose = self._current_display_pose()
        for i, pipe in enumerate(self._pipelines):
            self._quad.set_head_pose(i, pipe.filter_pose(pose))

    def _on_modifier_stack_changed(self) -> None:
        """A modifier was added / removed / reordered / retuned: re-apply the
        selected view's pipeline and refresh everything."""
        i = self._selected_view
        cfg = self._views[i]
        self._pipelines[i].apply_config(cfg)
        self._quad.set_title(i, *self._view_title(cfg))
        self._refresh_modifier_status()
        self._push_filtered_pose()
        self._refresh_review_frame()
        self._refresh_all_previews()
        self._save_ui_config()

    def _reset_ai_cache(self) -> None:
        """Drop every generated track, keeping only the mediapipe base."""
        for name in self._streams.names():
            if name != "mediapipe":
                self._streams.clear(name)
        self._ai_gen_for_audio = None
        self._update_bars_source_combo()
        self._update_pipeline_streams()

    def _update_bars_source_combo(self) -> None:
        """Rebuild the bottom-panel track list dynamically from the live tracks.

        "mix" is the only static entry (the read-only pipeline output); every
        other item is whatever track currently exists in the stream store, so no
        backend names are hardcoded here."""
        if not hasattr(self, "_bars_source_combo"):
            return
        combo = self._bars_source_combo
        prev = combo.currentData()

        # Remove stale dynamic entries (everything except "mix" at index 0).
        while combo.count() > 1:
            combo.removeItem(1)
        for name in self._track_names():
            combo.addItem(name, name)

        # Restore the previous selection if still valid, else fall back to mix.
        idx = combo.findData(prev)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _on_bars_source_changed(self, _idx: int) -> None:
        src = self._bars_source_combo.currentData()
        in_review = self._sm.mode == Mode.REVIEW
        # Only "mix" (the pipeline output) is read-only; every track is editable.
        editable_src = src != "mix"
        self._value_editor.set_read_only(not in_review or not editable_src)
        self._refresh_all_previews()

    def _generate_rhubarb_stream(self, audio_path: str | None,
                                 duration_ms: float) -> None:
        """During Process: if the rhubarb binary is present, generate a viseme
        track from the audio and add it to the live stream set as ``rhubarb``.

        Persistence is the caller's job (the streams are written into the take
        as a whole). No-op, and never raises, when rhubarb isn't installed.
        """
        from .. import rhubarb_lipsync
        if not audio_path or not rhubarb_lipsync.is_available():
            return
        try:
            frames = rhubarb_lipsync.generate_from_audio(
                audio_path, duration_ms=duration_ms)
        except Exception as exc:
            print(f"[main_window] rhubarb generation failed: {exc}")
            return
        if frames:
            self._streams.set("rhubarb", frames)

    def _refresh_review_frame(self) -> None:
        """Re-push the current REVIEW frame after a mixer change."""
        if self._sm.mode != Mode.REVIEW or not self._streams.has("mediapipe"):
            return
        pos_ms = self._audio.get_position_ms()
        values = self._review_blendshapes(pos_ms, reset_first=True)
        if values:
            mp = self._streams.get("mediapipe")
            pose = mp.head_pose_at(pos_ms) if mp else None
            self._emit_frame(pos_ms, values, pose, mode=MODE_REVIEW)
            self._last_review_values = values

    def _review_blendshapes(self, pos_ms: float,
                            reset_first: bool = False) -> list[float] | None:
        """Interpolated take blendshapes for *pos_ms*, run through every view
        pipeline (updating the quad previews) and returning the broadcast-output
        view's weights.

        *reset_first* clears temporal modifier state (e.g. SmoothModifier's EMA)
        before processing. Pass it for discontinuous jumps - scrub, step, single
        frame refresh - so the output isn't blended against an unrelated frame
        from a different point in the timeline. Continuous playback leaves it
        False to preserve smoothing.
        """
        if reset_first:
            self._reset_pipelines()
        mp = self._streams.get("mediapipe")
        if mp is None:
            return None
        take_vals = mp.values_at(pos_ms)
        if take_vals is None:
            return None
        ai_vals = self._streams.sample("ai", pos_ms)
        pose = mp.head_pose_at(pos_ms)
        return self._drive_previews(take_vals, ai_vals, pose, pos_ms)

    def _refresh_all_previews(self) -> None:
        """Recompute the quad previews (and broadcast) for the current frame."""
        if self._sm.mode == Mode.REVIEW and self._streams.has("mediapipe"):
            self._refresh_review_frame()
        elif self._last_raw_blendshapes is not None:
            pos_ms = self._audio.get_position_ms()
            ai_vals = self._streams.sample("ai", pos_ms)
            self._drive_previews(list(self._last_raw_blendshapes), ai_vals,
                                 self._last_head_pose, pos_ms)

    # ==================================================================
    # Modifier-stack dock (per-view source + ordered modifier list)
    # ==================================================================

    def _setup_tuning_toolbar(self) -> None:
        dock = QDockWidget("Modifiers", self)
        dock.setObjectName("dock_mix_settings")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        panel = QWidget(dock)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(6, 6, 6, 6)

        # Which view these controls edit (click a quad to switch).
        self._editing_label = QLabel(panel)
        self._editing_label.setStyleSheet("color:#9cf; font-weight:bold; padding:2px;")
        vbox.addWidget(self._editing_label)

        # 3D model picker + bake, both acting on the selected view.
        btn_row = QHBoxLayout()
        rename_view_btn = QPushButton("Rename View...", panel)
        rename_view_btn.clicked.connect(self._on_rename_view)
        btn_row.addWidget(rename_view_btn)
        model_btn = QPushButton("Choose 3D Model...", panel)
        model_btn.clicked.connect(self._on_choose_model)
        btn_row.addWidget(model_btn)
        self._export_glb_btn = QPushButton("Save as GLB...", panel)
        self._export_glb_btn.setToolTip("Export this view's processed animation as a GLB file")
        self._export_glb_btn.clicked.connect(self._on_save_as_glb)
        btn_row.addWidget(self._export_glb_btn)
        vbox.addLayout(btn_row)

        # Preset row.
        preset_row = QHBoxLayout()
        self._apply_preset_btn = QPushButton("Apply Preset v", panel)
        self._apply_preset_btn.clicked.connect(self._on_apply_preset_clicked)
        preset_row.addWidget(self._apply_preset_btn)
        save_preset_btn = QPushButton("Save Preset...", panel)
        save_preset_btn.clicked.connect(self._on_save_preset)
        preset_row.addWidget(save_preset_btn)
        vbox.addLayout(preset_row)

        self._model_label = QLabel(panel)
        self._model_label.setStyleSheet("color:#888; padding:0 2px;")
        vbox.addWidget(self._model_label)

        # The generic, per-view modifier stack.
        self._modifier_stack = ModifierStackWidget(panel)
        self._modifier_stack.set_available_streams(self._track_names())
        self._modifier_stack.changed.connect(self._on_modifier_stack_changed)
        self._modifier_stack.influence_curve_selected.connect(
            self._on_influence_curve_selected)
        scroll = QScrollArea(panel)
        scroll.setWidget(self._modifier_stack)
        scroll.setWidgetResizable(True)
        vbox.addWidget(scroll, stretch=1)

        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # Bind to the initially-selected view.
        self._load_view_config_into_widgets()

    # ---- View selection / binding ----------------------------------------

    def _selected_config(self) -> ViewConfig:
        return self._views[self._selected_view]

    def _on_view_selected(self, index: int) -> None:
        self._selected_view = index
        self._load_view_config_into_widgets()

    def _load_view_config_into_widgets(self) -> None:
        """Bind the selected view's ViewConfig to the modifier-stack dock."""
        cfg = self._selected_config()
        out = " (broadcast output)" if cfg.is_output else ""
        self._editing_label.setText(f"Editing: {cfg.label}{out}")
        mesh = self._get_mesh(cfg.mesh_path)
        if cfg.mesh_path and mesh is not None:
            self._model_label.setText(
                f"Model: {os.path.basename(cfg.mesh_path)} "
                f"({mesh.mapped_count}/52 mapped)"
            )
        elif cfg.mesh_path:
            self._model_label.setText(f"Model: {os.path.basename(cfg.mesh_path)} (load failed)")
        else:
            self._model_label.setText("Model: none")
        self._modifier_stack.bind(cfg)
        self._refresh_modifier_status()

    def _apply_selected_config(self) -> None:
        """Re-push the selected view's config into its pipeline and repaint."""
        i = self._selected_view
        cfg = self._views[i]
        self._pipelines[i].apply_config(cfg)
        self._quad.set_title(i, *self._view_title(cfg))
        self._save_ui_config()

    def _on_rename_view(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        cfg = self._selected_config()
        name, ok = QInputDialog.getText(
            self, "Rename View", "View name:", text=cfg.label
        )
        if not ok or not name.strip():
            return
        cfg.label = name.strip()
        self._quad.set_title(self._selected_view, *self._view_title(cfg))
        self._load_view_config_into_widgets()

    def _on_apply_preset_clicked(self) -> None:
        from PySide6.QtWidgets import QMenu
        if not self._presets:
            self._statusbar.showMessage("No presets saved yet.", 3000)
            return
        menu = QMenu(self)
        for name in sorted(self._presets):
            act = menu.addAction(name)
            act.triggered.connect(lambda _=False, n=name: self._apply_preset(n))
        menu.exec(self._apply_preset_btn.mapToGlobal(
            self._apply_preset_btn.rect().bottomLeft()))

    def _apply_preset(self, name: str) -> None:
        from ..modifiers import ModifierConfig
        mod_dicts = self._presets.get(name)
        if mod_dicts is None:
            return
        cfg = self._selected_config()
        cfg.modifiers = [ModifierConfig.from_dict(d) for d in mod_dicts]
        self._modifier_stack.bind(cfg)
        self._on_modifier_stack_changed()
        self._statusbar.showMessage(f"Applied preset '{name}'.", 3000)

    def _on_save_preset(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        cfg = self._selected_config()
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        self._presets[name] = [m.to_dict() for m in cfg.modifiers]
        self._save_ui_config()
        self._statusbar.showMessage(f"Saved preset '{name}'.", 3000)

    def _on_choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose 3D Model (glTF/GLB)", self._root_dir,
            "glTF Models (*.glb *.gltf)",
        )
        if not path:
            return
        cfg = self._selected_config()
        cfg.mesh_path = path
        mesh = self._get_mesh(path)
        self._quad.set_mesh(self._selected_view, mesh)
        if mesh is None:
            QMessageBox.warning(self, "Model load failed",
                                f"Could not load morph targets from:\n{path}")
        self._load_view_config_into_widgets()
        self._refresh_all_previews()

    def _rebroadcast_held_frame(self) -> None:
        plan = self._current_plan
        if plan is None or plan.mode != Mode.LIVE or not plan.paused:
            return
        if not self._last_raw_blendshapes:
            return
        self._reset_pipelines()
        pos_ms = self._audio.get_position_ms()
        ai_vals = self._streams.sample("ai", pos_ms)
        out = self._drive_previews(list(self._last_raw_blendshapes), ai_vals,
                                   self._last_head_pose, pos_ms)
        self._emit_frame(pos_ms, out, self._last_head_pose)

    def _on_value_edited(self, idx: int, value: float) -> None:
        """Single edit path for every track. Edits the selected stream's frame
        in place (mediapipe, ai, retarget, rhubarb, ...), schedules a take save,
        and re-broadcasts the recomputed output as a keyframe overwrite."""
        src = self._bars_source_combo.currentData() if hasattr(self, "_bars_source_combo") else "mediapipe"
        if src == "mix":
            return  # the pipeline output is read-only
        pos_ms = self._audio.get_position_ms()
        frames = self._streams.frames(src)
        positions = self._streams.positions(src)
        frame_i = self._nearest_frame_index(pos_ms, positions)
        if frame_i is None or not frames or frame_i >= len(frames):
            return
        bs = frames[frame_i].get("blendshapes")
        if not bs or idx >= len(bs):
            return
        bs[idx] = value
        self._schedule_take_save()
        # Recompute the output (drives the previews) and broadcast it as an edit.
        values = self._review_blendshapes(pos_ms, reset_first=True)
        if values:
            mp = self._streams.get("mediapipe")
            pose = mp.head_pose_at(pos_ms) if mp else None
            self._emit_frame(pos_ms, values, pose, mode=MODE_EDIT)

    def _nearest_frame_index(
        self, audio_pos_ms: float, positions: list[float] | None = None
    ) -> int | None:
        if positions is None:
            positions = self._streams.positions("mediapipe")
            if not positions:
                return None
        if not positions:
            return None
        i = bisect.bisect_left(positions, audio_pos_ms)
        if i <= 0:
            return 0
        if i >= len(positions):
            return len(positions) - 1
        before, after = positions[i - 1], positions[i]
        return i if (after - audio_pos_ms) < (audio_pos_ms - before) else i - 1

    def _schedule_take_save(self) -> None:
        self._save_timer.start(400)

    def _sync_streams_into_take(self) -> None:
        """Write the live in-memory stream set into the current take of the doc.

        The StreamStore is the single source of truth for the current take; this
        copies every track (mediapipe/ai/retarget/rhubarb/...) into the take so a
        whole-doc save persists them. Streams the take no longer has are dropped.
        """
        if self._current_synclip is None or not self._current_take_id:
            return
        for take in self._current_synclip.get("takes", []):
            if take.get("take_id") == self._current_take_id:
                take["streams"] = {
                    name: self._streams.frames(name)
                    for name in self._streams.names()
                }
                return

    def _save_current_take(self) -> None:
        if self._current_audio_path and self._current_synclip is not None:
            self._sync_streams_into_take()
            try:
                data_mod.save_synclip(self._current_audio_path, self._current_synclip)
                self._statusbar.showMessage("Take edits saved", 1500)
            except Exception as exc:
                self._statusbar.showMessage(f"Save error: {exc}", 4000)

    def _gather_capture_settings(self) -> dict:
        # A take stores the modifier stack of every view, so its full processing
        # setup is restored when the take is reselected. Mesh/label live in
        # ui.config (workspace), not per take.
        return {
            "view_modifiers": [
                [m.to_dict() for m in cfg.modifiers] for cfg in self._views
            ],
        }

    def _apply_capture_settings(self, settings: dict) -> None:
        if not settings:
            return
        from ..modifiers import ModifierConfig
        # Current format: per-view modifier stacks for every view.
        per_view = settings.get("view_modifiers")
        if per_view is not None:
            for i, mod_dicts in enumerate(per_view):
                if i >= len(self._views):
                    break
                self._views[i].modifiers = [
                    ModifierConfig.from_dict(m) for m in mod_dicts
                ]
                self._pipelines[i].apply_config(self._views[i])
            self._load_view_config_into_widgets()
            return
        # Legacy format: only the broadcast-output view's stack was stored.
        legacy = settings.get("modifiers")
        if legacy is not None:
            cfg = self._views[self._output_index]
            cfg.modifiers = [ModifierConfig.from_dict(m) for m in legacy]
            self._pipelines[self._output_index].apply_config(cfg)
            if self._selected_view == self._output_index:
                self._load_view_config_into_widgets()

    def _on_save_as_glb(self) -> None:
        """Export the output view's processed animation as a GLB file."""
        from PySide6.QtWidgets import QFileDialog
        from ..export_glb import export_glb
        if not self._streams.has("mediapipe") or not self._current_audio_path:
            self._statusbar.showMessage(
                "Export needs a loaded take (record/process or open one first).", 4000)
            return
        cfg = self._views[self._output_index]
        mesh_path = cfg.mesh_path or _default_head_mesh()
        if not mesh_path or not os.path.isfile(mesh_path):
            self._statusbar.showMessage(
                "Export needs a 3D model assigned to this view.", 4000)
            return
        default_name = os.path.splitext(
            os.path.basename(self._current_audio_path))[0] + "_synclip.glb"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save GLB", os.path.join(self._root_dir, default_name),
            "GLB files (*.glb)")
        if not out_path:
            return
        # Run the output view's pipeline over every frame to get final blendshapes.
        pipe = ViewPipeline(cfg)
        mp = self._streams.get("mediapipe")
        take_stream = Stream(mp.frames, mp.positions) if mp else Stream()
        named_streams = {
            name: Stream(s.frames, s.positions)
            for name, s in ((n, self._streams.get(n)) for n in self._streams.names())
            if name != "mediapipe" and s is not None
        }
        out_frames = pipe.process_all(take_stream, named_streams)
        try:
            n_keys = export_glb(mesh_path, out_frames, out_path)
        except Exception as exc:
            self._statusbar.showMessage(f"GLB export error: {exc}", 6000)
            return
        self._statusbar.showMessage(
            f"Saved '{os.path.basename(out_path)}' ({n_keys} keyframes).", 5000)

    # ==================================================================
    # Menus
    # ==================================================================

    def _setup_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_dir_action = file_menu.addAction("Open Directory...")
        open_dir_action.triggered.connect(self._on_open_directory)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        capture_menu = menubar.addMenu("Capture")
        capture_menu.addAction(self._act_record)
        capture_menu.addAction(self._act_process)
        capture_menu.addAction(self._act_live)
        capture_menu.addSeparator()
        self._act_landmarks = QAction("Show Face Landmarks", self)
        self._act_landmarks.setCheckable(True)
        # Reflect the actual default rather than a literal.
        self._act_landmarks.setChecked(self._landmarks_user_visible)
        self._act_landmarks.triggered.connect(self._on_landmarks_toggled)
        capture_menu.addAction(self._act_landmarks)
        capture_menu.addSeparator()
        cam_settings_action = capture_menu.addAction("Camera Settings...")
        cam_settings_action.triggered.connect(self._open_camera_settings)
        capture_menu.addSeparator()
        retarget_settings_action = capture_menu.addAction("Retarget Settings...")
        retarget_settings_action.triggered.connect(self._open_retarget_settings)

        take_menu = menubar.addMenu("Take")
        take_menu.addAction(self._act_set_default)
        take_menu.addAction(self._act_delete_take)

    def _on_landmarks_toggled(self, checked: bool) -> None:
        self._landmarks_user_visible = checked
        if self._current_plan is not None:
            self._webcam_view.set_overlay_visible(
                self._current_plan.show_landmarks and checked
            )

    def _open_retarget_settings(self) -> None:
        dlg = RetargetSettingsDialog(self._retarget_cfg, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._retarget_cfg = dlg.result_config()

    def _start_retarget(self) -> None:
        """Run analysis-by-synthesis retargeting on the take stream.

        Rendering is done by the pure-NumPy software rasterizer in
        retarget.retarget_stream - no OpenGL context required - so this is just
        a thin UI wrapper that supplies the mesh and reports progress.
        """
        if not isinstance(self._video_source, str):
            return
        if not self._streams.has("mediapipe"):
            return
        cfg_view = self._views[self._output_index]
        mesh = self._get_mesh(cfg_view.mesh_path)
        if mesh is None:
            self._statusbar.showMessage(
                "Retargeting needs a 3D model assigned to the output view.", 4000)
            return

        from PySide6.QtWidgets import QApplication
        cfg = self._retarget_cfg
        take_stream = self._streams.get("mediapipe")
        self._progress.setVisible(True)

        def progress(done: int, total: int) -> None:
            self._progress.setValue(int(done / max(1, total) * 100))
            self._statusbar.showMessage(f"Retargeting frame {done}/{total}...", 0)
            QApplication.processEvents()

        try:
            result, any_detected = retarget_stream(
                take_stream, mesh, cfg, progress_cb=progress
            )
        except Exception as exc:
            self._statusbar.showMessage(f"Retargeting error: {exc}", 8000)
            return
        finally:
            self._progress.setVisible(False)

        if not any_detected:
            self._statusbar.showMessage(
                "Retargeting: MediaPipe detected no face in the renders - "
                "original blendshapes kept.", 6000
            )
            return
        updated = result.frames
        self._streams.set("retarget", updated)
        self._save_current_take()  # persists the whole take, retarget included
        self._update_pipeline_streams()
        self._refresh_all_previews()
        self._statusbar.showMessage(
            f"Retargeting complete ({len(updated)} frames in 'retarget' stream).",
            5000
        )

    def _setup_timer(self) -> None:
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_current_take)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._on_poll)
        self._poll_timer.start()

    # ==================================================================
    # Timer poll
    # ==================================================================

    def _on_poll(self) -> None:
        self._audio.poll()
        self._update_ipc_label()

        if self._video_path and self._sm.mode != Mode.PROCESS_VIDEO:
            self._worker.set_audio_position(
                self._audio.get_position_ms(), self._audio.is_playing()
            )

        if not self._scrubbing:
            pos_ms = self._audio.get_position_ms()
            dur = self._audio.duration_ms or 0.0
            if dur > 0:
                self._timeline.blockSignals(True)
                self._timeline.setValue(int(min(1.0, pos_ms / dur) * 1000))
                self._timeline.blockSignals(False)
            self._update_time_label(pos_ms)

        if self._sm.mode == Mode.REVIEW and self._streams.has("mediapipe"):
            audio_pos_ms = self._audio.get_position_ms()
            values = self._review_blendshapes(audio_pos_ms)
            if values:
                paused = bool(self._current_plan and self._current_plan.paused)
                # Broadcast MODE_REVIEW while playback is actually advancing:
                # when paused (or scrubbing, handled separately) nothing changes,
                # so we stay quiet and don't shove a connected viewer's playhead.
                if not paused and not self._scrubbing and values != self._last_review_values:
                    mp = self._streams.get("mediapipe")
                    pose = mp.head_pose_at(audio_pos_ms) if mp else None
                    self._emit_frame(audio_pos_ms, values, pose, mode=MODE_REVIEW)
                    self._last_review_values = values
                # Reflect playback in the editable sliders (unless the user is
                # dragging a value slider).
                if self._scrubbing or self._value_editor.isAncestorOf(self.focusWidget()):
                    pass  # user is dragging a slider; don't clobber it

    # ==================================================================
    # Frame callback
    # ==================================================================

    def _on_frame_ready(self, bgr_frame, landmarks_raw, blendshape_values, head_pose=None) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                self._current_fps = (len(self._frame_times) - 1) / span

        self._webcam_view.update_frame(bgr_frame, landmarks_raw, fps=self._current_fps)

        mode = self._sm.mode
        plan = self._current_plan

        if mode == Mode.LIVE:
            if plan is not None and plan.paused:
                return
            self._last_raw_blendshapes = list(blendshape_values)
            self._last_head_pose = head_pose
            pos_ms = self._audio.get_position_ms()
            ai_vals = self._streams.sample("ai", pos_ms)
            out = self._drive_previews(list(blendshape_values), ai_vals, head_pose, pos_ms)
            self._emit_frame(pos_ms, out, head_pose)

        elif mode == Mode.PROCESS_VIDEO:
            # Accumulate a take frame; derive position from the progress
            # fraction * audio duration (no audio playback in PROCESS).
            dur = self._audio.duration_ms or 0.0
            pos_ms = self._process_frac * dur if dur > 0 else self._record_frame_index * 33.3
            self._recorded_frames.append({
                "frame_index": self._record_frame_index,
                "audio_position_ms": pos_ms,
                "blendshapes": list(blendshape_values),
                "head_pose": head_pose or {"rot": [0.0] * 3, "pos": [0.0] * 3},
            })
            self._record_frame_index += 1
            self._last_raw_blendshapes = list(blendshape_values)
            self._last_head_pose = head_pose
            ai_vals = self._streams.sample("ai", pos_ms)
            out = self._drive_previews(list(blendshape_values), ai_vals, head_pose, pos_ms)
            self._emit_frame(pos_ms, out, head_pose, mode=MODE_LIVE)

    # ==================================================================
    # Audio events
    # ==================================================================

    def _on_worker_error(self, message: str) -> None:
        print(f"[CaptureWorker error] {message}")
        if hasattr(self, "_statusbar"):
            self._statusbar.showMessage(message, 6000)

    # ==================================================================
    # File browser / source loading
    # ==================================================================

    def _on_file_selected(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext in _VIDEO_EXTS:
            self._load_video_file(path)
        else:
            self._load_audio_file(path)
        self._save_ui_config()

    def _load_media_common(self, path: str, prefer_take_id: str | None = None) -> None:
        """Shared setup after either a video or audio file has been chosen.

        Loads the synclip JSON, restores the last-used take, drives the state
        machine to REVIEW or LIVE, and saves ui.config.  Called by both
        _load_video_file and _load_audio_file after they set up their
        file-type-specific fields.
        """
        self._reset_ai_cache()
        self._current_synclip = data_mod.load_synclip(path)
        self._takes_panel.load_takes(self._current_synclip)
        self._current_take_id = None
        self._streams.clear_all()
        if self._current_synclip:
            take_ids = {t.get("take_id") for t in self._current_synclip.get("takes", [])}
            want_id = prefer_take_id if prefer_take_id in take_ids else None
            if want_id is None:
                want_id = self._current_synclip.get("default_take")
            if want_id:
                self._set_current_take_by_id(want_id)
                self._takes_panel.select_take(want_id)

        self._sm.set_video_source(self._make_video_source())
        self._sm.set_audio_source(self._make_audio_source())

        if self._streams.has("mediapipe"):
            self._go_review(from_start=True)
        else:
            self._apply(self._sm.to_live())
        self._rebuild_input_dropdowns()
        self._save_ui_config()

    def _load_video_file(self, path: str) -> None:
        from ..video_audio import AudioExtractWorker

        self._video_audio_path = None  # cached OGG lives next to video, don't delete

        if self._extract_worker is not None:
            try:
                self._extract_worker.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._extract_worker.quit()
            self._extract_worker = None

        self._video_path = path
        self._video_source = path
        self._audio_path = None
        self._current_audio_path = path
        self._load_media_common(path)

        self._statusbar.showMessage(f"Extracting audio from {os.path.basename(path)}...", 0)
        self._extract_worker = AudioExtractWorker(path)
        self._extract_worker.finished.connect(
            lambda wav, p=path: self._on_audio_extracted(p, wav)
        )
        self._extract_worker.start()

    def _on_audio_extracted(self, video_path: str, wav_path: str) -> None:
        self._extract_worker = None
        if self._video_path != video_path:
            if wav_path:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
            return
        if not wav_path:
            self._statusbar.showMessage("Could not extract audio - ffmpeg not found.", 6000)
        else:
            self._video_audio_path = wav_path
            self._audio_loaded_path = None  # force reload through _sync_audio
            self._sm.set_audio_source(self._make_audio_source())
            self._apply(self._sm.plan())
        self._update_controls()

    def _load_audio_file(self, path: str, prefer_take_id: str | None = None) -> None:
        if self._video_path:
            self._video_path = None
            self._video_source = 0
            self._video_audio_path = None  # cached OGG lives next to video, don't delete

        self._audio_path = path
        self._current_audio_path = path
        self._audio_loaded_path = None
        self._load_media_common(path, prefer_take_id=prefer_take_id)

    # ==================================================================
    # Takes panel
    # ==================================================================

    def _on_take_selected(self, take_id: str) -> None:
        if self._sm.mode == Mode.PROCESS_VIDEO:
            return
        if self._sm.mode == Mode.LIVE and self._live_settings is None:
            self._live_settings = self._gather_capture_settings()
        self._set_current_take_by_id(take_id)
        if self._streams.has("mediapipe"):
            self._go_review(from_start=True)
        self._save_ui_config()

    def _on_take_set_default(self, take_id: str) -> None:
        if self._current_audio_path is None:
            return
        try:
            data_mod.set_default_take(self._current_audio_path, take_id)
        except Exception as exc:
            self._statusbar.showMessage(f"Error: {exc}", 3000)
            return
        self._reload_synclip()

    def _on_take_deleted(self, take_id: str) -> None:
        if self._current_audio_path is None:
            return
        reply = QMessageBox.question(
            self, "Delete take",
            f"Delete {take_id}? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            data_mod.delete_take(self._current_audio_path, take_id)
        except Exception as exc:
            self._statusbar.showMessage(f"Error: {exc}", 3000)
            return
        self._reload_synclip()
        if self._current_take_id == take_id:
            self._current_take_id = None
            self._streams.clear("mediapipe")
            self._go_to_live()

    def _on_take_renamed(self, take_id: str, name: str) -> None:
        if not self._current_audio_path:
            return
        try:
            data_mod.rename_take(self._current_audio_path, take_id, name)
        except Exception as exc:
            self._statusbar.showMessage(f"Rename error: {exc}", 3000)
            return
        self._reload_synclip()
        self._takes_panel.load_takes(self._current_synclip)

    # ==================================================================
    # Mode transitions
    # ==================================================================

    def _go_to_live(self) -> None:
        if self._live_settings is not None:
            self._apply_capture_settings(self._live_settings)
            self._live_settings = None
        self._takes_panel.select_live()
        self._reset_pipelines()
        self._apply(self._sm.to_live())
        self._bars.set_values([0.0] * 52)
        self._update_status()

    def _go_review(self, from_start: bool = False) -> None:
        take_id = self._current_take_id or "take"
        self._last_review_values = None  # force the first playback frame to stream
        self._reset_pipelines()
        self._apply(self._sm.to_review(take_id))
        # Clear any stale scrub/step hold from a previous REVIEW session so the
        # picture isn't frozen when we (re)enter playback.
        self._worker.set_scrub(False)
        if from_start:
            self._audio.seek(0.0)
            if self._video_path:
                self._worker.restart_video()
                self._worker.request_resync()
        self._update_status()

    # ---- RECORD_VIDEO ----

    def _toggle_webcam_record(self) -> None:
        if self._sm.mode == Mode.RECORD_VIDEO:
            self._worker.stop_webcam_record()  # clears record_path -> finished signal
            self._apply(self._sm.stop_record_video())
            self._act_record.setText("Rec Video")
            self._statusbar.showMessage("Webcam recording stopped - finalising...", 0)
            return

        if self._sm.mode != Mode.LIVE:
            self._statusbar.showMessage("Recording is only available in LIVE mode.", 4000)
            return
        if isinstance(self._video_source, str):
            self._statusbar.showMessage(
                "Recording needs a camera source, not a video file.", 4000
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Webcam Recording", self._root_dir, "MP4 Video (*.mp4)"
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        fd, raw_path = tempfile.mkstemp(prefix="synclip_raw_", suffix=".mp4")
        os.close(fd)
        self._webcam_raw_path = raw_path
        self._webcam_final_path = path
        self._webcam_audio_path = self._playback_audio_path()
        self._webcam_audio_start_ms = self._audio.get_position_ms()

        self._apply(self._sm.start_record_video(raw_path))
        self._act_record.setText("Stop Rec")
        self._statusbar.showMessage(f"Recording webcam to {os.path.basename(path)}...", 0)

    def _on_webcam_record_finished(self, raw_path: str, actual_fps: float) -> None:
        from ..video_audio import WebcamMuxWorker
        from ..capture_worker import _DEFAULT_FPS

        out_path = self._webcam_final_path or raw_path
        self._statusbar.showMessage(f"Finalising recording ({actual_fps:.1f} fps capture)...", 0)
        self._webcam_mux_worker = WebcamMuxWorker(
            raw_path, actual_fps, float(_DEFAULT_FPS),
            self._webcam_audio_path, self._webcam_audio_start_ms, out_path,
        )
        self._webcam_mux_worker.finished.connect(
            lambda final, raw=raw_path: self._on_webcam_mux_finished(final, raw)
        )
        self._webcam_mux_worker.start()

    def _on_webcam_mux_finished(self, final_path: str, raw_path: str) -> None:
        try:
            if raw_path and os.path.exists(raw_path):
                os.remove(raw_path)
        except OSError:
            pass
        if not final_path:
            QMessageBox.warning(
                self, "Finalising Failed",
                "Could not re-time/mux the recording (ffmpeg missing or failed).",
            )
            return
        self._statusbar.showMessage(
            f"Webcam recording saved: {os.path.basename(final_path)}", 5000
        )
        reply = QMessageBox.question(
            self, "Process for SynClip?",
            f"Recording saved to:\n{final_path}\n\nLoad and process it now?",
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._load_video_file(final_path)

    # ---- PROCESS_VIDEO ----

    def _start_process(self) -> None:
        if not self._video_path:
            self._statusbar.showMessage(
                "Select a video file first (Process analyses a file into a take).", 4000
            )
            return
        plan = self._sm.start_process_video()
        if plan.mode != Mode.PROCESS_VIDEO:
            self._statusbar.showMessage("Cannot process: not a video-file source.", 4000)
            return
        self._recorded_frames = []
        self._record_frame_index = 0
        self._process_frac = 0.0
        self._reset_pipelines()
        self._progress.setValue(0)
        # Clear any stale audio clock so the worker won't seek the picture during
        # the straight-through analysis pass, and restart the file from the top.
        self._worker.set_audio_position(0.0, False)
        self._worker.set_scrub(False)
        self._worker.restart_video()
        self._apply(plan)
        self._statusbar.showMessage("Processing video into a take...", 0)
        self._update_status()

    def _on_process_progress(self, frac: float, fps: float) -> None:
        self._process_frac = frac
        self._progress.setValue(int(max(0.0, min(1.0, frac)) * 100))
        self._statusbar.showMessage(f"Processing... {frac * 100:.0f}%  ({fps:.0f} fps)", 0)

    def _on_process_finished(self, ai_frames: list) -> None:
        if self._sm.mode != Mode.PROCESS_VIDEO:
            return
        audio_duration_ms = self._audio.duration_ms or 0.0
        if self._current_audio_path is None or not self._recorded_frames:
            self._apply(self._sm.to_live())
            self._statusbar.showMessage("Processing produced no frames.", 4000)
            return
        recorded = list(self._recorded_frames)
        try:
            new_take = data_mod.append_take(
                self._current_audio_path,
                recorded,
                audio_duration_ms,
                capture_settings=self._gather_capture_settings(),
            )
        except Exception as exc:
            self._statusbar.showMessage(f"Save error: {exc}", 5000)
            self._apply(self._sm.to_live())
            return
        new_take_id = new_take.get("take_id")
        # The new take is the live set. Populate the in-memory streams directly
        # from what we just generated (the source of truth) so they survive even
        # if persistence fails -- no disk round-trip required to see them.
        self._reload_synclip()
        self._current_take_id = new_take_id
        self._streams.clear_all()
        self._streams.set("mediapipe", recorded)
        if ai_frames:
            nonzero = sum(1 for f in ai_frames if any(f.get("blendshapes", [])))
            print(f"[main_window] AI stream: received {len(ai_frames)} frames "
                  f"from generator ({nonzero} non-neutral)")
            self._streams.set("ai", ai_frames)
            self._ai_gen_for_audio = self._current_audio_path
        else:
            print("[main_window] AI stream: generator returned no frames")
        # Rhubarb viseme track (optional, offline) generated during Process.
        # Feed it the actual audio file (the extracted OGG for a video source),
        # never the video container -- rhubarb only reads WAV/OGG.
        self._generate_rhubarb_stream(self._playback_audio_path(), audio_duration_ms)
        # Persist every generated track into the take in one write.
        self._save_current_take()
        if new_take_id:
            self._takes_panel.select_take(new_take_id)
        self._update_bars_source_combo()
        self._update_pipeline_streams()
        # Run retargeting while still in PROCESS_VIDEO mode so the audio/video
        # player stays paused and the OpenGL renderer isn't driven by playback
        # timers during the analysis-by-synthesis grab loop.
        self._start_retarget()
        # Only now transition to REVIEW so playback can begin.
        self._apply(self._sm.finish_process_video(new_take_id or "take"))
        self._go_review(from_start=True)
        self._statusbar.showMessage(
            f"Processed {len(self._recorded_frames)} frames -> {new_take_id}", 5000
        )

    # ==================================================================
    # Misc helpers
    # ==================================================================

    def _step_frame(self, delta: int) -> None:
        if self._sm.mode != Mode.REVIEW or not self._streams.has("mediapipe"):
            return
        if not self._audio.is_paused():
            self._apply(self._sm.pause())
        pos_ms = self._audio.get_position_ms()
        positions = self._streams.positions("mediapipe")
        i = bisect.bisect_left(positions, pos_ms)
        i = max(0, min(i + delta, len(positions) - 1))
        target_ms = positions[i]
        self._audio.seek(target_ms)
        if self._video_path:
            self._worker.set_scrub(True, target_ms)
        dur = self._audio.duration_ms or 1.0
        self._timeline.blockSignals(True)
        self._timeline.setValue(int(min(1.0, target_ms / dur) * 1000))
        self._timeline.blockSignals(False)
        self._update_time_label(target_ms)
        values = self._review_blendshapes(target_ms, reset_first=True)
        if values:
            mp = self._streams.get("mediapipe")
            pose = mp.head_pose_at(target_ms) if mp else None
            self._emit_frame(target_ms, values, pose, mode=MODE_REVIEW)

    def _set_default_current_take(self) -> None:
        take_id = self._takes_panel.current_take_id() or self._current_take_id
        if take_id:
            self._on_take_set_default(take_id)

    def _delete_current_take(self) -> None:
        take_id = self._takes_panel.current_take_id() or self._current_take_id
        if take_id:
            self._on_take_deleted(take_id)

    def _update_controls(self) -> None:
        mode = self._sm.mode
        is_file = isinstance(self._video_source, str)
        self._act_record.setEnabled(mode == Mode.LIVE and not is_file)
        self._act_process.setEnabled(mode in (Mode.LIVE, Mode.REVIEW) and is_file)
        self._act_live.setEnabled(mode != Mode.LIVE)
        has_take = bool(self._takes_panel.current_take_id() or self._current_take_id)
        self._act_set_default.setEnabled(has_take)
        self._act_delete_take.setEnabled(has_take)

    def _update_ipc_label(self) -> None:
        n = 0
        try:
            n = self._ipc.client_count
        except Exception:
            pass
        host, port = self._ipc.host, self._ipc.port
        clients = "no clients" if n == 0 else f"{n} client{'s' if n != 1 else ''}"
        text = f"Godot <-> {host}:{port}  ({clients})"
        if text != self._ipc_label_cache:
            self._ipc_label_cache = text
            self._ipc_label.setText(text)

    def _set_current_take_by_id(self, take_id: str) -> None:
        self._current_take_id = take_id
        # A take owns all its tracks; load them as the new live stream set.
        self._streams.clear_all()
        if self._current_synclip is None:
            self._update_bars_source_combo()
            return
        for take in self._current_synclip.get("takes", []):
            if take.get("take_id") == take_id:
                for name, frames in (take.get("streams") or {}).items():
                    if frames:
                        self._streams.set(name, frames)
                self._apply_capture_settings(take.get("capture_settings", {}))
                break
        self._update_bars_source_combo()
        self._update_pipeline_streams()

    def _reload_synclip(self) -> None:
        if self._current_audio_path:
            self._current_synclip = data_mod.load_synclip(self._current_audio_path)
        else:
            self._current_synclip = None
        self._takes_panel.load_takes(self._current_synclip)

    # ==================================================================
    # UI config persistence
    # ==================================================================

    def _save_ui_config(self) -> None:
        if not self._init_complete:
            return
        data = {
            "video_source": self._video_source,
            "audio_path": self._current_audio_path,
            "current_take_id": self._current_take_id,
            "views": [cfg.to_dict() for cfg in self._views],
            "presets": self._presets,
            "window_geometry": self.saveGeometry().toBase64().data().decode(),
            "window_state": self.saveState().toBase64().data().decode(),
            "h_splitter_sizes": self._h_splitter.sizes(),
            "v_splitter_sizes": self._v_splitter.sizes(),
            "broadcast": self._sm.broadcast,
            "camera_mode": list(self._current_mode) if self._current_mode else None,
            "selected_file": self._file_browser.current_media_path(),
        }
        ui_config_mod.save(self._root_dir, data)

    def _restore_ui_config(self) -> None:
        from PySide6.QtCore import QByteArray, QTimer
        data = ui_config_mod.load(self._root_dir)
        if not data:
            return

        # Restore window geometry / dock layout.
        geom = data.get("window_geometry")
        state = data.get("window_state")
        if geom:
            try:
                self.restoreGeometry(QByteArray.fromBase64(geom.encode()))
            except Exception:
                pass
        if state:
            try:
                self.restoreState(QByteArray.fromBase64(state.encode()))
            except Exception:
                pass

        # Splitter sizes: defer until after the first layout pass so the widget
        # has non-zero dimensions and setSizes() actually sticks.
        h_sizes = data.get("h_splitter_sizes")
        v_sizes = data.get("v_splitter_sizes")
        if h_sizes and len(h_sizes) == 2:
            QTimer.singleShot(0, lambda: self._h_splitter.setSizes(h_sizes))
        if v_sizes and len(v_sizes) == 2:
            QTimer.singleShot(0, lambda: self._v_splitter.setSizes(v_sizes))

        # Restore per-view mix settings first (before restoring sources,
        # which may trigger refreshes that reference the view pipelines).
        saved_views = data.get("views", [])
        for i, cfg_dict in enumerate(saved_views):
            if i < len(self._views):
                restored = ViewConfig.from_dict(cfg_dict)
                # Preserve is_output flag from the hardcoded layout.
                restored.is_output = self._views[i].is_output
                self._views[i] = restored
                self._pipelines[i].apply_config(restored)
        if saved_views:
            self._load_view_meshes()
            self._load_view_config_into_widgets()

        # Restore modifier presets.
        saved_presets = data.get("presets")
        if isinstance(saved_presets, dict):
            self._presets = saved_presets

        # Restore broadcast toggle.
        if "broadcast" in data:
            self._broadcast_check.setChecked(bool(data["broadcast"]))

        # Restore camera resolution/fps mode.
        cam_mode = data.get("camera_mode")
        if cam_mode and len(cam_mode) == 3:
            self._current_mode = tuple(cam_mode)
            self._worker.set_capture_mode(*self._current_mode)

        # Restore the input source. Audio takes precedence over video, since a
        # restored audio file replaces the video as the active source anyway.
        take_id = data.get("current_take_id")
        audio_path = data.get("audio_path")
        video_src = data.get("video_source")
        if audio_path and os.path.isfile(audio_path):
            # Load the audio and select the persisted take in one synchronous
            # pass (the synclip data is read synchronously, so no deferral is
            # needed). This avoids a second, flickering re-apply of the stack.
            self._load_audio_file(audio_path, prefer_take_id=take_id)
        elif video_src is not None and video_src != self._video_source:
            if isinstance(video_src, str) and os.path.isfile(video_src):
                self._load_video_file(video_src)
            elif isinstance(video_src, int):
                # Route a camera index through the full source switch so the
                # state machine, _video_path and the input dropdowns all stay
                # consistent (a bare assignment left the combo out of sync).
                self._select_camera(video_src)

        # Restore file browser highlight (no signal, just visual).
        selected_file = data.get("selected_file")
        if selected_file and os.path.isfile(selected_file):
            self._file_browser.select_file(selected_file)

    def _update_title(self) -> None:
        if self._current_audio_path:
            self.setWindowTitle(f"SynClip Capture - {os.path.basename(self._current_audio_path)}")
        else:
            self.setWindowTitle("SynClip Capture")

    def _update_status(self) -> None:
        mode = self._sm.mode
        if mode == Mode.PROCESS_VIDEO:
            msg = "PROCESS"
        elif mode == Mode.RECORD_VIDEO:
            msg = "RECORDING VIDEO"
        elif mode == Mode.REVIEW:
            tid = self._current_take_id or "-"
            star = ""
            if self._current_synclip and self._current_synclip.get("default_take") == tid:
                star = " *"
            msg = f"REVIEW {tid}{star}"
        else:
            msg = "LIVE"
        self._statusbar.showMessage(msg)
        self._update_controls()

    def _select_camera(self, index: int) -> None:
        self._video_path = None
        self._video_source = int(index)
        self._sm.set_video_source(self._make_video_source())
        self._sm.set_audio_source(self._make_audio_source())
        self._apply(self._sm.to_live())
        self._rebuild_input_dropdowns()

    def _on_open_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Directory", self._root_dir)
        if path:
            self._root_dir = path
            self._file_browser.set_root(path)
            self._rebuild_input_dropdowns()

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        self._audio.stop()
        try:
            self._worker.stop()
        except Exception:
            pass
        try:
            self._ipc.stop()
        except Exception:
            pass
        if self._extract_worker is not None:
            try:
                self._extract_worker.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._extract_worker.quit()
            self._extract_worker.wait(2000)
        # Flush any pending take edits before exit.
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._save_current_take()
        # Cached OGG lives next to the video - don't delete on exit.
        self._save_ui_config()
        super().closeEvent(event)
