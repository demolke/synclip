"""
QThread subclass that owns the OpenCV webcam and MediaPipe FaceLandmarker.

Driven by an atomically-swapped WorkerConfig so the run loop never reads
inconsistent flag combinations. The UI thread calls configure() to push a new
config; the loop snapshots it once per iteration.

Emits:
  frame_ready(frame, mp_result, blendshapes_52, head_pose_dict)
  error(str)
  webcam_record_finished(path, actual_fps)
  process_progress(fraction: float, fps: float)   -- PROCESS_VIDEO only
  process_finished()                               -- PROCESS_VIDEO only
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from . import head_pose as head_pose_mod

# Path to the bundled (or downloaded) MediaPipe model file.
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_FILENAME = "face_landmarker.task"
_MODEL_PATH = os.path.join(_MODEL_DIR, _MODEL_FILENAME)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

_DEFAULT_FPS = 60


def _ensure_model() -> str:
    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 0:
        return _MODEL_PATH
    print(f"[CaptureWorker] Downloading FaceLandmarker model to {_MODEL_PATH} ...")
    tmp_path = _MODEL_PATH + ".part"
    try:
        with urllib.request.urlopen(_MODEL_URL, timeout=30) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp_path, _MODEL_PATH)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    print("[CaptureWorker] Download complete.")
    return _MODEL_PATH


@dataclass(frozen=True)
class WorkerConfig:
    """Immutable snapshot of what the capture loop should do this tick.

    Built by MainWindow._apply() from a Plan and swapped atomically via
    configure(). The loop reads self._cfg once per iteration.
    """
    source: int | str = 0
    run_mediapipe: bool = True
    throttle_fps: int | None = _DEFAULT_FPS  # None => no sleep (PROCESS)
    video_looping: bool = True
    paused: bool = False
    record_path: str | None = None           # set => write frames here
    audio_path: str | None = None            # set for PROCESS_VIDEO AI generation


class CaptureWorker(QThread):
    """Background thread: webcam capture + MediaPipe blendshape detection."""

    frame_ready = Signal(object, object, list, object)
    error = Signal(str)
    webcam_record_finished = Signal(str, float)
    process_progress = Signal(float, float)  # (fraction 0..1, fps)
    process_finished = Signal(list)           # emits AI frames (may be empty)

    def __init__(self, source: int | str = 0) -> None:
        super().__init__()
        self._cfg = WorkerConfig(source=source)
        self._cfg_lock = threading.Lock()
        self._running = False
        # Set by stop() and honoured by run(): guards the window where run()'s
        # slow startup (building the MediaPipe landmarker) hasn't reached the
        # point of setting _running=True yet, so a stop() arriving during startup
        # can't be clobbered into a thread that then loops forever.
        self._stop_requested = False

        # Audio-clock sync (set from UI thread)
        self._audio_state = (0.0, False)
        # Scrub/seek (set from UI thread)
        self._scrub_active = False
        self._scrub_pos_ms = 0.0
        self._resync_request = False
        self._seek_to_start = False
        # Set True after PROCESS_VIDEO emits process_finished, so it fires once
        # per pass; reset by restart_video() at the start of a new process.
        self._process_done_sent = False

        # Recording bookkeeping (worker thread only)
        self._webcam_writer: cv2.VideoWriter | None = None
        self._webcam_record_path: str | None = None
        self._webcam_frame_count = 0
        self._webcam_start_time: float | None = None

    # ------------------------------------------------------------------
    # Public control (safe to call from UI thread)
    # ------------------------------------------------------------------

    def configure(self, **kwargs) -> None:
        """Atomically update the worker config.

        Accepts any subset of WorkerConfig fields. Unknown keys are ignored so
        callers can pass a full Plan's fields without worrying about extras.
        """
        fields = WorkerConfig.__dataclass_fields__
        filtered = {k: v for k, v in kwargs.items() if k in fields}
        with self._cfg_lock:
            current = self._cfg
        # Replace with updated values
        current_dict = {f: getattr(current, f) for f in fields}
        new_cfg = WorkerConfig(**{**current_dict, **filtered})
        with self._cfg_lock:
            self._cfg = new_cfg

    def _snap_cfg(self) -> WorkerConfig:
        with self._cfg_lock:
            return self._cfg

    def stop(self) -> None:
        self._stop_requested = True
        self._running = False
        self.wait()

    # Legacy convenience setters kept for compatibility during Step 3 migration.
    # MainWindow will call configure() after Step 3; these will be removed then.

    def set_camera(self, index: int) -> None:
        self.configure(source=index)

    def restart_video(self) -> None:
        self._seek_to_start = True
        self._process_done_sent = False

    def set_audio_position(self, pos_ms: float, active: bool) -> None:
        self._audio_state = (pos_ms, active)

    def request_resync(self) -> None:
        self._resync_request = True

    def set_scrub(self, active: bool, pos_ms: float = 0.0) -> None:
        self._scrub_active = active
        self._scrub_pos_ms = pos_ms

    def set_processing(self, enabled: bool) -> None:
        self.configure(run_mediapipe=enabled)

    def start_webcam_record(self, path: str) -> None:
        self.configure(record_path=path)
        self._webcam_record_path = path
        self._webcam_frame_count = 0
        self._webcam_start_time = None

    def stop_webcam_record(self) -> None:
        self.configure(record_path=None)

    def set_capture_mode(self, width: int, height: int, fps: int) -> None:
        """Request a specific camera resolution/fps; applied on next reopen."""
        self._desired_mode = (int(width), int(height), int(fps))
        self._force_reopen = True

    # ------------------------------------------------------------------
    # Camera enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def list_cameras() -> list[tuple[int, str]]:
        results: list[tuple[int, str]] = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap is not None:
                if cap.isOpened():
                    results.append((i, f"Camera {i}"))
                cap.release()
        return results

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.core import base_options as mp_base
        except ImportError:
            self.error.emit(
                "mediapipe is not installed.\n"
                "Install it with:  pip install mediapipe>=0.10"
            )
            return

        try:
            model_path = _ensure_model()
        except Exception as exc:
            self.error.emit(f"Failed to obtain FaceLandmarker model: {exc}")
            return

        base_opts = mp_base.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
        try:
            landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            self.error.emit(f"Failed to create FaceLandmarker: {exc}")
            return

        # If stop() arrived while the landmarker was being built, don't start the
        # loop - fall straight through to the finally below (which releases the
        # capture device and landmarker) so wait() can return.
        self._running = not self._stop_requested
        _zero_blendshapes: list[float] = [0.0] * 52
        last_timestamp_ms = -1
        reopen_failures = 0
        cap: cv2.VideoCapture | None = None
        open_source: int | str | None = None
        video_fps = 0.0  # native fps of the current video file (0 for cameras)
        _cam_retry_after: float = 0.0  # monotonic time before which we skip camera reopen

        # Camera mode override (from set_capture_mode / Camera Settings dialog)
        self._desired_mode: tuple[int, int, int] | None = None
        self._force_reopen = False

        try:
            while self._running:
                t_start = time.monotonic()
                cfg = self._snap_cfg()

                source = cfg.source
                is_video = isinstance(source, str)
                audio_pos_ms, audio_active = self._audio_state

                # Force reopen when mode was changed via set_capture_mode()
                if self._force_reopen and not is_video:
                    self._force_reopen = False
                    if cap is not None:
                        cap.release()
                    cap = None
                    open_source = None

                # Paused: hold the last frame, don't advance, don't record
                if cfg.paused:
                    time.sleep(1.0 / (cfg.throttle_fps or _DEFAULT_FPS))
                    continue

                # While a camera is in backoff (recently failed), skip reopen so
                # that a source change (e.g. switch to video) is picked up on the
                # very next iteration rather than waiting for a blocking
                # cv2.VideoCapture() call on the dead device to time out.
                if cap is None and not is_video and time.monotonic() < _cam_retry_after:
                    time.sleep(0.05)
                    continue

                # (Re)open capture device when source changes
                if cap is None or source != open_source:
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(source)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    video_fps = 0.0
                    if not is_video:
                        if self._desired_mode is not None:
                            mw, mh, mfps = self._desired_mode
                        else:
                            mw, mh, mfps = 1920, 1080, _DEFAULT_FPS
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, mw)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mh)
                        cap.set(cv2.CAP_PROP_FPS, mfps)
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    else:
                        # Pace a video file to its own native frame rate (a 24/30
                        # fps clip must not play at the camera's 60 fps throttle).
                        try:
                            video_fps = float(cap.get(cv2.CAP_PROP_FPS))
                        except Exception:
                            video_fps = 0.0
                        if not (1.0 <= video_fps <= 240.0):
                            video_fps = 30.0
                    open_source = source
                    last_timestamp_ms = -1
                    if cap.isOpened():
                        reopen_failures = 0

                if not cap.isOpened():
                    reopen_failures += 1
                    if reopen_failures == 1 or reopen_failures % 60 == 0:
                        self.error.emit(f"Cannot open source: {source!r}")
                    cap.release()
                    cap = None
                    if not is_video:
                        _cam_retry_after = time.monotonic() + 2.0
                    time.sleep(0.05)
                    continue

                # Seek-to-start
                if self._seek_to_start:
                    self._seek_to_start = False
                    if is_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                # Resync on resume
                if is_video and self._resync_request:
                    self._resync_request = False
                    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, audio_pos_ms))

                # Audio-clock sync / scrub apply ONLY to throttled playback
                # (LIVE/REVIEW/RECORD). In PROCESS_VIDEO (throttle_fps is None) we
                # read straight through to EOF -- syncing to a stale audio clock
                # there would seek the picture backwards forever and the file
                # would never finish.
                throttled = cfg.throttle_fps is not None
                if is_video and throttled and self._scrub_active:
                    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, self._scrub_pos_ms))
                elif is_video and throttled and audio_active:
                    # Keep picture aligned to audio clock
                    target = audio_pos_ms
                    pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                    drift = target - pos
                    frame_ms = 1000.0 / (video_fps or cfg.throttle_fps or _DEFAULT_FPS)
                    if drift < -300.0 or drift > 500.0:
                        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, target))
                    elif drift > frame_ms:
                        skips = min(int(drift / frame_ms), 6)
                        for _ in range(skips):
                            if not cap.grab():
                                break

                ret, frame = cap.read()
                if not ret:
                    # EOF or transient camera failure.
                    if is_video and cfg.video_looping:
                        # LIVE/REVIEW: loop the clip from the top.
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret:
                        if is_video and not cfg.video_looping:
                            # PROCESS_VIDEO: file exhausted. Generate AI frames
                            # from the audio, then signal completion ONCE and
                            # idle (do NOT kill the thread -- the window
                            # reconfigures us for REVIEW/LIVE right after).
                            if not self._process_done_sent:
                                self._process_done_sent = True
                                ai_frames = []
                                if cfg.audio_path:
                                    try:
                                        from . import ai_blendshapes
                                        ai_frames = ai_blendshapes.generate_from_audio(
                                            cfg.audio_path
                                        )
                                    except Exception as exc:
                                        print(f"[CaptureWorker] AI generation failed: {exc}")
                                self.process_finished.emit(ai_frames)
                            time.sleep(0.05)
                            continue
                        cap.release()
                        cap = None
                        if not is_video:
                            # Back off before retrying the camera so a source
                            # change (switch to video) is seen immediately.
                            _cam_retry_after = time.monotonic() + 2.0
                            time.sleep(0.05)
                        continue

                # ---- MediaPipe detection ----
                # run_mediapipe is the SINGLE gate - not a combination of booleans
                if cfg.run_mediapipe:
                    h, w = frame.shape[:2]
                    if w > 640:
                        new_h = int(h * (640.0 / w))
                        mp_frame = cv2.resize(frame, (640, new_h),
                                              interpolation=cv2.INTER_AREA)
                    else:
                        mp_frame = frame
                    rgb = cv2.cvtColor(mp_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    timestamp_ms = int(time.monotonic() * 1000)
                    if timestamp_ms <= last_timestamp_ms:
                        timestamp_ms = last_timestamp_ms + 1
                    last_timestamp_ms = timestamp_ms

                    try:
                        result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    except Exception as exc:
                        print(f"[CaptureWorker] detect_for_video skipped: {exc}")
                        self.frame_ready.emit(
                            frame.copy(), None, list(_zero_blendshapes),
                            dict(head_pose_mod.ZERO_POSE),
                        )
                        continue

                    if result.face_blendshapes:
                        blendshapes: list[float] = [
                            cat.score for cat in result.face_blendshapes[0]
                        ]
                        if len(blendshapes) < 52:
                            blendshapes.extend([0.0] * (52 - len(blendshapes)))
                        elif len(blendshapes) > 52:
                            blendshapes = blendshapes[:52]
                    else:
                        blendshapes = list(_zero_blendshapes)

                    pose = (
                        head_pose_mod.decompose(result.facial_transformation_matrixes[0])
                        if result.facial_transformation_matrixes
                        else dict(head_pose_mod.ZERO_POSE)
                    )

                    # Emit progress in PROCESS mode (no throttle)
                    if cfg.throttle_fps is None and is_video:
                        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        frac = float(pos / total) if total > 0 else 0.0
                        elapsed = time.monotonic() - t_start
                        proc_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                        self.process_progress.emit(frac, proc_fps)
                else:
                    result = None
                    blendshapes = list(_zero_blendshapes)
                    pose = dict(head_pose_mod.ZERO_POSE)

                self.frame_ready.emit(frame.copy(), result, blendshapes, pose)

                # ---- Webcam recording ----
                # Record only when source is a camera (not a video file) and
                # record_path is set in the config.
                if not is_video:
                    active_record_path = cfg.record_path
                    if active_record_path:
                        h, w = frame.shape[:2]
                        if self._webcam_writer is None:
                            # Lazily create writer on first frame
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            self._webcam_writer = cv2.VideoWriter(
                                active_record_path, fourcc, _DEFAULT_FPS, (w, h),
                            )
                            self._webcam_record_path = active_record_path
                            self._webcam_start_time = time.monotonic()
                            self._webcam_frame_count = 0
                        if self._webcam_writer is not None:
                            self._webcam_writer.write(frame)
                            self._webcam_frame_count += 1
                    elif self._webcam_writer is not None:
                        # record_path cleared -> stop and emit finished
                        elapsed = time.monotonic() - (self._webcam_start_time or 0.0)
                        actual_fps = (
                            self._webcam_frame_count / elapsed
                            if elapsed > 0 else float(_DEFAULT_FPS)
                        )
                        self._webcam_writer.release()
                        path = self._webcam_record_path or ""
                        self._webcam_writer = None
                        self._webcam_record_path = None
                        self.webcam_record_finished.emit(path, actual_fps)

                # ---- Throttle ----
                # None means PROCESS_VIDEO: run as fast as possible, no sleep.
                # Otherwise pace: a video file to its own native fps, a camera to
                # the configured throttle (so a 30 fps clip never plays at 60).
                if cfg.throttle_fps is not None:
                    pace_fps = video_fps if (is_video and video_fps > 0) else cfg.throttle_fps
                    elapsed = time.monotonic() - t_start
                    sleep_s = (1.0 / pace_fps) - elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)

        finally:
            if cap is not None:
                cap.release()
            if self._webcam_writer is not None:
                self._webcam_writer.release()
                self._webcam_writer = None
            landmarker.close()
