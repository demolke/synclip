"""
Extract per-frame 2D body keypoints (or ChArUco detections) from ordinary video
files, producing small ``.npz`` files that can be uploaded for offline 3D
reconstruction. Heavy per-frame work happens where this runs; the output is tiny.

Pose mode (default)::

    python -m capmo.tools.extract_keypoints --out OUT --take take1 \\
        --model full  a.mp4 b.mp4 c.mp4 d.mp4

writes ``OUT/take1/<cam_id>.npz`` (kp, vis, world, t, frame_idx, width, height,
fps, source, ...) plus ``OUT/take1/meta.json``.

Calibration mode (``--calib``) runs the ChArUco detector on sampled frames and
writes ``OUT/<take|calib>/<cam_id>.calib.npz`` instead.

``--start``/``--end`` (seconds of video time) restrict either mode to a window,
e.g. the board segment at the start of a take::

    python -m capmo.tools.extract_keypoints --calib --start 0 --end 20 \\
        --calib-every 2 --out OUT --take take1 a.mp4 b.mp4 c.mp4 d.mp4

``t`` and ``frame_idx`` in the output always refer to the whole video: ``t`` is
the frame's presentation time (its timestamp in seconds, the video's own clock:
from 0 for ordinary MP4/MOV/AVI, from the container's ``start_time`` -- kept in
``meta.json`` -- for e.g. MKV/TS or an MP4 with an edit list, and the session
clock for ``session_to_take`` videos), and ``--start``/``--end`` are on that same
clock, here as in ``calibrate_board`` and ``reconstruct_take``.

A capmo **recording session** directory (``manifest.json`` + ``<slug>.h264``,
from the capmo UI or ``capmo.tools.record``) can be given instead of video
files::

    python -m capmo.tools.extract_keypoints --out kp --take take1 SESSION_DIR
    python -m capmo.tools.extract_keypoints --calib --start 0 --end 20 \\
        --out kp --take take1 SESSION_DIR

The raw H.264 is decoded directly (no conversion needed); each frame's ``t`` is
its host-clock capture time relative to the common session origin (the same
timeline ``session_to_take`` writes as PTS), so all cameras are already on one
clock; ``--start``/``--end`` are in that session time. Camera ids come from the
manifest (``cam_id``, default the slug); ``--cam-ids a,b,c`` renames them in
manifest order (``slug=a,...`` maps/selects). In ``--calib`` mode the board
stored in the manifest is used unless board options are given.

All pixel coordinates refer to the *displayed* orientation at the video's full
resolution (phone rotation metadata is applied), regardless of ``--max-height``,
which only shrinks the image fed to the model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from ..core.pose import NUM_LANDMARKS, estimate_full_res, resize_to_height as _resize_to_height  # noqa: E402,F401

PROXY_HEIGHT = 480


# --------------------------------------------------------------------- utils
class Progress:
    """Tiny dependency-free progress bar on stderr."""

    def __init__(self, total: int | None, label: str, enabled: bool = True) -> None:
        self.total, self.label, self.enabled = total, label, enabled
        self.n, self.t0, self._last = 0, time.time(), 0.0

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.time()
        if self.enabled and now - self._last > 0.25:
            self._last = now
            self._draw()

    def _draw(self) -> None:
        el = max(time.time() - self.t0, 1e-6)
        rate = self.n / el
        if self.total:
            frac = min(self.n / self.total, 1.0)
            fill = int(frac * 24)
            eta = (self.total - self.n) / rate if rate > 0 else 0
            msg = (f"\r{self.label}: [{'#' * fill}{'.' * (24 - fill)}] "
                   f"{self.n}/{self.total} {rate:.1f} f/s eta {eta:.0f}s ")
        else:
            msg = f"\r{self.label}: {self.n} frames {rate:.1f} f/s "
        sys.stderr.write(msg)
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            self._draw()
            sys.stderr.write("\n")
            sys.stderr.flush()


def _rotate_k(rotation_deg: float) -> int:
    """Number of 90-degree counter-clockwise turns to display upright."""
    return int(round(rotation_deg / 90.0)) % 4


def _stream_rotation(stream) -> float:
    """Display rotation (deg, counter-clockwise, ffmpeg convention) from metadata."""
    try:
        for sd in getattr(stream, "side_data", None) or []:
            rot = getattr(sd, "rotation", None)
            if rot is not None:
                return float(rot)
        # Some PyAV versions expose side data as a mapping of dicts.
        sd = getattr(stream, "side_data", None)
        if isinstance(sd, dict):
            for k, v in sd.items():
                if "DISPLAYMATRIX" in str(k).upper():
                    return float(v)
    except Exception:
        pass
    tag = (stream.metadata or {}).get("rotate")          # legacy tag, clockwise
    if tag:
        try:
            return -float(tag)
        except ValueError:
            pass
    return 0.0


class VideoSource:
    """Decodes a video yielding ``(idx, t_seconds, frame_bgr_upright)``."""

    def __init__(self, path: str) -> None:
        import av
        self.av = av
        self.path = path
        self.container = av.open(path)
        if not self.container.streams.video:
            raise ValueError("no video stream")
        self.stream = self.container.streams.video[0]
        try:
            self.stream.thread_type = "AUTO"
        except Exception:
            pass
        s = self.stream
        self.coded_w, self.coded_h = s.codec_context.width, s.codec_context.height
        rate = s.average_rate or s.guessed_rate
        self.fps = float(rate) if rate else 0.0
        dur = None
        if s.duration is not None and s.time_base is not None:
            dur = float(s.duration * s.time_base)
        elif self.container.duration:
            dur = self.container.duration / 1e6
        self.duration = dur
        self.start_time = (float(s.start_time * s.time_base)
                           if s.start_time is not None and s.time_base else 0.0)
        self.rotation = _stream_rotation(s)
        total = s.frames or (int(round(dur * self.fps)) if dur and self.fps else 0)
        self.est_frames = total or None
        self.codec = s.codec_context.name
        self.creation_time = (self.container.metadata or {}).get("creation_time")
        # Newer PyAV/FFmpeg expose the display matrix per frame: peek at frame 0
        # (then reopen) so the displayed size is known before decoding for real.
        try:
            first = next(self.container.decode(self.stream))
            if getattr(first, "rotation", None):
                self.rotation = float(first.rotation)
        except (StopIteration, av.error.FFmpegError):
            pass
        self.container.close()
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        try:
            self.stream.thread_type = "AUTO"
        except Exception:
            pass

    @property
    def size(self) -> tuple[int, int]:
        """Displayed (width, height)."""
        if _rotate_k(self.rotation) % 2:
            return self.coded_h, self.coded_w
        return self.coded_w, self.coded_h

    def window_frames(self, start: float | None = None, end: float | None = None) -> int | None:
        """Estimated number of frames in ``[start, end]`` seconds (None if unknown)."""
        if start is None and end is None:
            return self.est_frames
        if not self.fps:
            return None
        dur = self.duration if self.duration else (
            self.est_frames / self.fps if self.est_frames else None)
        lo = max(0.0, (start if start is not None else self.start_time) - self.start_time)
        hi = end - self.start_time if end is not None else dur
        if hi is None:
            return None
        if dur is not None:
            hi = min(hi, dur)
        return max(0, int(round((hi - lo) * self.fps)) + 1)

    def first_index(self, start: float | None = None) -> int:
        """Frame index (``idx`` of :meth:`frames`) of the first frame with ``t >= start``."""
        if not self.fps or start is None:
            return 0
        return max(0, math.ceil((start - self.start_time) * self.fps - 0.25))   # eps of frames()

    def _seek(self, start: float) -> bool:
        """Seek to the keyframe at or before *start* (seconds, the ``t`` clock). False if
        impossible or not needed."""
        s = self.stream
        if not s.time_base or start <= self.start_time:
            return False
        try:
            self.container.seek(int(start / s.time_base), stream=s,
                                backward=True, any_frame=False)
            return True
        except Exception:                 # unseekable input: decode from the top
            self.container.close()
            self.container = self.av.open(self.path)
            self.stream = self.container.streams.video[0]
            return False

    def frames(self, start: float | None = None, end: float | None = None):
        """Yield ``(idx, t, img)`` for frames with ``start <= t <= end``.

        ``t`` is the frame's presentation time in seconds (``frame.time``, which
        starts at the container's ``start_time``); ``start``/``end`` are on that
        clock. Without a window every frame is decoded and ``idx`` is the decode
        order (unchanged behaviour). With *start* the demuxer seeks to the
        previous keyframe first and skips forward; ``idx`` is then derived from
        the first kept frame's timestamp (``round((t - start_time) * fps)``) and
        counted up from there, which equals the decode order for
        constant-frame-rate video. *end* stops decoding after the window. A
        frame without a timestamp (or one going backwards) gets
        ``t = start_time + idx / fps``.
        """
        idx = 0
        last_t = None
        seeked = start is not None and self._seek(start)
        first = True
        eps = 0.25 / self.fps if self.fps else 1e-3
        try:
            for frame in self.container.decode(self.stream):
                t = frame.time
                if seeked and first:
                    if t is None:
                        seeked = False          # cannot place it: count from here
                    elif t < start - eps:
                        continue                # still before the window
                    else:
                        idx = int(round((t - self.start_time) * self.fps)) if self.fps else 0
                if t is None or (last_t is not None and t < last_t):
                    t = self.start_time + (idx / self.fps if self.fps else float(idx))
                last_t = t
                first = False
                if end is not None and t > end + eps:
                    break
                if start is not None and t < start - eps:
                    idx += 1
                    continue
                img = frame.to_ndarray(format="bgr24")
                k = _rotate_k(self.rotation)
                if k:
                    img = np.ascontiguousarray(np.rot90(img, k))
                yield idx, float(t), img
                idx += 1
        except self.av.error.FFmpegError as exc:
            print(f"\n  warning: decode error after {idx} frames ({exc}); keeping what was read",
                  file=sys.stderr)

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:
            pass


class SessionVideoSource:
    """A camera of a capmo recording session, decoded straight from its raw
    ``.h264`` with per-frame times from the session timeline (seconds since the
    session origin, see :class:`capmo.core.session_io.Session`).

    Same interface as :class:`VideoSource`; ``idx`` counts decodable frames
    (0 = the first IDR), i.e. the frame index of the ``session_to_take`` video.
    """

    def __init__(self, session, cam) -> None:
        import av
        self.av = av
        self.session, self.cam = session, cam
        self.path = cam.h264_path
        self.ts = session.frame_times_s(cam)
        if not len(self.ts):
            raise ValueError(f"{cam.cam_id}: no decodable frame in {cam.h264_path}")
        self.coded_w, self.coded_h = int(cam.width or 0), int(cam.height or 0)
        self.fps = float(cam.fps_nominal or cam.fps_measured or 0.0)
        self.start_time = 0.0
        self.duration = float(self.ts[-1])
        self.rotation = float(cam.rotation_deg_ccw or 0.0)
        self.est_frames = len(self.ts)
        self.codec = "h264"
        self.creation_time = None

    @property
    def size(self) -> tuple[int, int]:
        if _rotate_k(self.rotation) % 2:
            return self.coded_h, self.coded_w
        return self.coded_w, self.coded_h

    def _window(self, start, end):
        eps = 0.25 / self.fps if self.fps else 1e-3
        lo = 0 if start is None else int(np.searchsorted(self.ts, start - eps))
        hi = len(self.ts) if end is None else int(np.searchsorted(self.ts, end + eps, "right"))
        return lo, hi

    def window_frames(self, start: float | None = None, end: float | None = None) -> int:
        lo, hi = self._window(start, end)
        return max(0, hi - lo)

    def first_index(self, start: float | None = None) -> int:
        """Frame index (``idx`` of :meth:`frames`) of the first frame with ``t >= start``."""
        return self._window(start, None)[0]

    def frames(self, start: float | None = None, end: float | None = None):
        lo, hi = self._window(start, end)
        if lo >= hi:
            return
        cc = self.av.CodecContext.create("h264", "r")
        k_rot = _rotate_k(self.rotation)
        n_done = 0

        def emit(frames):
            for fr in frames:
                k = int(fr.pts) if fr.pts is not None else -1
                if k < lo or k >= hi:
                    continue
                img = fr.to_ndarray(format="bgr24")
                if k_rot:
                    img = np.ascontiguousarray(np.rot90(img, k_rot))
                yield k, float(self.ts[k]), img

        try:
            for k, _i, data in self.cam.access_units(start_frame=lo):
                if k >= hi:
                    break
                pkt = self.av.Packet(data)
                pkt.pts = pkt.dts = k
                for item in emit(cc.decode(pkt)):
                    n_done += 1
                    yield item
            for item in emit(cc.decode(None)):
                n_done += 1
                yield item
        except self.av.error.FFmpegError as exc:
            print(f"\n  warning: decode error after {n_done} frames ({exc}); keeping what "
                  f"was read", file=sys.stderr)

    def info(self) -> dict:
        c = self.cam
        return {"session": os.path.abspath(self.session.path), "slug": c.slug,
                "phone_id": c.phone_id, "time_method": c.times.method,
                "clock_offset_ms": c.offset.a if c.offset else None,
                "session_t0_ms": self.session.t0_ms,
                "leading_undecodable_frames": int(c.first_decodable)}

    def close(self) -> None:
        pass


def open_video(path: str):
    """:class:`VideoSource` for a video file, :class:`SessionVideoSource` for the
    ``<slug>.h264`` of a capmo session (the path ``meta.json`` records for it)."""
    d = os.path.dirname(os.path.abspath(path))
    if path.endswith(".h264") and os.path.isfile(os.path.join(d, "manifest.json")):
        from ..core.session_io import load_session
        sess = load_session(d)
        for c in sess.cameras:
            if os.path.abspath(c.h264_path) == os.path.abspath(path):
                return SessionVideoSource(sess, c)
    return VideoSource(path)


class ProxyWriter:
    """480p H.264 MP4 with the source's (possibly variable) timestamps."""

    def __init__(self, path: str, src_size: tuple[int, int], fps: float) -> None:
        import av
        from fractions import Fraction
        w, h = src_size
        ph = min(PROXY_HEIGHT, h)
        pw = int(round(w * ph / h))
        self.size = (pw - pw % 2, ph - ph % 2)
        self.av = av
        self.container = av.open(path, "w")
        self.stream = self.container.add_stream("libx264", rate=Fraction(1000, 1))
        self.stream.width, self.stream.height = self.size
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.time_base = Fraction(1, 1000)
        self.stream.options = {"crf": "28", "preset": "veryfast"}
        self._last_pts = -1

    def write(self, img: np.ndarray, t: float) -> None:
        import cv2
        small = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        fr = self.av.VideoFrame.from_ndarray(small, format="bgr24")
        pts = max(int(round(t * 1000)), self._last_pts + 1)
        self._last_pts = pts
        fr.pts = pts
        from fractions import Fraction
        fr.time_base = Fraction(1, 1000)
        for pkt in self.stream.encode(fr):
            self.container.mux(pkt)

    def close(self) -> None:
        for pkt in self.stream.encode():
            self.container.mux(pkt)
        self.container.close()


def _save_thumb(img: np.ndarray, path: str, width: int = 640) -> None:
    import cv2
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _keyframe_set(total: int | None, k: int) -> set[int]:
    """Frame indices to thumbnail: K evenly spread picks (just frame 0 if length unknown)."""
    if k <= 0:
        return set()
    if not total:
        return {0}
    if k >= total:
        return set(range(total))
    return {int(round((i + 0.5) * total / k - 0.5)) for i in range(k)}


# --------------------------------------------------------------- pose / calib
def extract_pose(src: VideoSource, estimator, *, every: int = 1, max_height: int = 1080,
                 on_frame=None, progress: Progress | None = None,
                 start: float | None = None, end: float | None = None) -> dict:
    """Run *estimator* over a video (or the ``[start, end]`` s window); returns the npz arrays."""
    W, H = src.size
    kps, viss, worlds, ts, idxs, det = [], [], [], [], [], []
    n_err = 0
    nan_kp = np.full((NUM_LANDMARKS, 2), np.nan, np.float32)
    nan_w = np.full((NUM_LANDMARKS, 3), np.nan, np.float32)
    first_idx = None
    for idx, t, img in src.frames(start, end):
        if first_idx is None:
            first_idx = idx
        if on_frame:
            on_frame(idx, t, img)
        if progress:
            progress.update()
        if idx % every:
            continue
        try:
            res = estimate_full_res(estimator, img, max_height, size=(W, H))
        except Exception as exc:       # one bad frame must not kill a long run
            n_err += 1
            if n_err <= 3:
                print(f"\n  warning: pose failed on frame {idx}: {exc}", file=sys.stderr)
            res = None
        ts.append(t)
        idxs.append(idx)
        if res is None:
            kps.append(nan_kp)
            viss.append(np.zeros(NUM_LANDMARKS, np.float32))
            worlds.append(nan_w)
            det.append(False)
        else:
            kps.append(np.asarray(res.points, np.float32))
            viss.append(np.asarray(res.visibility, np.float32))
            w = getattr(res, "world", None)
            worlds.append(nan_w if w is None else np.asarray(w, np.float32))
            det.append(True)
    T = len(ts)
    return dict(
        kp=np.stack(kps) if T else np.zeros((0, NUM_LANDMARKS, 2), np.float32),
        vis=np.stack(viss) if T else np.zeros((0, NUM_LANDMARKS), np.float32),
        world=np.stack(worlds) if T else np.zeros((0, NUM_LANDMARKS, 3), np.float32),
        t=np.asarray(ts, np.float64), frame_idx=np.asarray(idxs, np.int64),
        detected=np.asarray(det, bool), pose_errors=n_err,
        window_first_frame_idx=np.int64(first_idx or 0))


def extract_calib(src: VideoSource, board_cfg, *, samples: int = 200, min_corners: int = 4,
                  on_frame=None, progress: Progress | None = None,
                  start: float | None = None, end: float | None = None,
                  every: int | None = None) -> dict:
    """ChArUco detections (legacy/new layout autodetect).

    Frames are taken from the ``[start, end]`` s window (whole video by default):
    every *every*-th frame when given (1 = all frames), else ~*samples* evenly
    spaced ones. Views with fewer than *min_corners* corners are dropped.
    """
    from ..core.board import BoardDetector
    bd = BoardDetector(board_cfg)          # legacy/new layout autodetect (shared code)
    if every:
        step = max(1, int(every))
    else:
        n_win = src.window_frames(start, end)
        step = max(1, (n_win or samples * 5) // max(samples, 1))
    t_list, i_list, nm_list, corners, ids, offs = [], [], [], [], [], [0]
    tried = 0
    for idx, t, img in src.frames(start, end):
        if on_frame:
            on_frame(idx, t, img)
        if progress:
            progress.update()
        if idx % step:
            continue
        tried += 1
        n_switch = bd.switched
        det = bd(img)
        if bd.switched != n_switch:
            print(f"\n  note: board matched with legacy={bd.cfg.legacy}", file=sys.stderr)
        if det.count >= min_corners:
            t_list.append(t)
            i_list.append(idx)
            nm_list.append(det.n_markers)
            corners.append(np.asarray(det.corners, np.float32).reshape(-1, 2))
            ids.append(np.asarray(det.ids, np.int32).reshape(-1))
            offs.append(offs[-1] + det.count)
    return dict(
        t=np.asarray(t_list, np.float64), frame_idx=np.asarray(i_list, np.int64),
        n_markers=np.asarray(nm_list, np.int32),
        corners=np.concatenate(corners) if corners else np.zeros((0, 2), np.float32),
        ids=np.concatenate(ids) if ids else np.zeros((0,), np.int32),
        offsets=np.asarray(offs, np.int64), board_json=json.dumps(bd.cfg.to_dict()),
        frames_tried=tried, sample_step=np.int64(step), min_corners=np.int64(min_corners),
        window=np.array([np.nan if start is None else start,
                         np.nan if end is None else end], np.float64)), bd.cfg


# ------------------------------------------------------------------- driver
def process_video(path: str, cam_id: str, out_dir: str, args, estimator=None,
                  board_cfg=None, src=None) -> dict:
    src = src or VideoSource(path)
    W, H = src.size
    os.makedirs(out_dir, exist_ok=True)
    label = f"{cam_id}{' [calib]' if args.calib else ''}"
    start, end = args.start, args.end
    n_win = src.window_frames(start, end)
    prog = Progress(n_win, label, enabled=not args.quiet)
    first = src.first_index(start)
    kf_set = {first + k for k in _keyframe_set(n_win, args.keyframes)}
    kf_dir = os.path.join(out_dir, "keyframes")
    saved_kf: list[str] = []
    proxy = ProxyWriter(os.path.join(out_dir, f"{cam_id}.proxy.mp4"), (W, H), src.fps) \
        if args.proxy else None
    if args.keyframes:
        os.makedirs(kf_dir, exist_ok=True)

    def on_frame(idx, t, img):
        if idx in kf_set:
            fn = os.path.join(kf_dir, f"{cam_id}_{idx:06d}.jpg")
            _save_thumb(img, fn)
            saved_kf.append(os.path.basename(fn))
        if proxy:
            proxy.write(img, t)

    info = dict(file=os.path.basename(path), path=os.path.abspath(path), cam_id=cam_id,
                width=W, height=H, coded_width=src.coded_w, coded_height=src.coded_h,
                rotation_deg_ccw=src.rotation, fps=src.fps, duration=src.duration,
                start_time=src.start_time, codec=src.codec,
                creation_time=src.creation_time)
    if hasattr(src, "info"):
        info.update(src.info())
    npz_extra = ({"session_t0_ms": np.float64(src.session.t0_ms)}
                 if isinstance(src, SessionVideoSource) else {})
    try:
        if args.calib:
            data, cfg = extract_calib(src, board_cfg, samples=args.calib_samples,
                                      min_corners=args.min_corners, every=args.calib_every,
                                      start=start, end=end, on_frame=on_frame, progress=prog)
            npz = f"{cam_id}.calib.npz"
            tried = data.pop("frames_tried")
            np.savez_compressed(os.path.join(out_dir, npz), width=W, height=H, fps=src.fps,
                                source=os.path.basename(path), **npz_extra, **data)
            info.update(npz=npz, n_detections=int(len(data["t"])), frames_tried=tried,
                        board=cfg.to_dict(), start=start, end=end,
                        sample_step=int(data["sample_step"]), min_corners=args.min_corners)
        else:
            data = extract_pose(src, estimator, every=args.every, start=start, end=end,
                                max_height=args.max_height, on_frame=on_frame, progress=prog)
            npz = f"{cam_id}.npz"
            errors = data.pop("pose_errors")
            np.savez_compressed(os.path.join(out_dir, npz), width=W, height=H, fps=src.fps,
                                source=os.path.basename(path), **npz_extra, **data)
            n = len(data["t"])
            info.update(npz=npz, n_frames=n, n_detected=int(data["detected"].sum()),
                        pose_errors=errors, every=args.every, start=start, end=end,
                        t_first=float(data["t"][0]) if n else None,
                        t_last=float(data["t"][-1]) if n else None)
    finally:
        prog.close()
        if proxy:
            proxy.close()
        src.close()
    if proxy:
        info["proxy"] = f"{cam_id}.proxy.mp4"
    if saved_kf:
        info["keyframes"] = sorted(saved_kf)
    return info


def _update_meta(out_dir: str, key: str, entries: list[dict], extra: dict) -> None:
    """Write ``meta.json`` for this run: *entries* **replace** the *key* list
    (``videos`` or ``calib_videos``) and ``extra`` (which always holds
    ``timebase``: ``"session"`` or ``"video"``) is merged in.

    Whatever an earlier run left that this run did not refresh is dropped, so
    the take never mixes runs: cameras of the same kind not produced now, and,
    when the time base changed (e.g. plain videos into a take that held a
    session extraction), everything else too (the other kind's cameras, the
    ``session`` block). The npz files of dropped cameras are deleted, so the
    loaders' directory-listing fallback cannot pick them up either.
    """
    path = os.path.join(out_dir, "meta.json")
    meta = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    timebase = extra["timebase"]
    keep = {e["cam_id"] for e in entries}
    stale = [e for e in meta.get(key, []) if e.get("cam_id") not in keep]
    if meta.get("timebase", "video") != timebase:
        other = "videos" if key == "calib_videos" else "calib_videos"
        stale += meta.get(other, [])
        meta = {}
    for e in stale:
        npz = e.get("npz")
        p = os.path.join(out_dir, os.path.basename(npz)) if npz else None
        if p and os.path.exists(p) and os.path.basename(p) not in {x.get("npz") for x in entries}:
            os.remove(p)
            print(f"  removed stale {os.path.basename(p)} (not part of this run)")
    meta[key] = list(entries)
    if timebase != "session":
        meta.pop("session", None)
    meta.update(extra)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m capmo.tools.extract_keypoints",
        description="Extract MediaPipe keypoints (or ChArUco detections) from video files.")
    p.add_argument("videos", nargs="+",
                   help="input video files (MP4/MOV/...), or one capmo session directory")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--take", default=None, help="subfolder name (default: 'take', or 'calib' with --calib)")
    p.add_argument("--cam-ids", default=None, help="comma-separated camera ids (default: file stems)")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="full")
    p.add_argument("--max-height", type=int, default=1080,
                   help="downscale frames taller than this before the model (0 = never)")
    p.add_argument("--every", type=int, default=1, help="process every Nth frame")
    p.add_argument("--keyframes", type=int, default=0, help="save K JPEG thumbnails per video")
    p.add_argument("--proxy", action="store_true", help="write a 480p H.264 proxy MP4 per video")
    p.add_argument("--quiet", action="store_true", help="no progress bar")
    p.add_argument("--start", type=float, default=None,
                   help="only process video time >= START seconds (seeks; both modes)")
    p.add_argument("--end", type=float, default=None,
                   help="only process video time <= END seconds (both modes)")
    g = p.add_argument_group("calibration (--calib)")
    g.add_argument("--calib", action="store_true", help="detect ChArUco instead of pose")
    g.add_argument("--calib-samples", type=int, default=200,
                   help="frames sampled evenly per video (or per --start/--end window)")
    g.add_argument("--calib-every", type=int, default=None,
                   help="detect on every Nth frame of the window instead (1 = all frames)")
    g.add_argument("--min-corners", type=int, default=4,
                   help="keep board views with at least this many ChArUco corners")
    # board defaults (8x5, 0.04, 0.02, DICT_6X6_250) are applied in main() so a
    # session's stored board can fill in what is not given
    g.add_argument("--squares-x", type=int, default=None, help="default 8")
    g.add_argument("--squares-y", type=int, default=None, help="default 5")
    g.add_argument("--square-len", type=float, default=None, help="metres (default 0.04)")
    g.add_argument("--marker-len", type=float, default=None, help="metres (default 0.02)")
    g.add_argument("--dict", dest="dict_name", default=None, help="default DICT_6X6_250")
    g.add_argument("--legacy", action="store_true", help="start with the legacy ChArUco layout")
    return p


def main(argv=None, estimator=None) -> int:
    return run(build_parser().parse_args(argv), estimator)


def run(args, estimator=None, on_camera=None) -> int:
    """The extractor on parsed *args* (see :func:`build_parser`); the library entry
    used by :mod:`capmo.core.pipeline`. *estimator*: a pose estimator to use
    instead of loading ``--model``; *on_camera(cam_id, i, n)* is called before
    each camera is processed (progress / cancellation hook). Returns the exit
    code (0 ok, 1 some camera failed, 2 bad arguments)."""
    if args.every < 1:
        print("--every must be >= 1", file=sys.stderr)
        return 2
    if args.calib_every is not None and args.calib_every < 1:
        print("--calib-every must be >= 1", file=sys.stderr)
        return 2
    if args.start is not None and args.end is not None and args.end <= args.start:
        print("--end must be > --start", file=sys.stderr)
        return 2
    session = None
    sources = None
    if any(os.path.isdir(v) for v in args.videos):
        from ..core.session_io import is_session_dir, load_session
        if len(args.videos) != 1 or not is_session_dir(args.videos[0]):
            print("give either video files or exactly one capmo session directory "
                  "(with manifest.json)", file=sys.stderr)
            return 2
        try:
            session = load_session(args.videos[0], args.cam_ids)
        except (OSError, ValueError, KeyError) as exc:
            print(f"cannot read session {args.videos[0]}: {exc}", file=sys.stderr)
            return 2
        if not session.cameras:
            print(f"session {args.videos[0]} has no camera with frames", file=sys.stderr)
            return 2
        cams = session.cam_ids
        sources = {c.cam_id: c for c in session.cameras}
        videos = [c.h264_path for c in session.cameras]
        print(f"capmo session {session.name}: {len(cams)} camera(s) {', '.join(cams)}; "
              f"times = host capture clock - t0 ({session.t0_ms:.0f} ms)")
    else:
        videos = args.videos
        cams = ([c.strip() for c in args.cam_ids.split(",")] if args.cam_ids
                else [os.path.splitext(os.path.basename(v))[0] for v in videos])
        if len(cams) != len(videos) or len(set(cams)) != len(cams):
            print(f"need {len(videos)} unique cam ids, got {cams}", file=sys.stderr)
            return 2
        missing = [v for v in videos if not os.path.isfile(v)]
        if missing:
            print(f"missing files: {missing}", file=sys.stderr)
            return 2
    take = args.take or ("calib" if args.calib else "take")
    out_dir = os.path.join(args.out, take)

    board_cfg = None
    owns_estimator = False
    if args.calib:
        from ..calibration import BoardConfig
        stored = (session.board or {}) if session is not None else {}
        dflt = BoardConfig()

        def pick(arg, key):
            return arg if arg is not None else stored.get(key, getattr(dflt, key))
        board_cfg = BoardConfig(pick(args.squares_x, "squares_x"),
                                pick(args.squares_y, "squares_y"),
                                pick(args.square_len, "square_len"),
                                pick(args.marker_len, "marker_len"),
                                pick(args.dict_name, "dict_name"),
                                args.legacy or bool(stored.get("legacy", False)))
        if stored:
            print("  (board taken from the session manifest where not given)")
        print(f"ChArUco board: {board_cfg.to_dict()}")
    elif estimator is None:
        owns_estimator = True
        from ..core import pose
        estimator = pose.load_estimator(args.model)

    entries, failed = [], []
    try:
        for i, (v, cam) in enumerate(zip(videos, cams)):
            if on_camera is not None:
                on_camera(cam, i, len(cams))      # may raise (cancel)
            print(f"== {cam}: {v}")
            try:
                src = SessionVideoSource(session, sources[cam]) if session is not None else None
                info = process_video(v, cam, out_dir, args, estimator, board_cfg, src=src)
            except Exception as exc:
                failed.append(cam)
                print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            entries.append(info)
            if args.calib:
                print(f"  {info['n_detections']} board detections in {info['frames_tried']} "
                      "sampled frames")
            else:
                print(f"  {info['n_frames']} frames, pose found in {info['n_detected']}")
    finally:
        if owns_estimator and estimator is not None:
            estimator.close()
    if entries:
        key = "calib_videos" if args.calib else "videos"
        extra = {} if args.calib else {"model": args.model, "max_height": args.max_height,
                                       "take": take}
        extra["timebase"] = "video"
        if session is not None:
            extra["session"] = {k: v for k, v in session.to_meta().items() if k != "cameras"}
            extra["timebase"] = "session"
        _update_meta(out_dir, key, entries, extra)
        print(f"Wrote {len(entries)} file(s) to {out_dir}")
    if failed:
        print(f"Failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
