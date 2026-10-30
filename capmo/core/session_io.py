"""
Reader (and schema) for capmo recording sessions.

A session directory, as :class:`capmo.session.RecordingSession` writes it::

    manifest.json              session + per-camera info (schema below)
    <slug>.h264                raw H.264 access units, verbatim, concatenated
    <slug>.index.jsonl         per received VIDEO message: n, offset, size,
                               recv_ms, key, meta_ms, server_ms
    <slug>.clock.jsonl         (schema >= 2) clock samples, ~1/s:
                               host_ms, offset_ms, rtt_ms[, raw_offset_ms, raw_rtt_ms]

Manifest schema (``schema_version`` 2; version-1 manifests have no version key
and only ``name``, ``created_ms`` and per phone ``phone_id slug h264 index
frames bytes first_recv_ms last_recv_ms clock_offset_ms``). Version 2 adds,
**additively**::

    schema_version, stopped_ms, nominal_fps, requested_size [w,h], preset,
    board (ChArUco BoardConfig dict or null), recorder {...}
    phones[i]: cam_id (default: slug), width, height, fps_nominal (SPS VUI or
      the requested fps), fps_measured, rotation_deg_ccw (null: phones do not
      report it), keyframes, clock (file name), clock_samples,
      clock_summary {offset_first_ms, offset_last_ms, rtt_min_ms},
      codec_config_b64 (Annex-B SPS+PPS seen before/at the start)

:func:`load_session` reads both versions and derives what an old session lacks
(size from the SPS in the stream, offsets from the keyframe ``server_ms``).
Each :class:`SessionCamera` exposes per-frame host capture times (``t_ms``) for
**every** frame (see :mod:`capmo.core.timeline` for the model), and
:meth:`SessionCamera.access_units` yields decodable access units (codec config
prepended where missing, config-only messages folded into the next frame,
frames before the first IDR skipped). Frames are assumed to have no B-frame
reordering (phones stream Baseline H.264), so decode order = capture order.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import h264
from .timeline import FrameTimes, OffsetModel, host_frame_times, offset_model

SCHEMA_VERSION = 2
MANIFEST = "manifest.json"
_HEAD = 256


def is_session_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, MANIFEST))


def read_index(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:          # torn last line after a crash
                break
    return rows


def read_clock(path: str) -> list[dict]:
    return read_index(path) if os.path.isfile(path) else []


def clock_triples(rows: list[dict]) -> list[tuple[float, float, float]]:
    """``(host_ms, offset_ms, rtt_ms)`` samples, preferring the raw exchange
    values (deduplicated) over the estimator's filtered ones."""
    out, seen = [], set()
    for r in rows:
        off = r.get("raw_offset_ms", r.get("offset_ms"))
        rtt = r.get("raw_rtt_ms", r.get("rtt_ms"))
        if off is None or rtt is None:
            continue
        key = (r.get("raw_t_ms"), off, rtt) if "raw_offset_ms" in r else (r.get("host_ms"), off, rtt)
        if key in seen:
            continue
        seen.add(key)
        out.append((float(r.get("raw_t_ms") or r["host_ms"]), float(off), float(rtt)))
    return out


@dataclass
class SessionCamera:
    cam_id: str
    phone_id: str
    slug: str
    h264_path: str
    index_path: str
    rows: list[dict]                       # raw index rows
    frame_rows: np.ndarray                 # (N,) row index of each picture frame
    config_rows: list[int]                 # rows carrying only SPS/PPS/SEI
    recv_ms: np.ndarray                    # (N,)
    meta_ms: np.ndarray                    # (N,) nan where none
    key: np.ndarray                        # (N,) bool
    first_decodable: int                   # frame (not row) index of the first usable IDR
    width: int | None
    height: int | None
    fps_nominal: float | None
    fps_measured: float | None
    rotation_deg_ccw: float | None
    codec_config: bytes
    clock_samples: list[tuple[float, float, float]]
    offset: OffsetModel | None
    times: FrameTimes
    entry: dict = field(default_factory=dict)   # the manifest entry

    @property
    def n_frames(self) -> int:
        return len(self.frame_rows)

    @property
    def t_ms(self) -> np.ndarray:
        """Host-clock capture time (ms) of every picture frame."""
        return self.times.t_ms

    @property
    def decodable_t_ms(self) -> np.ndarray:
        return self.times.t_ms[self.first_decodable:]

    def _read(self, f, row: dict) -> bytes:
        f.seek(int(row["offset"]))
        return f.read(int(row["size"]))

    def access_units(self, start_frame: int | None = None):
        """Yield ``(k, frame_index, data)`` for decodable frames.

        ``k`` counts decodable frames (0 = first IDR), ``frame_index`` is the
        picture-frame index (into ``t_ms``). With *start_frame* (a ``k``),
        decoding starts at the last IDR at or before it (``k`` keeps counting
        from that IDR).
        """
        first = self.first_decodable
        if first >= self.n_frames:
            return
        start = first
        if start_frame is not None and start_frame > 0:
            want = first + int(start_frame)
            idrs = [i for i in range(first, min(want, self.n_frames - 1) + 1) if self.key[i]]
            start = idrs[-1] if idrs else first
        cfg_rows = set(self.config_rows)
        with open(self.h264_path, "rb") as f:
            # config-only rows (and in-band config) before the start
            pending = b""
            r0 = int(self.frame_rows[start])
            for r in range(r0):
                if r in cfg_rows:
                    pending = self._read(f, self.rows[r])
            for i in range(start, self.n_frames):
                r = int(self.frame_rows[i])
                prev = int(self.frame_rows[i - 1]) if i > 0 else -1
                extra = b"".join(self._read(f, self.rows[q]) for q in range(prev + 1, r)
                                 if q in cfg_rows)
                data = self._read(f, self.rows[r])
                if i == start:
                    types = h264.nal_types(data[:_HEAD] if len(data) > _HEAD else data)
                    has_sps = h264.NAL_SPS in types or h264.NAL_SPS in h264.nal_types(extra)
                    if not has_sps:
                        extra = (pending or self.codec_config) + extra
                if extra:
                    data = extra + data
                yield i - first, i, data

    def summary(self) -> dict:
        t = self.decodable_t_ms
        return {
            "cam_id": self.cam_id, "phone_id": self.phone_id, "slug": self.slug,
            "h264": os.path.basename(self.h264_path),
            "frames": self.n_frames, "decodable_frames": int(len(t)),
            "leading_undecodable_frames": int(self.first_decodable),
            "config_only_messages": len(self.config_rows),
            "width": self.width, "height": self.height,
            "fps_nominal": self.fps_nominal, "fps_measured": self.fps_measured,
            "rotation_deg_ccw": self.rotation_deg_ccw,
            "time_method": self.times.method,
            "period_ms": self.times.period_ms,
            "anchors": self.times.n_anchors,
            "anchors_rejected": self.times.n_anchors_rejected,
            "dropped_frames_inferred": int(self.times.dropped_slots + self.gap_frames()),
            "anchor_jitter_rms_ms": self.times.anchor_resid_ms,
            "clock": self.offset.to_dict() if self.offset else None,
            "first_t_ms": float(t[0]) if len(t) else None,
            "last_t_ms": float(t[-1]) if len(t) else None,
        }

    def gap_frames(self) -> int:
        """Frames missing outside the anchor model (gaps > 1.5 periods), e.g. before
        the first / after the last keyframe."""
        t = self.t_ms
        P = self.times.period_ms
        if len(t) < 2 or not P:
            return 0
        d = np.diff(t)
        g = d[d > 1.5 * P]
        return int(np.round(g / P - 1).sum()) if len(g) and self.times.method != "meta_anchors" else 0


@dataclass
class Session:
    path: str
    name: str
    created_ms: float | None
    schema_version: int
    cameras: list[SessionCamera]
    manifest: dict
    board: dict | None = None
    nominal_fps: float | None = None

    @property
    def cam_ids(self) -> list[str]:
        return [c.cam_id for c in self.cameras]

    def camera(self, cam_id: str) -> SessionCamera:
        for c in self.cameras:
            if c.cam_id == cam_id:
                return c
        raise KeyError(cam_id)

    @property
    def t0_ms(self) -> float:
        """Common session origin: earliest decodable frame over all cameras."""
        firsts = [c.decodable_t_ms[0] for c in self.cameras if len(c.decodable_t_ms)]
        return float(min(firsts)) if firsts else 0.0

    def frame_times_s(self, cam: SessionCamera) -> np.ndarray:
        """Decodable-frame times of *cam* in seconds since :attr:`t0_ms`."""
        return (cam.decodable_t_ms - self.t0_ms) / 1000.0

    def sync_report(self) -> dict:
        """Per-camera clock info and a take-level sync quality estimate.

        ``pair_uncertainty_ms`` bounds the relative error between the two worst
        cameras: the sum of their NTP bounds (rtt_min / 2) plus their META
        anchor jitter. ``None`` if any camera has no clock-based mapping.
        """
        unc = []
        for c in self.cameras:
            u = c.offset.uncertainty_ms if c.offset else None
            if u is None or c.times.method != "meta_anchors":
                unc.append(None)
            else:
                unc.append(u + (c.times.anchor_resid_ms or 0.0))
        known = sorted([u for u in unc if u is not None], reverse=True)
        pair = (sum(known[:2]) if len(known) >= 2 else (known[0] if known else None)) \
            if all(u is not None for u in unc) else None
        P = [c.times.period_ms for c in self.cameras if c.times.period_ms]
        return {
            "t0_ms": self.t0_ms,
            "clock_offsets_ms": {c.cam_id: (c.offset.a if c.offset else None) for c in self.cameras},
            "time_methods": {c.cam_id: c.times.method for c in self.cameras},
            "per_camera_uncertainty_ms": dict(zip(self.cam_ids, unc)),
            "pair_uncertainty_ms": pair,
            "pair_uncertainty_frames": (pair / float(np.median(P))) if (pair is not None and P) else None,
        }

    def to_meta(self) -> dict:
        return {"session": os.path.abspath(self.path), "name": self.name,
                "schema_version": self.schema_version, "created_ms": self.created_ms,
                "t0_ms": self.t0_ms, "nominal_fps": self.nominal_fps, "board": self.board,
                "cameras": [c.summary() for c in self.cameras], "sync": self.sync_report()}


# ------------------------------------------------------------------- loading
def classify_rows(h264_path: str, rows: list[dict]) -> tuple[list[int], list[int], bytes, dict | None]:
    """Split index rows into picture frames and config-only messages.

    Reads only the first bytes of each row (whole row when it is small or no
    slice shows up in the head). Returns ``(frame_rows, config_rows,
    first_codec_config, first_sps_info)``.
    """
    frame_rows, cfg_rows = [], []
    cfg = b""
    sps = None
    if not os.path.isfile(h264_path):
        return [], [], b"", None
    with open(h264_path, "rb") as f:
        for i, r in enumerate(rows):
            size = int(r["size"])
            f.seek(int(r["offset"]))
            head = f.read(min(size, _HEAD))
            types = h264.nal_types(head)
            slice_ = any(t in (h264.NAL_SLICE, h264.NAL_IDR) for t in types)
            data = head
            if not slice_ and size > _HEAD:
                data = head + f.read(size - _HEAD)
                types = h264.nal_types(data)
                slice_ = any(t in (h264.NAL_SLICE, h264.NAL_IDR) for t in types)
            if not types:
                slice_ = True                 # not Annex-B we understand: keep as a frame
            if h264.NAL_SPS in types and (sps is None or not cfg):
                if len(data) < size and r.get("key"):
                    f.seek(int(r["offset"]))
                    data = f.read(size)
                c = h264.codec_config(data)
                if c and not cfg:
                    cfg = c
                if sps is None:
                    sps = h264.parse_sps(data)
            (frame_rows if slice_ else cfg_rows).append(i)
    return frame_rows, cfg_rows, cfg, sps


def _probe_size_pyav(codec_config: bytes, first_au: bytes) -> tuple[int, int] | None:
    try:
        import av
        cc = av.CodecContext.create("h264", "r")
        for fr in cc.decode(av.Packet(codec_config + first_au)):
            return fr.width, fr.height
        for fr in cc.decode(None):
            return fr.width, fr.height
    except Exception:
        return None
    return None


def load_camera(session_dir: str, entry: dict, nominal_fps: float | None = None) -> SessionCamera:
    slug = entry.get("slug") or entry.get("phone_id") or "cam"
    h264_path = os.path.join(session_dir, entry.get("h264") or f"{slug}.h264")
    index_path = os.path.join(session_dir, entry.get("index") or f"{slug}.index.jsonl")
    rows = read_index(index_path) if os.path.isfile(index_path) else []
    frame_rows, cfg_rows, cfg_stream, sps = classify_rows(h264_path, rows)
    cfg_b64 = entry.get("codec_config_b64")
    cfg_manifest = base64.b64decode(cfg_b64) if cfg_b64 else b""
    codec_config = cfg_manifest or cfg_stream
    if sps is None and codec_config:
        sps = h264.parse_sps(codec_config)
    N = len(frame_rows)
    fr = np.asarray(frame_rows, np.int64)
    recv = np.array([float(rows[r]["recv_ms"]) for r in frame_rows], np.float64)
    meta = np.full(N, np.nan)
    key = np.zeros(N, bool)
    cfg_set = set(cfg_rows)
    pending_meta = None
    k = 0
    for i, r in enumerate(rows):
        if i in cfg_set:                       # META attached to a config-only message
            if r.get("meta_ms") is not None:
                pending_meta = r["meta_ms"]
            continue
        m = r.get("meta_ms")
        if m is None and pending_meta is not None and r.get("key"):
            m = pending_meta
        pending_meta = None
        if m is not None:
            meta[k] = float(m)
        key[k] = bool(r.get("key"))
        k += 1
    # first decodable frame: the first keyframe once codec config is known
    first = N
    have_cfg = bool(codec_config)
    with open(h264_path, "rb") if os.path.isfile(h264_path) else _Null() as f:
        for i in range(N):
            r = rows[frame_rows[i]]
            if not key[i]:
                continue
            f.seek(int(r["offset"]))
            data = f.read(int(r["size"]))
            types = h264.nal_types(data)
            if h264.NAL_SPS in types:
                have_cfg = True
            if h264.NAL_IDR in types and have_cfg:
                first = i
                break
            if h264.NAL_IDR not in types:
                key[i] = False                 # META without an IDR: not a sync point
    # clock
    clock_name = entry.get("clock") or f"{slug}.clock.jsonl"
    clock_rows = read_clock(os.path.join(session_dir, clock_name))
    samples = clock_triples(clock_rows)
    kpairs = [(float(r["server_ms"]), float(r["meta_ms"])) for r in rows
              if r.get("server_ms") is not None and r.get("meta_ms") is not None]
    mrpairs = [(recv[i], meta[i]) for i in range(N) if np.isfinite(meta[i])]
    om = offset_model(samples, kpairs, entry.get("clock_offset_ms"), mrpairs)
    fps_nom = entry.get("fps_nominal") or (sps or {}).get("fps") or nominal_fps
    times = host_frame_times(recv, meta, om, fps_nom)
    width, height = entry.get("width"), entry.get("height")
    if (width is None or height is None) and sps:
        width, height = sps["width"], sps["height"]
    if (width is None or height is None) and first < N and codec_config:
        with open(h264_path, "rb") as f:
            r = rows[frame_rows[first]]
            f.seek(int(r["offset"]))
            wh = _probe_size_pyav(codec_config, f.read(int(r["size"])))
        if wh:
            width, height = wh
    fps_meas = entry.get("fps_measured")
    if fps_meas is None and N >= 2:
        span = times.t_ms[-1] - times.t_ms[0]
        fps_meas = (N - 1) * 1000.0 / span if span > 0 else None
    return SessionCamera(
        cam_id=entry.get("cam_id") or slug, phone_id=entry.get("phone_id", slug), slug=slug,
        h264_path=h264_path, index_path=index_path, rows=rows, frame_rows=fr,
        config_rows=cfg_rows, recv_ms=recv, meta_ms=meta, key=key, first_decodable=first,
        width=width, height=height, fps_nominal=fps_nom, fps_measured=fps_meas,
        rotation_deg_ccw=entry.get("rotation_deg_ccw"), codec_config=codec_config,
        clock_samples=samples, offset=om, times=times, entry=entry)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def seek(self, *_):
        pass

    def read(self, *_):
        return b""


def load_session(session_dir: str, cam_ids=None) -> Session:
    """Load a capmo session directory (schema 1 or 2).

    *cam_ids*: optional renaming, see :func:`parse_cam_ids`.
    """
    with open(os.path.join(session_dir, MANIFEST), encoding="utf-8") as f:
        man = json.load(f)
    nominal = man.get("nominal_fps")
    cams = [load_camera(session_dir, e, nominal) for e in man.get("phones", [])]
    cams = [c for c in cams if c.n_frames > 0]
    sess = Session(path=session_dir, name=man.get("name") or os.path.basename(
        os.path.normpath(session_dir)), created_ms=man.get("created_ms"),
        schema_version=int(man.get("schema_version", 1)), cameras=cams, manifest=man,
        board=man.get("board"), nominal_fps=nominal)
    if cam_ids:
        apply_cam_ids(sess, cam_ids)
    return sess


def parse_cam_ids(spec) -> list[str] | dict[str, str]:
    """``"a,b,c"`` (rename in manifest order) or ``"slug1=a,slug2=b"`` (mapping
    by slug / phone id / current cam id; unmapped cameras are dropped)."""
    if isinstance(spec, (list, tuple)):
        items = [s.strip() for s in spec]
    else:
        items = [s.strip() for s in str(spec).split(",") if s.strip()]
    if items and all("=" in s for s in items):
        return dict(s.split("=", 1) for s in items)
    return items


def apply_cam_ids(sess: Session, spec) -> Session:
    ids = parse_cam_ids(spec)
    if isinstance(ids, dict):
        keep = []
        for c in sess.cameras:
            for k in (c.cam_id, c.slug, c.phone_id):
                if k in ids:
                    c.cam_id = ids[k]
                    keep.append(c)
                    break
        sess.cameras = keep
    else:
        if len(ids) != len(sess.cameras):
            raise ValueError(f"--cam-ids: {len(ids)} ids for {len(sess.cameras)} cameras "
                             f"({', '.join(sess.cam_ids)})")
        for c, n in zip(sess.cameras, ids):
            c.cam_id = n
    if len(set(sess.cam_ids)) != len(sess.cameras):
        raise ValueError(f"duplicate cam ids: {sess.cam_ids}")
    return sess


def measured_fps(rows: list[dict]) -> float | None:
    """Cheap fps estimate from index rows (keyframe META span, else arrivals)."""
    anchors = [(r["n"], r["meta_ms"]) for r in rows if r.get("meta_ms") is not None]
    if len(anchors) >= 2 and anchors[-1][0] > anchors[0][0]:
        (n0, m0), (n1, m1) = anchors[0], anchors[-1]
        if m1 > m0:
            return (n1 - n0) * 1000.0 / (m1 - m0)
    if len(rows) >= 2:
        span = rows[-1]["recv_ms"] - rows[0]["recv_ms"]
        if span > 0:
            return (len(rows) - 1) * 1000.0 / span
    return None
