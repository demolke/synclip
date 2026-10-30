"""
Recording sessions: persist every phone's stream **as-is** plus the metadata we
need to line frames up later.

The user requirement is to store the encoded video verbatim (no re-encode, no
quality loss), so per phone we write:

    <session>/<phone>.h264          raw H.264 access units, concatenated
    <session>/<phone>.index.jsonl   one JSON line per frame (see FrameIndex)
    <session>/<phone>.clock.jsonl   clock samples (host_ms, offset_ms, rtt_ms), ~1/s
    <session>/manifest.json         session-level info (phones, clock offsets...)

The index lets the *process* phase seek frames and know each frame's capture
time on the **server timeline** (so frames from different phones can be matched
by timestamp). Per-frame we always record the server arrival time; keyframes
additionally carry the phone's reported capture time (META), which -- together
with the phone's clock offset -- anchors the whole stream to real time.

Pure filesystem + json; no sockets, no Qt. Testable with a tmp dir.

The manifest schema (``schema_version`` 2, a strict superset of the original
one) and the reader for it live in :mod:`capmo.core.session_io`.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict

from .core import h264
from .core.session_io import SCHEMA_VERSION, measured_fps, read_index


def _safe_id(phone_id: str) -> str:
    """A filename-safe slug for a phone id (model strings have spaces/punct)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", phone_id).strip("_")
    return slug or "cam"


@dataclass
class FrameIndex:
    """One row of a phone's .index.jsonl."""
    n: int                      # frame number for this phone (0-based)
    offset: int                 # byte offset into the .h264 file
    size: int                   # payload length in bytes
    recv_ms: float              # server clock when the frame arrived
    key: bool                   # keyframe (IDR)?
    meta_ms: int | None = None  # phone capture time (keyframes only)
    server_ms: float | None = None  # capture time on the server timeline, if known


@dataclass
class PhoneRecording:
    phone_id: str
    slug: str
    h264_path: str
    index_path: str
    frame_count: int = 0
    byte_offset: int = 0
    first_recv_ms: float | None = None
    last_recv_ms: float | None = None
    cam_id: str | None = None
    keyframes: int = 0
    width: int | None = None
    height: int | None = None
    sps_fps: float | None = None
    fps_measured: float | None = None
    rotation_deg_ccw: float | None = None
    codec_config: bytes = b""
    clock_path: str | None = None
    clock_samples: int = 0
    clock_first: dict | None = None
    clock_last: dict | None = None
    clock_rtt_min: float | None = None
    _h264: object = field(default=None, repr=False)
    _index: object = field(default=None, repr=False)
    _clock: object = field(default=None, repr=False)


class RecordingSession:
    """Owns a session directory and one :class:`PhoneRecording` per phone.

    Open with a root dir; phones are registered lazily on their first frame.
    Call :meth:`close` to flush and write the final manifest.
    """

    def __init__(self, root_dir: str, name: str | None = None,
                 info: dict | None = None) -> None:
        """*info*: session-level manifest fields (``nominal_fps``,
        ``requested_size``, ``preset``, ``board`` ...), merged into the manifest."""
        self.created_ms = time.time() * 1000.0
        self.info = dict(info or {})
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.created_ms / 1000.0))
        base = name or f"session_{stamp}"
        # Never reuse a directory that already holds a recording (two sessions
        # started within the same second would otherwise overwrite each other).
        self.name, i = base, 2
        while os.path.isdir(os.path.join(root_dir, self.name)) and \
                os.listdir(os.path.join(root_dir, self.name)):
            self.name = f"{base}_{i}"
            i += 1
        self.dir = os.path.join(root_dir, self.name)
        os.makedirs(self.dir, exist_ok=True)
        self._phones: dict[str, PhoneRecording] = {}
        self._closed = False
        # Per-phone data that may arrive before the phone's first frame.
        self._pending_clock: dict[str, list[dict]] = {}
        self._pending_config: dict[str, bytes] = {}
        self._pending_info: dict[str, dict] = {}

    # -- registration --------------------------------------------------------

    def _phone(self, phone_id: str) -> PhoneRecording:
        rec = self._phones.get(phone_id)
        if rec is not None:
            return rec
        slug = _safe_id(phone_id)
        # Avoid collisions if two phones slug to the same name.
        base, i = slug, 2
        used = {r.slug for r in self._phones.values()}
        while slug in used:
            slug = f"{base}_{i}"
            i += 1
        h264_path = os.path.join(self.dir, f"{slug}.h264")
        index_path = os.path.join(self.dir, f"{slug}.index.jsonl")
        rec = PhoneRecording(
            phone_id=phone_id, slug=slug, h264_path=h264_path, index_path=index_path,
            _h264=open(h264_path, "wb"), _index=open(index_path, "w", encoding="utf-8"),
        )
        rec.clock_path = os.path.join(self.dir, f"{slug}.clock.jsonl")
        self._phones[phone_id] = rec
        cfg = self._pending_config.pop(phone_id, b"")
        if cfg:
            self._take_config(rec, cfg)
        for k, v in self._pending_info.pop(phone_id, {}).items():
            setattr(rec, k, v)
        for sample in self._pending_clock.pop(phone_id, []):
            self._write_clock(rec, sample)
        return rec

    # -- extra per-phone info (schema 2) ------------------------------------

    @staticmethod
    def _take_config(rec: PhoneRecording, data: bytes) -> None:
        """Remember the codec config (SPS/PPS) and the size it declares."""
        cfg = h264.codec_config(data)
        if not cfg:
            return
        if not rec.codec_config:
            rec.codec_config = cfg
        if rec.width is None:
            sps = h264.parse_sps(cfg)
            if sps:
                rec.width, rec.height, rec.sps_fps = sps["width"], sps["height"], sps["fps"]

    def set_codec_config(self, phone_id: str, data: bytes) -> None:
        """SPS/PPS seen on the live stream before recording started (so a session
        armed mid-stream is still decodable without its in-band config)."""
        rec = self._phones.get(phone_id)
        if rec is None:
            self._pending_config[phone_id] = data
        else:
            self._take_config(rec, data)

    def set_phone_info(self, phone_id: str, **info) -> None:
        """Per-phone fields: ``cam_id``, ``rotation_deg_ccw``, ``width``, ``height``."""
        rec = self._phones.get(phone_id)
        if rec is None:
            self._pending_info.setdefault(phone_id, {}).update(info)
        else:
            for k, v in info.items():
                setattr(rec, k, v)

    def add_clock_sample(self, phone_id: str, host_ms: float, offset_ms: float,
                         rtt_ms: float, raw_offset_ms: float | None = None,
                         raw_rtt_ms: float | None = None,
                         raw_t_ms: float | None = None) -> None:
        """Append one clock sample (the estimator's best offset/rtt and, if
        given, the latest raw exchange) to ``<slug>.clock.jsonl``."""
        if self._closed:
            return
        sample = {"host_ms": host_ms, "offset_ms": offset_ms, "rtt_ms": rtt_ms}
        if raw_offset_ms is not None:
            sample.update(raw_offset_ms=raw_offset_ms, raw_rtt_ms=raw_rtt_ms, raw_t_ms=raw_t_ms)
        rec = self._phones.get(phone_id)
        if rec is None:
            self._pending_clock.setdefault(phone_id, []).append(sample)
        else:
            self._write_clock(rec, sample)

    def _write_clock(self, rec: PhoneRecording, sample: dict) -> None:
        if rec._clock is None:
            rec._clock = open(rec.clock_path, "a", encoding="utf-8")
        rec._clock.write(json.dumps(sample) + "\n")
        rec.clock_samples += 1
        rec.clock_first = rec.clock_first or sample
        rec.clock_last = sample
        rtt = sample.get("raw_rtt_ms", sample["rtt_ms"])
        if rtt is not None:
            rec.clock_rtt_min = rtt if rec.clock_rtt_min is None else min(rec.clock_rtt_min, rtt)

    # -- writing -------------------------------------------------------------

    def write_frame(self, phone_id: str, payload: bytes, recv_ms: float,
                    is_keyframe: bool, meta_ms: int | None = None,
                    server_ms: float | None = None) -> FrameIndex:
        """Append one frame's bytes and index row. Returns the index row."""
        if self._closed:
            raise RuntimeError("session is closed")
        rec = self._phone(phone_id)
        row = FrameIndex(
            n=rec.frame_count, offset=rec.byte_offset, size=len(payload),
            recv_ms=recv_ms, key=bool(is_keyframe),
            meta_ms=meta_ms, server_ms=server_ms,
        )
        rec._h264.write(payload)
        rec._index.write(json.dumps(asdict(row)) + "\n")
        if is_keyframe:
            rec.keyframes += 1
        # SPS/PPS ride on keyframes or come as tiny config-only messages.
        if (is_keyframe or len(payload) < 1024) and (rec.width is None or not rec.codec_config):
            self._take_config(rec, payload)
        rec.frame_count += 1
        rec.byte_offset += len(payload)
        rec.first_recv_ms = rec.first_recv_ms if rec.first_recv_ms is not None else recv_ms
        rec.last_recv_ms = recv_ms
        return row

    # -- manifest / lifecycle ------------------------------------------------

    def phone_ids(self) -> list[str]:
        return list(self._phones.keys())

    def manifest(self, clock_offsets: dict[str, float] | None = None,
                 extra: dict | None = None) -> dict:
        clock_offsets = clock_offsets or {}
        man = {
            "name": self.name,
            "created_ms": self.created_ms,
            "phones": [self._phone_entry(r, clock_offsets) for r in self._phones.values()],
        }
        man["schema_version"] = SCHEMA_VERSION
        man["stopped_ms"] = time.time() * 1000.0 if self._closed else None
        for k in ("nominal_fps", "requested_size", "preset", "board"):
            man[k] = self.info.get(k)
        man.update({k: v for k, v in self.info.items() if k not in man})
        man.setdefault("recorder", {"tool": "capmo"})
        man.update(extra or {})
        return man

    def _phone_entry(self, r: PhoneRecording, clock_offsets: dict) -> dict:
        fps_nom = r.sps_fps or self.info.get("nominal_fps")
        return {
            # schema 1
            "phone_id": r.phone_id,
            "slug": r.slug,
            "h264": os.path.basename(r.h264_path),
            "index": os.path.basename(r.index_path),
            "frames": r.frame_count,
            "bytes": r.byte_offset,
            "first_recv_ms": r.first_recv_ms,
            "last_recv_ms": r.last_recv_ms,
            "clock_offset_ms": clock_offsets.get(r.phone_id),
            # schema 2 (additive)
            "cam_id": r.cam_id or r.slug,
            "width": r.width,
            "height": r.height,
            "fps_nominal": fps_nom,
            "fps_measured": r.fps_measured,
            "rotation_deg_ccw": r.rotation_deg_ccw,
            "keyframes": r.keyframes,
            "clock": os.path.basename(r.clock_path) if r.clock_samples else None,
            "clock_samples": r.clock_samples,
            "clock_summary": {
                "offset_first_ms": (r.clock_first or {}).get("offset_ms"),
                "offset_last_ms": (r.clock_last or {}).get("offset_ms"),
                "rtt_min_ms": r.clock_rtt_min,
            } if r.clock_samples else None,
            "codec_config_b64": base64.b64encode(r.codec_config).decode("ascii")
            if r.codec_config else None,
        }

    def write_manifest(self, clock_offsets: dict[str, float] | None = None,
                       extra: dict | None = None) -> str:
        path = os.path.join(self.dir, "manifest.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest(clock_offsets, extra), f, indent=2)
        os.replace(tmp, path)
        return path

    def flush(self) -> None:
        for r in self._phones.values():
            if r._h264:
                r._h264.flush()
            if r._index:
                r._index.flush()
            if r._clock:
                r._clock.flush()

    def close(self, clock_offsets: dict[str, float] | None = None,
              extra: dict | None = None) -> None:
        if self._closed:
            return
        for r in self._phones.values():
            try:
                if r._h264:
                    r._h264.close()
                if r._index:
                    r._index.close()
                if r._clock:
                    r._clock.close()
            except OSError:
                pass
            self._finalize_phone(r)
        self._closed = True
        self.write_manifest(clock_offsets, extra)

    def _finalize_phone(self, r: PhoneRecording) -> None:
        """Measured fps from the index; size via a PyAV decode if no SPS was parsed."""
        try:
            r.fps_measured = measured_fps(read_index(r.index_path))
        except (OSError, ValueError, KeyError):
            r.fps_measured = None
        if r.width is None and r.codec_config:
            try:
                from .core.session_io import _probe_size_pyav
                with open(r.index_path, encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                key = next((x for x in rows if x.get("key")), None)
                if key:
                    with open(r.h264_path, "rb") as f:
                        f.seek(key["offset"])
                        wh = _probe_size_pyav(r.codec_config, f.read(key["size"]))
                    if wh:
                        r.width, r.height = wh
            except Exception:
                pass
