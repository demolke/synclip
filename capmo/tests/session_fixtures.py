"""Fabricate capmo recording sessions exactly as the recorder writes them.

Frames are synthetic H.264 (libx264 via PyAV) pushed through the real
:class:`RecorderController` / :class:`RecordingSession` with fake phone
connections: per-camera clock offsets, jittered arrival times, META only on
keyframes, optional encoder drops and the three ways the codec config can
arrive (separate config-only message, in-band on every IDR, or only before
recording started).

Each encoded frame carries its encode index ``k`` as the grey level of a flat
block (``code = 30 + 3*k``), so tests can check which frame a timestamp landed on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pytest

BLOCK = 32


def code_for(k: int) -> int:
    return 30 + 3 * k


def code_of(img: np.ndarray) -> float:
    """Grey level of the code block of a decoded (BGR) frame."""
    return float(img[4:BLOCK - 4, 4:BLOCK - 4].mean())


def make_image(k: int, w: int, h: int) -> np.ndarray:
    import cv2
    img = np.full((h, w, 3), 40, np.uint8)
    img[:BLOCK, :BLOCK] = code_for(k)
    cv2.circle(img, (BLOCK + 8 + (3 * k) % max(1, w - BLOCK - 16), h // 2), 6, (255, 255, 255), -1)
    return img


def encode_h264(n: int, w: int = 160, h: int = 96, gop: int = 10, fps: int = 30,
                repeat_headers: bool = False) -> list[bytes]:
    """*n* Annex-B access units (IDR every *gop* frames, no B-frames)."""
    av = pytest.importorskip("av")
    pytest.importorskip("cv2")
    cc = None
    for name in ("libx264", "h264"):
        try:
            cc = av.CodecContext.create(name, "w")
            break
        except Exception:
            cc = None
    if cc is None:
        pytest.skip("no H.264 encoder in this PyAV build")
    cc.width, cc.height, cc.pix_fmt = w, h, "yuv420p"
    cc.time_base = Fraction(1, fps)
    cc.framerate = Fraction(fps)
    params = f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0"
    if repeat_headers:
        params += ":repeat-headers=1"
    cc.options = {"preset": "ultrafast", "tune": "zerolatency", "x264-params": params}
    out = []
    try:
        for k in range(n):
            fr = av.VideoFrame.from_ndarray(make_image(k, w, h), format="bgr24")
            fr.pts = k
            out.extend(bytes(p) for p in cc.encode(fr))
        out.extend(bytes(p) for p in cc.encode(None))
    except Exception as exc:                          # pragma: no cover
        pytest.skip(f"H.264 encoder unusable: {exc}")
    assert len(out) == n
    return out


@dataclass
class _Sample:
    offset: float
    rtt: float
    t_ms: float | None = None


class FakeClock:
    def __init__(self, offset_ms: float, rtt_ms: float = 6.0) -> None:
        self.ready = True
        self.offset_ms = offset_ms
        self.rtt_ms = rtt_ms
        self.last_sample = _Sample(offset_ms, rtt_ms, time.time() * 1000.0)

    def to_server_ms(self, phone_ms):
        return phone_ms - self.offset_ms


class FakeConn:
    def __init__(self, phone_id: str, offset_ms: float) -> None:
        self.phone_id = phone_id
        self.label = phone_id
        self.clock = FakeClock(offset_ms)


@dataclass
class CamSpec:
    phone_id: str
    offset_ms: float = 0.0           # phone clock - host clock
    start_ms: float = 0.0            # first capture, host time after the base
    config: str = "separate"         # "separate" | "inband" | "midstream"
    drop: tuple = ()                 # capture slots the encoder skipped
    latency_ms: float = 15.0         # mean extra arrival latency (exponential)
    lead_frames: int = 3             # for "midstream": frames seen before recording


@dataclass
class FabricatedSession:
    dir: str
    base_ms: float
    fps: float
    # phone_id -> (n,) true host capture time (ms) of encoded frame k
    truth: dict = field(default_factory=dict)
    # phone_id -> index of the first recorded encoded frame
    first_k: dict = field(default_factory=dict)


def fabricate_session(root: str, specs: list[CamSpec], *, n: int = 40, fps: int = 30,
                      gop: int = 10, w: int = 160, h: int = 96, seed: int = 0,
                      info: dict | None = None, name: str = "synth") -> FabricatedSession:
    from ..recorder import RecorderController
    from ..server import CaptureServer
    rng = np.random.default_rng(seed)
    P = 1000.0 / fps
    base = time.time() * 1000.0
    srv = CaptureServer(host="127.0.0.1", port=0)      # never started
    conns = [FakeConn(s.phone_id, s.offset_ms) for s in specs]
    srv.connections = lambda: list(conns)               # for clock sampling / stop()
    rec = RecorderController(srv, root, clock_interval=3600)
    events = []          # (recv_ms, order, conn, payload, is_key, meta, server_ms, before_start)
    fab = FabricatedSession(dir="", base_ms=base, fps=fps)
    for ci, (s, conn) in enumerate(zip(specs, conns)):
        aus = encode_h264(n, w, h, gop, fps, repeat_headers=(s.config == "inband"))
        slots = np.delete(np.arange(n + len(s.drop)), list(s.drop))[:n].astype(float)
        cap_host = base + s.start_ms + slots * P + rng.normal(0, 0.3, n)
        fab.truth[s.phone_id] = cap_host
        lat = 4.0 + rng.exponential(s.latency_ms, n)
        recv = np.maximum.accumulate(cap_host + lat)
        lead = s.lead_frames if s.config == "midstream" else 0
        fab.first_k[s.phone_id] = lead
        for k in range(n):
            from ..core import h264
            payload = aus[k]
            idr = h264.is_idr(payload)
            meta = int(round(cap_host[k] + s.offset_ms)) if idr else None
            srv_ms = (meta - (s.offset_ms + rng.normal(0, 1.0))) if meta is not None else None
            if s.config == "separate" and k == 0:
                cfg = h264.codec_config(payload)
                body = b"".join(payload[a:b] for t, a, b in h264.nal_units(payload)
                                if t not in (h264.NAL_SPS, h264.NAL_PPS))
                events.append((recv[k] - 0.5, len(events), conn, cfg, False, None, None, False))
                payload = body
            events.append((recv[k], len(events), conn, payload, idr, meta, srv_ms, k < lead))
    events.sort(key=lambda e: (e[0], e[1]))
    # frames seen before recording was armed (codec config gets cached)
    for e in [e for e in events if e[7]]:
        rec._on_frame(e[2], e[3], e[0], e[4], e[5], e[6])
    rec.start(name=name, info=info if info is not None else {"nominal_fps": fps,
                                                               "requested_size": [w, h]})
    rec.sample_clocks()
    for i, e in enumerate([e for e in events if not e[7]]):
        rec._on_frame(e[2], e[3], e[0], e[4], e[5], e[6])
        if i % 25 == 0:
            for c in conns:   # a slightly noisy clock estimate over time
                c.clock.last_sample = _Sample(c.clock.offset_ms + rng.normal(0, 0.5),
                                              5.0 + rng.exponential(2.0), e[0])
            rec.sample_clocks()
    fab.dir = rec.stop()
    return fab
