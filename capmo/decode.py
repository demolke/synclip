"""
Live H.264 decode for the camera grid.

Each phone sends a raw H.264 (Annex-B) elementary stream; PyAV turns the packet
bytes into frames. The first decodable frame appears once a keyframe + parameter
sets (SPS/PPS) arrive, which the phone emits at each I-frame interval.

Decoding is CPU-heavy and must not block the network reader, so
:class:`LivePreviewHub` runs one worker thread per phone fed by a depth-1 queue:
we only ever care about the *latest* frame for a live preview, so a busy decoder
simply drops stale packets rather than backing up. The GUI thread reads the
latest decoded frame via :meth:`LivePreviewHub.latest` (or, with the host
arrival time of the packet that completed it and a per-phone sequence number,
:meth:`LivePreviewHub.latest_stamped` -- the live board tracker pairs cameras
by that time).

PyAV is optional: if ``av`` isn't installed, ``AVAILABLE`` is False, the hub
no-ops, and the grid keeps showing status only.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

log = logging.getLogger("capmo.decode")

try:
    import av  # type: ignore
    AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where av is absent
    av = None
    AVAILABLE = False


class H264Decoder:
    """Incremental raw-H.264 decoder. Feed packet bytes, get BGR frames."""

    def __init__(self) -> None:
        if not AVAILABLE:
            raise RuntimeError("PyAV (av) is not installed")
        self._ctx = av.CodecContext.create("h264", "r")

    def decode(self, payload: bytes) -> list[np.ndarray]:
        """Return any frames completed by *payload* as HxWx3 uint8 BGR arrays."""
        out: list[np.ndarray] = []
        try:
            for packet in self._ctx.parse(payload):
                for frame in self._ctx.decode(packet):
                    out.append(frame.to_ndarray(format="bgr24"))
        except av.error.FFmpegError as exc:
            # Corrupt/partial bitstream before sync -- skip until the next keyframe.
            # Normal right after connect (frames before the first SPS/PPS+IDR);
            # logged at DEBUG so a persistent failure is still visible with -v.
            log.debug("h264 decode skipped %d-byte unit: %s", len(payload), exc)
        return out


class LivePreviewHub:
    """Per-phone background decoders feeding a latest-frame cache.

    *decoder_factory* lets tests inject a fake decoder so the threading/queue
    behaviour is testable without PyAV.
    """

    def __init__(self, decoder_factory: Callable[[], object] | None = None) -> None:
        self._factory = decoder_factory or (H264Decoder if AVAILABLE else None)
        self._lock = threading.Lock()
        self._workers: dict[str, _Worker] = {}
        self._latest: dict[str, np.ndarray] = {}
        self._stamp: dict[str, tuple[float | None, int]] = {}   # pid -> (t_ms, seq)
        if self._factory is None:
            log.warning("PyAV (av) is not installed -- live video decode is DISABLED; "
                        "tiles will stay 'waiting for video...'. Install it with: "
                        "pip install av")
        elif decoder_factory is None:
            log.info("live H.264 decode enabled (PyAV %s)", getattr(av, "__version__", "?"))

    @property
    def available(self) -> bool:
        return self._factory is not None

    def feed(self, phone_id: str, payload: bytes, t_ms: float | None = None) -> None:
        """Queue one access unit; *t_ms*: its host arrival time (optional)."""
        if self._factory is None:
            return
        with self._lock:
            w = self._workers.get(phone_id)
            if w is None:
                w = _Worker(self._factory(),
                            lambda img, t=None, pid=phone_id: self._set(pid, img, t))
                self._workers[phone_id] = w
        w.submit(payload, t_ms)

    def latest(self, phone_id: str) -> np.ndarray | None:
        with self._lock:
            return self._latest.get(phone_id)

    def latest_stamped(self, phone_id: str):
        """``(image, t_ms, seq)`` of the newest frame, or None. *t_ms* is the host
        arrival time of the packet that completed it (None if not given); *seq*
        increments with every decoded frame (to skip frames already seen)."""
        with self._lock:
            img = self._latest.get(phone_id)
            if img is None:
                return None
            t, seq = self._stamp.get(phone_id, (None, 0))
            return img, t, seq

    def remove(self, phone_id: str) -> None:
        with self._lock:
            w = self._workers.pop(phone_id, None)
            self._latest.pop(phone_id, None)
            self._stamp.pop(phone_id, None)
        if w is not None:
            w.stop()

    def stop(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
            self._latest.clear()
            self._stamp.clear()
        for w in workers:
            w.stop()

    def _set(self, phone_id: str, img: np.ndarray, t_ms: float | None = None) -> None:
        with self._lock:
            first = phone_id not in self._latest
            self._latest[phone_id] = img
            seq = self._stamp.get(phone_id, (None, 0))[1] + 1
            self._stamp[phone_id] = (t_ms, seq)
        if first:
            log.info("%s: first decoded frame %dx%d -- video is live",
                     phone_id, img.shape[1], img.shape[0])


class _Worker:
    """One decode thread with a depth-1 mailbox (keep newest, drop the rest)."""

    def __init__(self, decoder, on_frame: Callable[..., None]) -> None:
        self._decoder = decoder
        self._on_frame = on_frame
        self._pending: bytes | None = None
        self._pending_t: float | None = None
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, payload: bytes, t_ms: float | None = None) -> None:
        with self._cond:
            # Append so an access unit isn't split, but cap backlog: if the
            # decoder is keeping up this stays tiny; if not, we coalesce (the
            # newest packet's time stamps whatever the coalesced data decodes to).
            self._pending = (self._pending or b"") + payload
            self._pending_t = t_ms
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait()
                if not self._running:
                    return
                data, self._pending = self._pending, None
                t_ms = self._pending_t
            for img in self._decoder.decode(data):
                self._on_frame(img, t_ms)
