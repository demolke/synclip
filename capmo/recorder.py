"""
Glue between the live :class:`CaptureServer` and an on-disk
:class:`RecordingSession`: when recording is armed, every frame the server
dispatches is written verbatim with its timing metadata.

Kept separate from both so the server stays a pure transport and the session
stays pure filesystem; this is the small stateful controller the UI (or the
headless runner) drives. Frame listeners can be chained, so a live preview can
observe the same frames without the recorder knowing about it.

While recording, a background thread samples every connected phone's clock
estimate (offset + rtt, best and latest raw exchange) about once per second
into ``<slug>.clock.jsonl``, and the latest SPS/PPS seen on each live stream is
handed to a new session so a recording armed mid-stream stays decodable.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from .core import h264
from .server import CaptureServer, PhoneConnection
from .session import RecordingSession


class RecorderController:
    def __init__(self, server: CaptureServer, root_dir: str,
                 clock_interval: float = 1.0) -> None:
        self._server = server
        self._root_dir = root_dir
        self._session: RecordingSession | None = None
        self._lock = threading.Lock()
        # Extra frame observers (e.g. a live preview), called after recording.
        self._listeners: list[Callable[..., None]] = []
        self._codec_config: dict[str, bytes] = {}     # phone_id -> last SPS/PPS seen
        self._clock_interval = clock_interval
        self._clock_stop = threading.Event()
        self._clock_thread: threading.Thread | None = None
        self.last_session_dir: str | None = None
        server.on_frame = self._on_frame

    def add_frame_listener(self, fn: Callable[..., None]) -> None:
        self._listeners.append(fn)

    @property
    def recording(self) -> bool:
        return self._session is not None

    @property
    def session_dir(self) -> str | None:
        return self._session.dir if self._session else None

    def start(self, name: str | None = None, info: dict | None = None) -> str:
        """Arm recording (idempotent). *info*: session-level manifest fields
        (``nominal_fps``, ``requested_size``, ``preset``, ``board`` ...)."""
        with self._lock:
            if self._session is None:
                self._session = RecordingSession(self._root_dir, name, info=info)
                for pid, cfg in self._codec_config.items():
                    self._session.set_codec_config(pid, cfg)
                self._clock_stop.clear()
                self._clock_thread = threading.Thread(target=self._clock_loop,
                                                      args=(self._session,), daemon=True)
                self._clock_thread.start()
            return self._session.dir

    def update_info(self, **info) -> bool:
        """Merge session-level manifest fields into the running recording (they
        are written with the manifest at :meth:`stop`). False if not recording."""
        with self._lock:
            sess = self._session
            if sess is None:
                return False
            sess.info.update(info)
            return True

    def stop(self) -> str | None:
        """Disarm and finalize the session; returns its directory (or None)."""
        with self._lock:
            sess = self._session
            self._session = None
        self._clock_stop.set()
        t = self._clock_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._clock_thread = None
        if sess is not None:
            self.sample_clocks(sess)
            offsets = {c.phone_id: c.clock.offset_ms for c in self._server.connections()}
            sess.close(clock_offsets=offsets)
            self.last_session_dir = sess.dir
            return sess.dir
        return None

    # -- clock sampling ------------------------------------------------------

    def sample_clocks(self, sess: RecordingSession | None = None) -> int:
        """Record one clock sample per synced phone. Returns how many."""
        sess = sess or self._session
        if sess is None:
            return 0
        n = 0
        now = time.time() * 1000.0
        for c in self._server.connections():
            clk = c.clock
            if not getattr(clk, "ready", False):
                continue
            last = getattr(clk, "last_sample", None)
            sess.add_clock_sample(
                c.phone_id, now, clk.offset_ms, clk.rtt_ms,
                raw_offset_ms=getattr(last, "offset", None),
                raw_rtt_ms=getattr(last, "rtt", None),
                raw_t_ms=getattr(last, "t_ms", None))
            n += 1
        return n

    def _clock_loop(self, sess: RecordingSession) -> None:
        while not self._clock_stop.wait(self._clock_interval):
            try:
                self.sample_clocks(sess)
            except Exception:          # never let sampling kill recording
                pass

    # -- server callback -----------------------------------------------------

    def _on_frame(self, conn: PhoneConnection, payload: bytes, recv_ms: float,
                  is_key: bool, meta_ms: int | None, server_ms: float | None) -> None:
        if is_key or len(payload) < 1024:
            # cheap (bytes.find) scan; keeps the latest codec config per phone
            cfg = h264.codec_config(payload)
            if cfg:
                self._codec_config[conn.phone_id] = cfg
        with self._lock:
            sess = self._session
        if sess is not None:
            try:
                sess.write_frame(conn.phone_id, payload, recv_ms, is_key, meta_ms, server_ms)
            except (RuntimeError, OSError, ValueError):   # closed under us
                pass
        for fn in self._listeners:
            fn(conn, payload, recv_ms, is_key, meta_ms, server_ms)
