"""
capmo capture server: accept connections from phones, sync their clocks, and
hand decoded protocol messages to a sink (recorder / live preview).

Threaded, like synclip's ``ipc_server``: one accept thread, one reader thread
per phone, plus a periodic PING thread for clock sync. No Qt here, so it runs
headless and is unit-testable with a mock phone socket.

The server is the TCP listener; each phone dials in. Phone identity is not in
the wire protocol yet, so a connection is keyed by peer IP (stable on a LAN);
a human label ``cam_N`` is assigned in arrival order.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable

from . import protocol
from .core.sync.clock import ClockEstimator

log = logging.getLogger("capmo.server")


def now_ms() -> float:
    """Server wall clock in ms since the epoch (same domain as the phone's
    ``System.currentTimeMillis()``)."""
    return time.time() * 1000.0


def local_ip() -> str:
    """Best-guess primary LAN IPv4 of this host (what a phone should dial).

    Opens a UDP socket toward a public address to learn which interface the OS
    would route through -- no packets are actually sent. Falls back to loopback.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class PhoneConnection:
    """Per-phone state and the only place that writes to its socket."""

    def __init__(self, phone_id: str, label: str, sock: socket.socket, addr) -> None:
        self.phone_id = phone_id
        self.label = label
        self.addr = addr
        self._sock = sock
        self._send_lock = threading.Lock()
        self.decoder = protocol.FrameDecoder()
        self.clock = ClockEstimator()
        self._pending_meta_ms: int | None = None
        # live stats
        self.frames = 0
        self.keyframes = 0
        self.bytes = 0
        self.connected_ms = now_ms()
        self.last_recv_ms: float | None = None
        self._recent_recv: list[float] = []   # for an fps estimate
        # one-shot logging milestones
        self._logged_first_bytes = False
        self._logged_first_frame = False
        self._logged_sync = False

    # -- outbound (server -> phone) -----------------------------------------

    def send(self, data: bytes) -> bool:
        """Send a (text command) line; returns False if the socket is dead."""
        with self._send_lock:
            try:
                self._sock.sendall(data)
                return True
            except OSError:
                return False

    # -- stats ---------------------------------------------------------------

    def note_frame(self, recv_ms: float, size: int, is_key: bool) -> None:
        self.frames += 1
        self.bytes += size
        if is_key:
            self.keyframes += 1
        self.last_recv_ms = recv_ms
        self._recent_recv.append(recv_ms)
        if len(self._recent_recv) > 30:
            self._recent_recv.pop(0)

    @property
    def fps(self) -> float:
        if len(self._recent_recv) < 2:
            return 0.0
        span = self._recent_recv[-1] - self._recent_recv[0]
        return (len(self._recent_recv) - 1) / (span / 1000.0) if span > 0 else 0.0


class CaptureServer:
    """Accepts phones, runs clock sync, and dispatches frames to callbacks.

    Callbacks (set as attributes, all optional)::

        on_connect(conn)
        on_disconnect(conn)
        on_frame(conn, payload, recv_ms, is_key, meta_ms, server_ms)
        on_sync(conn)        # after each clock exchange
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 12345,
                 ping_interval: float = 2.0) -> None:
        self._host = host
        self._port = port
        self._ping_interval = ping_interval
        self._server_sock: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._conns: dict[str, PhoneConnection] = {}
        self._lock = threading.Lock()
        self._running = False
        self._label_seq = 0

        self.on_connect: Callable[[PhoneConnection], None] | None = None
        self.on_disconnect: Callable[[PhoneConnection], None] | None = None
        self.on_frame: Callable[..., None] | None = None
        self.on_sync: Callable[[PhoneConnection], None] | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self._host, self._port))
        self._port = self._server_sock.getsockname()[1]   # resolve port 0
        self._server_sock.listen(8)
        self._running = True
        log.info("listening on %s:%d  (point the phone at  %s:%d )",
                 self._host, self._port, local_ip(), self._port)
        for target in (self._accept_loop, self._ping_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        with self._lock:
            conns = list(self._conns.values())
        for c in conns:
            try:
                c._sock.close()
            except OSError:
                pass

    @property
    def port(self) -> int:
        return self._port

    def connections(self) -> list[PhoneConnection]:
        with self._lock:
            return list(self._conns.values())

    # -- control (server -> phone) ------------------------------------------

    def broadcast(self, data: bytes) -> None:
        for c in self.connections():
            c.send(data)

    def send_to(self, phone_id: str, data: bytes) -> bool:
        with self._lock:
            c = self._conns.get(phone_id)
        return c.send(data) if c else False

    # -- internals -----------------------------------------------------------

    def _accept_loop(self) -> None:
        log.debug("accept loop started")
        while self._running:
            try:
                sock, addr = self._server_sock.accept()
            except OSError as exc:
                if self._running:
                    log.warning("accept failed: %s", exc)
                break
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            phone_id = addr[0]
            with self._lock:
                self._label_seq += 1
                label = f"cam_{self._label_seq}"
                replaced = phone_id in self._conns
                conn = PhoneConnection(phone_id, label, sock, addr)
                self._conns[phone_id] = conn
                n = len(self._conns)
            log.info("TCP connection accepted from %s:%d -> %s (%d connected)%s",
                     addr[0], addr[1], label, n,
                     "  [replacing an existing connection from this IP]" if replaced else "")
            if self.on_connect:
                self.on_connect(conn)
            t = threading.Thread(target=self._read_loop, args=(conn,), daemon=True)
            t.start()
        log.debug("accept loop ended")

    def _read_loop(self, conn: PhoneConnection) -> None:
        reason = "server stopping"
        try:
            while self._running:
                try:
                    chunk = conn._sock.recv(65536)
                except OSError as exc:
                    reason = f"recv error: {exc}"
                    break
                if not chunk:
                    reason = "peer closed the connection (recv returned 0 bytes)"
                    break
                if not conn._logged_first_bytes:
                    conn._logged_first_bytes = True
                    log.info("%s: first %d bytes from phone: %s%s", conn.label,
                             len(chunk), chunk[:32].hex(),
                             " ..." if len(chunk) > 32 else "")
                else:
                    log.debug("%s: recv %d bytes", conn.label, len(chunk))
                try:
                    msgs = list(conn.decoder.feed(chunk))
                except Exception as exc:        # malformed framing shouldn't be silent
                    log.warning("%s: protocol decode error (%s); is the phone using "
                                "capmo's framing? dropping connection", conn.label, exc)
                    reason = f"decode error: {exc}"
                    break
                for msg in msgs:
                    log.debug("%s: message type=0x%02x len=%d",
                              conn.label, msg.type, len(msg.payload))
                    self._dispatch(conn, msg)
        finally:
            with self._lock:
                # Only drop it if this exact connection is still the registered one.
                if self._conns.get(conn.phone_id) is conn:
                    del self._conns[conn.phone_id]
            try:
                conn._sock.close()
            except OSError:
                pass
            log.info("%s (%s) disconnected after %d frames: %s",
                     conn.label, conn.phone_id, conn.frames, reason)
            if self.on_disconnect:
                self.on_disconnect(conn)

    def _dispatch(self, conn: PhoneConnection, msg: protocol.Message) -> None:
        if msg.type == protocol.TYPE_SYNC:
            t3 = now_ms()
            try:
                t0, t1, t2 = protocol.parse_sync(msg.payload)
            except ValueError:
                log.warning("%s: bad SYNC payload (%d bytes)", conn.label, len(msg.payload))
                return
            conn.clock.add_exchange(t0, t1, t2, t3)
            if conn.clock.ready and not conn._logged_sync:
                conn._logged_sync = True
                log.info("%s: clock synced (offset=%+.0fms rtt=%.0fms)",
                         conn.label, conn.clock.offset_ms, conn.clock.rtt_ms)
            if self.on_sync:
                self.on_sync(conn)
        elif msg.type == protocol.TYPE_META:
            try:
                conn._pending_meta_ms = protocol.parse_meta_ms(msg.payload)
            except ValueError:
                log.warning("%s: bad META payload (%d bytes)", conn.label, len(msg.payload))
                conn._pending_meta_ms = None
        elif msg.type == protocol.TYPE_VIDEO:
            recv = now_ms()
            meta = conn._pending_meta_ms
            conn._pending_meta_ms = None
            # A META precedes a keyframe; also verify from the bitstream so a
            # dropped META doesn't mislabel an IDR.
            is_key = meta is not None or protocol.is_keyframe(msg.payload)
            server_ms = conn.clock.to_server_ms(meta) if (meta is not None and conn.clock.ready) else None
            conn.note_frame(recv, len(msg.payload), is_key)
            if not conn._logged_first_frame:
                conn._logged_first_frame = True
                log.info("%s: first VIDEO frame (%d bytes, keyframe=%s, clock_ready=%s)",
                         conn.label, len(msg.payload), is_key, conn.clock.ready)
            else:
                log.debug("%s: VIDEO frame (%d bytes, key=%s)",
                          conn.label, len(msg.payload), is_key)
            if self.on_frame:
                self.on_frame(conn, msg.payload, recv, is_key, meta, server_ms)
        else:
            log.warning("%s: unknown message type 0x%02x (%d bytes) -- protocol mismatch?",
                        conn.label, msg.type, len(msg.payload))

    def _ping_loop(self) -> None:
        # Stagger so a fresh connection gets a probe quickly.
        while self._running:
            for c in self.connections():
                c.send(protocol.cmd_ping(int(now_ms())))
            time.sleep(self._ping_interval)
