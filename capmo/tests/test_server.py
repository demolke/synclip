"""End-to-end server test with a mock phone socket (no Qt, no real camera)."""

from __future__ import annotations

import socket
import threading
import time

from .. import protocol as P
from ..server import CaptureServer


class MockPhone:
    """Connects to the server, answers PINGs, and can push META/VIDEO frames."""

    def __init__(self, host: str, port: int, phone_lead_ms: int = 100) -> None:
        self._sock = socket.create_connection((host, port), timeout=3.0)
        self._sock.settimeout(3.0)
        self._lead = phone_lead_ms
        self._buf = b""

    def answer_one_ping(self) -> bool:
        """Read one PING line and reply with a SYNC carrying a known offset."""
        while b"\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                return False
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        parts = line.decode().split()
        if not parts or parts[0] != "PING":
            return False
        t0 = int(parts[1])
        # Phone clock leads the server by self._lead ms.
        t1 = int(time.time() * 1000) + self._lead
        t2 = t1 + 1
        self._sock.sendall(P.build_frame(P.TYPE_SYNC, P.build_sync_payload(t0, t1, t2)))
        return True

    def send_keyframe(self, phone_ms: int, body: bytes = b"\x00\x00\x00\x01\x65idr") -> None:
        self._sock.sendall(P.build_frame(P.TYPE_META, P.build_meta_payload(phone_ms)))
        self._sock.sendall(P.build_frame(P.TYPE_VIDEO, body))

    def send_delta(self, body: bytes = b"\x00\x00\x00\x01\x61slice") -> None:
        self._sock.sendall(P.build_frame(P.TYPE_VIDEO, body))

    def close(self) -> None:
        self._sock.close()


def _wait(pred, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_phone_connects_syncs_and_streams_frames():
    frames: list[tuple] = []
    synced = threading.Event()

    srv = CaptureServer(host="127.0.0.1", port=0, ping_interval=0.1)
    srv.on_frame = lambda conn, payload, recv, key, meta, server_ms: frames.append(
        (conn.phone_id, key, meta, server_ms))
    srv.on_sync = lambda conn: synced.set()
    srv.start()
    try:
        phone = MockPhone("127.0.0.1", srv.port, phone_lead_ms=100)
        assert _wait(lambda: len(srv.connections()) == 1), "phone never registered"

        assert phone.answer_one_ping(), "no PING received"
        assert synced.wait(3.0), "server never processed a SYNC"
        conn = srv.connections()[0]
        assert conn.clock.ready
        assert abs(conn.clock.offset_ms - 100.0) < 5.0   # ~phone lead

        phone.send_keyframe(phone_ms=1_700_000_000_000)
        phone.send_delta()
        assert _wait(lambda: len(frames) == 2), f"frames not received: {frames}"

        # First frame: keyframe with META -> a server-timeline timestamp.
        _id, key0, meta0, server0 = frames[0]
        assert key0 is True and meta0 == 1_700_000_000_000
        assert abs(server0 - (meta0 - conn.clock.offset_ms)) < 1e-6
        # Second frame: delta, no META, recognised as non-key from the bitstream.
        _id, key1, meta1, server1 = frames[1]
        assert key1 is False and meta1 is None and server1 is None

        assert conn.frames == 2 and conn.keyframes == 1
        phone.close()
        assert _wait(lambda: len(srv.connections()) == 0), "disconnect not detected"
    finally:
        srv.stop()


def test_keyframe_recognised_without_meta_from_bitstream():
    frames: list[tuple] = []
    srv = CaptureServer(host="127.0.0.1", port=0, ping_interval=10.0)
    srv.on_frame = lambda conn, payload, recv, key, meta, server_ms: frames.append((key, meta))
    srv.start()
    try:
        phone = MockPhone("127.0.0.1", srv.port)
        assert _wait(lambda: len(srv.connections()) == 1)
        # IDR sent with no preceding META -> still flagged as a keyframe.
        phone.send_delta(b"\x00\x00\x00\x01\x65idr")
        assert _wait(lambda: len(frames) == 1)
        assert frames[0] == (True, None)
        phone.close()
    finally:
        srv.stop()
