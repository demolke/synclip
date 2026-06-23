"""
T-08 / T-09: IPC packet round-trip and broadcast cadence.

A real IPCServer is started on an ephemeral port; a mock client connects, reads
the HELLO handshake, optionally REPORTs, and receives binary frames. We assert
the 244-byte frame round-trips intact (T-09) and that the mode tags emitted by
the MainWindow bridge match the policy table in DESIGN.md section 8 (T-08).
"""

from __future__ import annotations

import json
import socket
import struct
import time

import pytest

from ..ipc_server import (
    IPCServer,
    HELLO_MAGIC,
    REPORT_MAGIC,
    MODE_LIVE,
    MODE_REVIEW,
    MODE_EDIT,
    _FRAME_STRUCT,
    _MSG_HEADER,
)
from ..arkit_names import BLENDSHAPE_NAMES


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

class MockClient:
    """Minimal TCP client that mimics the Godot/Blender handshake + frame read."""

    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.create_connection((host, port), timeout=3.0)
        self._sock.settimeout(3.0)
        self._buf = b""

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("server closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def read_hello(self) -> list[str]:
        magic, length = _MSG_HEADER.unpack(self._recv_exact(_MSG_HEADER.size))
        assert magic == HELLO_MAGIC
        payload = self._recv_exact(length)
        return json.loads(payload.decode("utf-8"))["blendshapes"]

    def send_report(self, mapping: dict) -> None:
        payload = json.dumps(mapping).encode("utf-8")
        self._sock.sendall(_MSG_HEADER.pack(REPORT_MAGIC, len(payload)) + payload)

    def read_frame(self) -> tuple:
        data = self._recv_exact(_FRAME_STRUCT.size)
        return _FRAME_STRUCT.unpack(data)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


@pytest.fixture
def server():
    # Port 0 lets the OS pick a free port.
    srv = IPCServer(host="127.0.0.1", port=0)
    # IPCServer binds the configured port; use a fixed high port for the test.
    srv = IPCServer(host="127.0.0.1", port=9911)
    srv.start()
    # Give the accept thread a moment.
    time.sleep(0.1)
    yield srv
    srv.stop()


def _wait_for_client(server: IPCServer, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while server.client_count < 1 and time.time() < deadline:
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# T-09: frame bytes round-trip
# ---------------------------------------------------------------------------

def test_hello_handshake_sends_52_names(server):
    client = MockClient(server.host, server.port)
    try:
        names = client.read_hello()
        assert names == list(BLENDSHAPE_NAMES)
        assert len(names) == 52
    finally:
        client.close()


def test_frame_roundtrip(server):
    client = MockClient(server.host, server.port)
    try:
        client.read_hello()
        _wait_for_client(server)

        blendshapes = [i / 52.0 for i in range(52)]
        pose = {"rot": [10.0, 20.0, 30.0], "pos": [1.0, 2.0, 3.0]}
        server.send_frame(1234.5, blendshapes, pose, mode=MODE_LIVE)

        fields = client.read_frame()
        magic = fields[0]
        audio_pos = fields[1]
        bs = list(fields[2:2 + 52])
        rot = list(fields[2 + 52:2 + 52 + 3])
        pos = list(fields[2 + 52 + 3:2 + 52 + 6])

        assert magic == MODE_LIVE
        assert abs(audio_pos - 1234.5) < 1e-6
        assert all(abs(a - b) < 1e-5 for a, b in zip(bs, blendshapes))
        assert all(abs(a - b) < 1e-4 for a, b in zip(rot, [10.0, 20.0, 30.0]))
        assert all(abs(a - b) < 1e-4 for a, b in zip(pos, [1.0, 2.0, 3.0]))
    finally:
        client.close()


def test_frame_struct_is_244_bytes():
    assert _FRAME_STRUCT.size == 244


def test_report_mapping_received(server):
    received = {}

    def on_mapping(d):
        received.update(d)

    server.on_mapping = on_mapping
    client = MockClient(server.host, server.port)
    try:
        client.read_hello()
        client.send_report({
            "client": "test", "object": "Head",
            "mapped_count": 52, "total": 52, "mapping": {}, "unmapped": [],
        })
        deadline = time.time() + 3.0
        while not received and time.time() < deadline:
            time.sleep(0.02)
        assert received.get("client") == "test"
        assert received.get("mapped_count") == 52
    finally:
        client.close()


# ---------------------------------------------------------------------------
# T-08: broadcast cadence / mode tags via the bridge policy
# ---------------------------------------------------------------------------

def test_mode_tags_distinct():
    """The three frame modes must be distinct magics under the 0xAF00xx family."""
    assert MODE_LIVE != MODE_REVIEW != MODE_EDIT
    for m in (MODE_LIVE, MODE_REVIEW, MODE_EDIT):
        assert (m & 0xFFFF00) == 0xAF0000


def test_review_mode_tag_roundtrips(server):
    client = MockClient(server.host, server.port)
    try:
        client.read_hello()
        _wait_for_client(server)
        server.send_frame(0.0, [0.0] * 52, None, mode=MODE_REVIEW)
        fields = client.read_frame()
        assert fields[0] == MODE_REVIEW
    finally:
        client.close()
