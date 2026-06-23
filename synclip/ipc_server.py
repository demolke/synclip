"""
TCP server that streams blendshape frames to connected Godot/Blender
(or other) clients.

Connection handshake (negotiation), sent once, server -> client, immediately on
connect, BEFORE any frames:
    magic   : uint32 = 0xAF0001  (HELLO)
    length  : uint32            (byte length of the JSON payload that follows)
    payload : <length> bytes UTF-8 JSON {"blendshapes": [...52 names...]}

The client maps those names to its own rig and reports the result back, so the
server can log exactly how the rig was mapped (and catch order mismatches):
    magic   : uint32 = 0xAF0005  (REPORT)
    length  : uint32
    payload : <length> bytes UTF-8 JSON
              {"client": "...", "object": "...", "mapped_count": N,
               "total": 52, "mapping": {blendshape_name: rig_target, ...},
               "unmapped": [...]}

Binary frame protocol (little-endian, 244 bytes per frame):
    magic       : uint32  = 0xAF0002
    audio_pos   : float64           (ms)
    blendshapes : float32 x 52
    head_rot    : float32 x 3        (euler degrees: x, y, z)
    head_pos    : float32 x 3        (translation: x, y, z)
    TOTAL       : 4 + 8 + 208 + 12 + 12 = 244 bytes
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import TYPE_CHECKING, Callable

from .arkit_names import BLENDSHAPE_NAMES

_MAGIC: int = 0xAF0002
# The magic doubles as a frame-mode tag: the high 3 bytes are constant
# (0xAF00xx) and the low byte selects how the receiver should treat the frame.
# Live viewers (Godot) accept any 0xAF00xx and just display; the Blender client
# uses the mode to decide insert-keyframe vs move-playhead vs update-keyframe.
HELLO_MAGIC: int = 0xAF0001  # handshake: server announces blendshape names
MODE_LIVE: int = 0xAF0002    # live preview / recording -> insert keyframes
MODE_REVIEW: int = 0xAF0003  # playback / scrub -> move playhead only, no insert
MODE_EDIT: int = 0xAF0004    # value edit -> overwrite keyframe at this position
REPORT_MAGIC: int = 0xAF0005  # handshake reply: client reports its mapping
# magic + audio_pos + 52 blendshapes + 3 head rot + 3 head pos
_FRAME_STRUCT = struct.Struct("<I d 52f 3f 3f")  # 4 + 8 + 208 + 12 + 12 = 244 bytes
_MSG_HEADER = struct.Struct("<I I")  # magic + payload length (HELLO / REPORT)

assert _FRAME_STRUCT.size == 244, f"Unexpected frame size: {_FRAME_STRUCT.size}"


class IPCServer:
    """TCP server for real-time blendshape streaming."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9876) -> None:
        self._host = host
        self._port = port

        self._server_sock: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._clients_lock = threading.Lock()

        self._running = False
        self._accept_thread: threading.Thread | None = None

        # Optional callback(dict) invoked when a client reports its mapping.
        self.on_mapping: Callable[[dict], None] | None = None

        # Cached HELLO message (blendshape-name negotiation), built once.
        payload = json.dumps({"blendshapes": list(BLENDSHAPE_NAMES)}).encode("utf-8")
        self._hello = _MSG_HEADER.pack(HELLO_MAGIC, len(payload)) + payload

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind, listen, and start accepting clients on a daemon thread."""
        if self._running:
            return

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._server_sock.bind((self._host, self._port))
            self._server_sock.listen(16)
            # Update port in case 0 was passed (OS picks a free port).
            self._port = self._server_sock.getsockname()[1]
        except OSError as exc:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None
            print(f"[IPCServer] could not bind {self._host}:{self._port}: {exc}")
            return
        self._server_sock.settimeout(1.0)  # allows clean shutdown

        self._running = True

        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="IPCServer-accept"
        )
        self._accept_thread.start()

    def stop(self) -> None:
        """Signal the server to stop and close all connections."""
        self._running = False

        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        with self._clients_lock:
            for client in self._clients:
                try:
                    client.close()
                except OSError:
                    pass
            self._clients.clear()

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=3.0)
            self._accept_thread = None

    # ------------------------------------------------------------------
    # Frame sending
    # ------------------------------------------------------------------

    def send_frame(
        self,
        audio_pos_ms: float,
        blendshapes: list[float],
        head_pose: dict | None = None,
        mode: int = MODE_LIVE,
    ) -> None:
        """Pack one binary frame and send it to all connected clients.

        Clients that have disconnected are silently removed.
        *blendshapes* is padded with 0.0 or truncated to exactly 52 values.
        *head_pose* is {"rot": [x,y,z], "pos": [x,y,z]} or None (treated as zero).
        *mode* is one of MODE_LIVE / MODE_REVIEW / MODE_EDIT (see above); it is
        written as the frame magic so receivers can branch on it.
        """
        # Normalise length to exactly 52 floats.
        bs = list(blendshapes)
        if len(bs) < 52:
            bs.extend([0.0] * (52 - len(bs)))
        elif len(bs) > 52:
            bs = bs[:52]

        if head_pose:
            rot = head_pose.get("rot", [0.0, 0.0, 0.0])
            pos = head_pose.get("pos", [0.0, 0.0, 0.0])
        else:
            rot = [0.0, 0.0, 0.0]
            pos = [0.0, 0.0, 0.0]

        data = _FRAME_STRUCT.pack(mode, audio_pos_ms, *bs, *rot[:3], *pos[:3])

        dead: list[socket.socket] = []
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.sendall(data)
                except OSError:
                    # Includes a send timeout (a stalled client) - drop it rather
                    # than blocking the capture pipeline.
                    dead.append(client)
            for d in dead:
                try:
                    self._clients.remove(d)
                except ValueError:
                    pass  # already removed by _watch_client
                try:
                    d.close()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client_count(self) -> int:
        """Number of currently connected clients."""
        with self._clients_lock:
            return len(self._clients)

    @property
    def host(self) -> str:
        """Host the server is bound to."""
        return self._host

    @property
    def port(self) -> int:
        """Port the server is listening on."""
        return self._port

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        """Background thread: accept new client connections."""
        while self._running and self._server_sock is not None:
            try:
                client_sock, _addr = self._server_sock.accept()
            except TimeoutError:
                continue
            except OSError:
                # Server socket was closed - exit gracefully.
                break

            # A send timeout means one stalled viewer can't block the whole
            # capture pipeline; send_frame drops it on timeout (OSError).
            client_sock.settimeout(2.0)

            # Send the HELLO handshake FIRST, before this socket is eligible for
            # frame broadcasts, so the name list is guaranteed to be the very
            # first bytes the client reads.
            try:
                client_sock.sendall(self._hello)
            except OSError:
                try:
                    client_sock.close()
                except OSError:
                    pass
                continue

            with self._clients_lock:
                self._clients.append(client_sock)

            # Spawn a lightweight thread just to detect disconnection.
            t = threading.Thread(
                target=self._watch_client,
                args=(client_sock,),
                daemon=True,
                name="IPCServer-client",
            )
            t.start()

    def _watch_client(self, client_sock: socket.socket) -> None:
        """Read the client's mapping REPORT (if any), then watch for disconnect.

        The client may send one length-prefixed REPORT message describing how it
        mapped our blendshape names onto its rig. We parse and log it, then keep
        draining bytes purely to detect when the client goes away.
        """
        buf = b""
        report_done = False
        try:
            while True:
                try:
                    data = client_sock.recv(4096)
                except socket.timeout:
                    # The send timeout also applies to recv; a quiet client is
                    # not a disconnected one, so keep waiting.
                    continue
                if not data:
                    break
                if report_done:
                    continue  # already have the report; just detecting close
                buf += data
                # Need at least a header to know the payload length.
                while len(buf) >= _MSG_HEADER.size and not report_done:
                    magic, length = _MSG_HEADER.unpack_from(buf, 0)
                    if magic != REPORT_MAGIC:
                        # Not a report stream we understand - stop parsing and
                        # fall back to pure disconnect detection.
                        report_done = True
                        break
                    if len(buf) < _MSG_HEADER.size + length:
                        break  # wait for the rest of the payload
                    payload = buf[_MSG_HEADER.size:_MSG_HEADER.size + length]
                    buf = buf[_MSG_HEADER.size + length:]
                    report_done = True
                    self._handle_report(payload)
        except OSError:
            pass
        finally:
            with self._clients_lock:
                try:
                    self._clients.remove(client_sock)
                except ValueError:
                    pass
            try:
                client_sock.close()
            except OSError:
                pass

    def _handle_report(self, payload: bytes) -> None:
        """Parse and log a client mapping report."""
        try:
            report = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            print(f"[IPCServer] received an unparseable mapping report: {exc}")
            return

        client = report.get("client", "?")
        obj = report.get("object", "?")
        mapped = report.get("mapped_count", 0)
        total = report.get("total", len(BLENDSHAPE_NAMES))
        mapping = report.get("mapping", {})
        unmapped = report.get("unmapped", [])

        print(f"[IPCServer] {client!r} mapped {mapped}/{total} blendshapes "
              f"onto '{obj}':")
        for name in BLENDSHAPE_NAMES:
            target = mapping.get(name)
            if target:
                print(f"[IPCServer]   {name:24s} -> {target}")
        if unmapped:
            print(f"[IPCServer]   unmapped ({len(unmapped)}): {', '.join(unmapped)}")

        if self.on_mapping is not None:
            try:
                self.on_mapping(report)
            except Exception:
                pass
