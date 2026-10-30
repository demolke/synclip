"""
capmo wire protocol (matches the StreamCam Android client).

A phone is a TCP **client** that connects to the capmo desktop **server**. Two
directions, two encodings:

Server -> phone : newline-terminated **text commands**::

    PING <t0_ms>          # clock-sync probe; t0 is the server's send time (ms)
    START <w> <h>         # (re)start the encoder at this resolution
    STOP
    FRAMERATE <fps>
    IFRAME <seconds>      # keyframe interval
    FOCUS <x> <y>         # normalised 0..1
    ZOOM <ratio>

Phone -> server : length-prefixed **binary frames**, big-endian (Java
``ByteBuffer`` default order)::

    [type:uint8][length:int32][payload: <length> bytes]

    type 0x01 VIDEO : raw H.264 (Annex-B, Baseline) access-unit bytes
    type 0x02 META  : int64 phone wall-clock ms, sent immediately *before* the
                      next keyframe (so it tags that keyframe's capture time)
    type 0x03 SYNC  : t0,t1,t2 (3x int64) -- reply to a PING (see timesync.py)

This module is pure (no sockets, no Qt), so it is unit-testable and reusable by
both capmo and, eventually, synclip's IPC layer.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# -- message types (phone -> server) ----------------------------------------
TYPE_VIDEO = 0x01
TYPE_META = 0x02
TYPE_SYNC = 0x03

_HEADER = struct.Struct(">BI")   # type (uint8) + length (int32), big-endian
_I64 = struct.Struct(">q")       # one int64, big-endian
_SYNC = struct.Struct(">qqq")    # t0, t1, t2

HEADER_SIZE = _HEADER.size       # 5
SYNC_PAYLOAD_SIZE = _SYNC.size   # 24
META_PAYLOAD_SIZE = _I64.size    # 8


@dataclass
class Message:
    """One decoded phone->server message."""
    type: int
    payload: bytes


class FrameDecoder:
    """Incremental parser for the phone->server byte stream.

    Feed it whatever bytes arrive off the socket; it yields whole
    :class:`Message` objects as they complete and buffers the remainder. This
    decouples message framing from the (blocking/async) transport.
    """

    # Guard against a corrupt/desynced length claiming a huge allocation. Frames
    # are H.264 access units; a few MB is already very generous.
    MAX_PAYLOAD = 16 * 1024 * 1024

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[Message]:
        self._buf.extend(data)
        out: list[Message] = []
        while True:
            if len(self._buf) < HEADER_SIZE:
                break
            mtype, length = _HEADER.unpack_from(self._buf, 0)
            if length < 0 or length > self.MAX_PAYLOAD:
                raise ValueError(f"implausible frame length {length} (type {mtype})")
            if len(self._buf) < HEADER_SIZE + length:
                break
            start = HEADER_SIZE
            payload = bytes(self._buf[start:start + length])
            del self._buf[:start + length]
            out.append(Message(mtype, payload))
        return out

    @property
    def buffered(self) -> int:
        return len(self._buf)


# -- payload (de)serialisation ----------------------------------------------

def parse_sync(payload: bytes) -> tuple[int, int, int]:
    """Return (t0, t1, t2) from a SYNC payload."""
    if len(payload) != SYNC_PAYLOAD_SIZE:
        raise ValueError(f"SYNC payload must be {SYNC_PAYLOAD_SIZE} bytes, got {len(payload)}")
    return _SYNC.unpack(payload)


def parse_meta_ms(payload: bytes) -> int:
    """Return the phone wall-clock ms from a META payload."""
    if len(payload) != META_PAYLOAD_SIZE:
        raise ValueError(f"META payload must be {META_PAYLOAD_SIZE} bytes, got {len(payload)}")
    return _I64.unpack(payload)[0]


def build_frame(mtype: int, payload: bytes) -> bytes:
    """Encode one framed message (used by tests / a mock phone)."""
    return _HEADER.pack(mtype, len(payload)) + payload


def build_sync_payload(t0: int, t1: int, t2: int) -> bytes:
    return _SYNC.pack(t0, t1, t2)


def build_meta_payload(phone_ms: int) -> bytes:
    return _I64.pack(phone_ms)


# -- server -> phone text commands ------------------------------------------

def cmd_ping(t0_ms: int) -> bytes:
    return f"PING {int(t0_ms)}\n".encode("ascii")


def cmd_start(width: int, height: int) -> bytes:
    return f"START {int(width)} {int(height)}\n".encode("ascii")


def cmd_stop() -> bytes:
    return b"STOP\n"


def cmd_framerate(fps: int) -> bytes:
    return f"FRAMERATE {int(fps)}\n".encode("ascii")


def cmd_iframe(seconds: int) -> bytes:
    return f"IFRAME {int(seconds)}\n".encode("ascii")


def cmd_focus(x: float, y: float) -> bytes:
    return f"FOCUS {float(x)} {float(y)}\n".encode("ascii")


def cmd_zoom(ratio: float) -> bytes:
    return f"ZOOM {float(ratio)}\n".encode("ascii")


# -- H.264 helpers -----------------------------------------------------------

def annexb_nal_types(payload: bytes) -> list[int]:
    """Return the H.264 NAL unit types present in an Annex-B *payload*.

    Used to recognise keyframes (IDR = type 5) and parameter sets (SPS=7, PPS=8)
    independently of the META signal, so recordings are robust if a META is ever
    dropped.
    """
    types: list[int] = []
    i, n = 0, len(payload)
    while i + 3 < n:
        # start code 00 00 01 or 00 00 00 01
        if payload[i] == 0 and payload[i + 1] == 0:
            if payload[i + 2] == 1:
                hdr = i + 3
            elif payload[i + 2] == 0 and i + 3 < n and payload[i + 3] == 1:
                hdr = i + 4
            else:
                i += 1
                continue
            if hdr < n:
                types.append(payload[hdr] & 0x1F)
            i = hdr + 1
        else:
            i += 1
    return types


def is_keyframe(payload: bytes) -> bool:
    """True if the access unit contains an IDR slice (NAL type 5)."""
    return 5 in annexb_nal_types(payload)
