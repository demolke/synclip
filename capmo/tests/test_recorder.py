"""RecorderController: arm/disarm + verbatim capture + frame fan-out."""

from __future__ import annotations

import json
import os

from ..server import CaptureServer
from ..recorder import RecorderController


class _FakeConn:
    def __init__(self, phone_id):
        self.phone_id = phone_id
        class _C:
            ready = True
            offset_ms = 42.0
        self.clock = _C()


def test_records_only_while_armed_and_fans_out(tmp_path):
    srv = CaptureServer(host="127.0.0.1", port=0)   # not started; we call _on_frame directly
    rec = RecorderController(srv, str(tmp_path))
    seen = []
    rec.add_frame_listener(lambda *a: seen.append(a[1]))   # payload
    conn = _FakeConn("camA")

    # Before arming: nothing recorded, but listeners still fire.
    rec._on_frame(conn, b"pre", 1.0, True, 1, 0.9)
    assert not rec.recording and seen == [b"pre"]

    d = rec.start()
    rec._on_frame(conn, b"\x00idr", 2.0, True, 2, 1.9)
    rec._on_frame(conn, b"delta", 3.0, False, None, None)
    rec.stop()

    assert seen == [b"pre", b"\x00idr", b"delta"]
    h264 = os.path.join(d, "camA.h264")
    assert open(h264, "rb").read() == b"\x00idr" + b"delta"   # 'pre' excluded
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert man["phones"][0]["frames"] == 2
