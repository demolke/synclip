"""Decoder + live-preview hub tests.

The real H.264 round-trip needs PyAV (+ an encoder) and is skipped otherwise.
The hub's threading/queue/latest behaviour is tested with a fake decoder, so it
runs everywhere.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ..decode import LivePreviewHub


def _encode_h264(frames: list[np.ndarray], w: int, h: int) -> bytes:
    """Encode BGR frames to a raw H.264 (Annex-B) byte string via PyAV."""
    import io
    import av
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="h264")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for arr in frames:
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return buf.getvalue()


def test_h264_round_trip_decodes_frames():
    av = pytest.importorskip("av")
    from ..decode import H264Decoder, AVAILABLE
    assert AVAILABLE
    w, h, n = 64, 48, 12
    frames = [np.full((h, w, 3), i * 15, dtype=np.uint8) for i in range(n)]
    try:
        data = _encode_h264(frames, w, h)
    except av.error.FFmpegError as exc:  # encoder missing in this build
        pytest.skip(f"no usable h264 encoder: {exc}")

    dec = H264Decoder()
    got = dec.decode(data)
    assert got, "decoder produced no frames"
    assert got[0].shape == (h, w, 3) and got[0].dtype == np.uint8


class _FakeDecoder:
    """Turns each payload into one 2x2 frame whose pixels equal payload[0]."""
    def decode(self, payload):
        if not payload:
            return []
        return [np.full((2, 2, 3), payload[0], dtype=np.uint8)]


def _wait(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_hub_decodes_per_phone_and_keeps_latest():
    hub = LivePreviewHub(decoder_factory=_FakeDecoder)
    assert hub.available
    hub.feed("camA", bytes([10]))
    hub.feed("camB", bytes([20]))
    assert _wait(lambda: hub.latest("camA") is not None and hub.latest("camB") is not None)
    assert hub.latest("camA")[0, 0, 0] == 10
    assert hub.latest("camB")[0, 0, 0] == 20
    # A newer frame replaces the latest.
    hub.feed("camA", bytes([99]))
    assert _wait(lambda: hub.latest("camA")[0, 0, 0] == 99)
    hub.remove("camA")
    assert hub.latest("camA") is None
    hub.stop()


def test_hub_without_decoder_is_noop(monkeypatch):
    # Simulate PyAV being absent: the default factory resolves to None.
    monkeypatch.setattr("capmo.decode.AVAILABLE", False)
    hub = LivePreviewHub()
    assert not hub.available
    hub.feed("camA", b"\x01\x02")     # must not raise
    assert hub.latest("camA") is None
    hub.stop()
