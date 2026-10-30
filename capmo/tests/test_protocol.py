"""Wire-protocol codec tests (framing, payloads, H.264 keyframe detection)."""

from __future__ import annotations

import pytest

from .. import protocol as P


def test_frame_decoder_splits_multiple_messages():
    dec = P.FrameDecoder()
    blob = (P.build_frame(P.TYPE_META, P.build_meta_payload(123))
            + P.build_frame(P.TYPE_VIDEO, b"abc"))
    msgs = dec.feed(blob)
    assert [(m.type, m.payload) for m in msgs] == [
        (P.TYPE_META, P.build_meta_payload(123)),
        (P.TYPE_VIDEO, b"abc"),
    ]
    assert dec.buffered == 0


def test_frame_decoder_handles_partial_and_split_feeds():
    dec = P.FrameDecoder()
    frame = P.build_frame(P.TYPE_VIDEO, b"hello")
    assert dec.feed(frame[:3]) == []          # header incomplete
    assert dec.feed(frame[3:7]) == []          # payload incomplete
    msgs = dec.feed(frame[7:])
    assert len(msgs) == 1 and msgs[0].payload == b"hello"


def test_frame_decoder_rejects_implausible_length():
    dec = P.FrameDecoder()
    import struct
    bad = struct.pack(">BI", P.TYPE_VIDEO, P.FrameDecoder.MAX_PAYLOAD + 1)
    with pytest.raises(ValueError):
        dec.feed(bad)


def test_sync_payload_roundtrip():
    payload = P.build_sync_payload(1000, 1005, 1006)
    assert P.parse_sync(payload) == (1000, 1005, 1006)


def test_meta_payload_roundtrip():
    assert P.parse_meta_ms(P.build_meta_payload(1_700_000_000_123)) == 1_700_000_000_123


def test_commands_are_newline_terminated_text():
    assert P.cmd_ping(42) == b"PING 42\n"
    assert P.cmd_start(1280, 720) == b"START 1280 720\n"
    assert P.cmd_stop() == b"STOP\n"
    assert P.cmd_framerate(30) == b"FRAMERATE 30\n"


def test_keyframe_detection_from_annexb():
    # 4-byte start code + NAL header byte; type is low 5 bits.
    sps = b"\x00\x00\x00\x01\x67rest"   # 0x67 -> type 7 (SPS)
    idr = b"\x00\x00\x00\x01\x65rest"   # 0x65 -> type 5 (IDR)
    non = b"\x00\x00\x00\x01\x61rest"   # 0x61 -> type 1 (non-IDR slice)
    assert P.is_keyframe(sps + idr) is True
    assert P.is_keyframe(non) is False
    assert 7 in P.annexb_nal_types(sps)
