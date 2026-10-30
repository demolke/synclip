"""Recording-session tests: raw .h264 + index + manifest."""

from __future__ import annotations

import json
import os

from ..session import RecordingSession, _safe_id


def test_writes_raw_bytes_and_index(tmp_path):
    sess = RecordingSession(str(tmp_path))
    sess.write_frame("Pixel 5", b"\x00\x01keyframe", recv_ms=1000.0,
                     is_keyframe=True, meta_ms=1700, server_ms=1600.0)
    sess.write_frame("Pixel 5", b"delta", recv_ms=1033.0, is_keyframe=False)
    sess.close(clock_offsets={"Pixel 5": 100.0})

    slug = _safe_id("Pixel 5")
    h264 = os.path.join(sess.dir, f"{slug}.h264")
    # Stored verbatim, concatenated, no re-encode.
    assert open(h264, "rb").read() == b"\x00\x01keyframe" + b"delta"

    rows = [json.loads(l) for l in open(os.path.join(sess.dir, f"{slug}.index.jsonl"))]
    assert len(rows) == 2
    assert rows[0]["offset"] == 0 and rows[0]["size"] == len(b"\x00\x01keyframe")
    assert rows[0]["key"] is True and rows[0]["meta_ms"] == 1700
    assert rows[1]["offset"] == len(b"\x00\x01keyframe") and rows[1]["key"] is False


def test_manifest_lists_phones_and_offsets(tmp_path):
    sess = RecordingSession(str(tmp_path))
    sess.write_frame("camA", b"x", recv_ms=1.0, is_keyframe=True, meta_ms=1)
    sess.write_frame("camB", b"yy", recv_ms=2.0, is_keyframe=True, meta_ms=2)
    sess.close(clock_offsets={"camA": 10.0, "camB": -5.0})

    man = json.load(open(os.path.join(sess.dir, "manifest.json")))
    by_id = {p["phone_id"]: p for p in man["phones"]}
    assert by_id["camA"]["frames"] == 1 and by_id["camA"]["clock_offset_ms"] == 10.0
    assert by_id["camB"]["bytes"] == 2 and by_id["camB"]["clock_offset_ms"] == -5.0


def test_slug_collisions_get_unique_files(tmp_path):
    sess = RecordingSession(str(tmp_path))
    sess.write_frame("a b", b"1", recv_ms=1.0, is_keyframe=True)   # slug "a_b"
    sess.write_frame("a/b", b"2", recv_ms=1.0, is_keyframe=True)   # also slugs to "a_b"
    slugs = {sess._phones[p].slug for p in sess.phone_ids()}
    assert len(slugs) == 2
    sess.close()
