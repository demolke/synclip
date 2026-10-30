"""
Turn a capmo recording session into ordinary, already-synchronised video files.

    python -m capmo.tools.session_to_take SESSION_DIR --out VIDEOS_DIR [--cam-ids a,b,c]

For every camera the raw ``<slug>.h264`` is **remuxed without re-encoding**
(PyAV stream copy) into ``VIDEOS_DIR/<cam_id>.mp4`` (or ``.mkv``) whose packet
timestamps are the per-frame **host-clock capture times** (see
:mod:`capmo.core.timeline`) relative to one common session origin ``t0`` (the
earliest decodable frame of any camera). All files therefore share one
timeline: the same PTS means the same instant, to within the clock-sync
uncertainty reported in ``take.json``.

Handled: codec config (SPS/PPS) missing at the start of the recording (taken
from a config-only message or the manifest's ``codec_config_b64``, prepended to
the first IDR), frames before the first IDR (dropped: undecodable), frames
dropped by the phone (gaps stay gaps -- variable frame rate), in-band config
messages (folded into the next frame). Assumes no B-frames (phones stream
Baseline H.264).

Writes ``VIDEOS_DIR/take.json``: session info, ``t0_ms``, per-camera clock
offsets / timeline method / drop counts, a sync-quality estimate and, when
``ffprobe`` is available, the max |PTS - expected| per file (``--no-verify``
skips that check).

``--cam-ids``: ``a,b,c`` renames the cameras in manifest order; ``slug=a,...``
maps (and selects) by slug / phone id. ``--cam-ids`` given to
``extract_keypoints`` later must then be omitted (the file stems are the ids).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction

import numpy as np

from ..core.session_io import Session, SessionCamera, load_session

TIME_BASE = Fraction(1, 90000)


def _template_stream(first_au: bytes):
    """An input stream (codec parameters + extradata) probed from the first AU."""
    import av
    inp = av.open(io.BytesIO(first_au), format="h264")
    if not inp.streams.video:
        raise ValueError("cannot parse the H.264 stream (no SPS/PPS?)")
    return inp, inp.streams.video[0]


def remux_camera(sess: Session, cam: SessionCamera, path: str) -> dict:
    """Stream-copy *cam* into *path* with PTS = host capture time - session t0."""
    import av
    t0 = sess.t0_ms
    aus = cam.access_units()
    try:
        k, i, first = next(aus)
    except StopIteration:
        raise ValueError(f"{cam.cam_id}: no decodable frame (no IDR with codec config)")
    inp, tmpl = _template_stream(first)
    expected = []
    try:
        with av.open(path, "w") as out:
            ost = out.add_stream_from_template(tmpl)
            ost.time_base = TIME_BASE
            out.metadata["comment"] = (f"capmo session {sess.name} cam {cam.cam_id}; "
                                       f"pts = host capture time - t0_ms {t0:.3f}")
            last = -1

            def put(i, data):
                nonlocal last
                pts = int(round((cam.t_ms[i] - t0) / 1000.0 / float(TIME_BASE)))
                pts = max(pts, last + 1)              # strictly increasing
                last = pts
                pkt = av.Packet(data)
                pkt.stream = ost
                pkt.time_base = TIME_BASE
                pkt.pts = pkt.dts = pts
                pkt.is_keyframe = bool(cam.key[i])
                out.mux(pkt)
                expected.append(pts * float(TIME_BASE))

            put(i, first)
            for k, i, data in aus:
                put(i, data)
    finally:
        inp.close()
    exp = np.asarray(expected)
    return {"file": os.path.basename(path), "frames": int(len(exp)),
            "first_pts_s": float(exp[0]), "last_pts_s": float(exp[-1]),
            "expected_pts_s": exp}


def ffprobe_pts(path: str) -> np.ndarray | None:
    """Packet PTS (s) of the first video stream via ffprobe (None if unavailable)."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "packet=pts_time", "-of", "json", path],
                           capture_output=True, text=True, timeout=120, check=True)
        pk = json.loads(r.stdout).get("packets", [])
        return np.array(sorted(float(p["pts_time"]) for p in pk if "pts_time" in p))
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def convert(session_dir: str, out_dir: str, cam_ids=None, fmt: str = "mp4",
            verify: bool = True, log=print) -> dict:
    sess = load_session(session_dir, cam_ids)
    if not sess.cameras:
        raise ValueError(f"{session_dir}: no camera with frames")
    os.makedirs(out_dir, exist_ok=True)
    meta = sess.to_meta()
    meta["time_base"] = str(TIME_BASE)
    meta["videos"] = []
    log(f"session {sess.name}: {len(sess.cameras)} camera(s), t0 = {sess.t0_ms:.1f} ms (host)")
    for cam in sess.cameras:
        path = os.path.join(out_dir, f"{cam.cam_id}.{fmt}")
        info = remux_camera(sess, cam, path)
        exp = info.pop("expected_pts_s")
        if verify:
            got = ffprobe_pts(path)
            if got is not None:
                n = min(len(got), len(exp))
                info["ffprobe_frames"] = int(len(got))
                info["ffprobe_max_err_ms"] = (float(np.abs(got[:n] - exp[:n]).max() * 1000)
                                              if n else None)
        info.update(cam_id=cam.cam_id, slug=cam.slug, phone_id=cam.phone_id,
                    leading_undecodable_frames=int(cam.first_decodable),
                    time_method=cam.times.method,
                    clock_offset_ms=cam.offset.a if cam.offset else None,
                    dropped_frames_inferred=int(cam.times.dropped_slots + cam.gap_frames()),
                    fps_measured=cam.fps_measured, width=cam.width, height=cam.height)
        meta["videos"].append(info)
        chk = (f", ffprobe max |dPTS| {info['ffprobe_max_err_ms']:.3f} ms"
               if info.get("ffprobe_max_err_ms") is not None else "")
        log(f"  {cam.cam_id}: {info['frames']} frames, t {info['first_pts_s']:.3f}.."
            f"{info['last_pts_s']:.3f} s, {cam.times.method}, offset "
            f"{info['clock_offset_ms'] if info['clock_offset_ms'] is not None else float('nan'):+.1f} ms"
            f"{chk} -> {path}")
    sync = meta["sync"]
    if sync.get("pair_uncertainty_ms") is not None:
        log(f"  sync uncertainty (worst pair): {sync['pair_uncertainty_ms']:.1f} ms "
            f"({sync['pair_uncertainty_frames']:.2f} frames)")
    else:
        log("  sync: at least one camera has no clock-based timeline; rely on "
            "reconstruct_take's epipolar sync")
    with open(os.path.join(out_dir, "take.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m capmo.tools.session_to_take",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("session_dir")
    ap.add_argument("--out", required=True, help="output folder for the videos + take.json")
    ap.add_argument("--cam-ids", default=None,
                    help="a,b,c (rename in manifest order) or slug=a,... (map/select)")
    ap.add_argument("--format", choices=["mp4", "mkv"], default="mp4")
    ap.add_argument("--no-verify", action="store_true", help="skip the ffprobe PTS check")
    a = ap.parse_args(argv)
    try:
        convert(a.session_dir, a.out, a.cam_ids, a.format, verify=not a.no_verify)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
