"""
Offline processing pipeline: a recorded session -> a 3D skeleton per instant.

Ties the pieces together:

    decode each phone's H.264 (with its server-timeline timestamps)
        -> 2D pose per frame (MediaPipe)
        -> synchronize frames across phones by timestamp
        -> triangulate each synced set into a 3D skeleton (calibrated rig)

The decoder and pose estimator are injected (defaults use PyAV + MediaPipe), so
the orchestration is testable with fakes and no heavy deps. Cameras without a
calibration entry, or frames where a phone found no person, simply don't vote.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .calibration import RigCalibration
from .pose import PoseEstimator
from .reconstruct import Skeleton3D, View, reconstruct
from .sync import synchronized_frames

# A frame reader yields (server_ms, BGR image) pairs for one phone.
FrameReader = Callable[[str, dict], list[tuple[float, np.ndarray]]]


@dataclass
class ProcessedFrame:
    t_ms: float
    skeleton: Skeleton3D
    cam_ids: list[str]         # cameras that contributed a 2D pose


def decode_phone(session_dir: str, phone: dict) -> list[tuple[float, np.ndarray]]:
    """Default reader: decode a phone's .h264 and pair frames with index times.

    The index rows are in append (decode) order; the phone streams carry no
    B-frames, so decoded frames come out in the same order and pair 1:1. Each
    frame is timestamped on the server timeline (``server_ms``, falling back to
    ``recv_ms``).
    """
    from .decode import H264Decoder        # imported lazily (needs PyAV)

    with open(os.path.join(session_dir, phone["index"]), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(session_dir, phone["h264"]), "rb") as f:
        data = f.read()

    decoder = H264Decoder()
    images: list[np.ndarray] = []
    for r in rows:
        images.extend(decoder.decode(data[r["offset"]:r["offset"] + r["size"]]))

    times = [r["server_ms"] if r.get("server_ms") is not None else r["recv_ms"]
             for r in rows]
    # Decoder warm-up can swallow the first frames; align to the tail.
    times = times[len(times) - len(images):] if len(images) <= len(times) else times
    return list(zip(times, images))


def process_session(session_dir: str, rig: RigCalibration, *,
                    pose_estimator: PoseEstimator | None = None,
                    frame_reader: FrameReader = decode_phone,
                    cam_id_for: Callable[[dict], str] | None = None,
                    tol_ms: float = 20.0, vis_thresh: float = 0.5,
                    min_views: int = 2) -> list[ProcessedFrame]:
    """Process a recorded session into a per-instant 3D skeleton track."""
    with open(os.path.join(session_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if pose_estimator is None:
        pose_estimator = PoseEstimator()
    cam_id_for = cam_id_for or (lambda p: p["phone_id"])

    # Decode + pose every phone that has a calibration, keyed by rig cam_id.
    times: dict[str, list[float]] = {}
    poses: dict[str, list] = {}
    proj: dict[str, np.ndarray] = {}
    for phone in manifest.get("phones", []):
        cam_id = cam_id_for(phone)
        if cam_id not in rig.cameras:
            continue
        frames = frame_reader(session_dir, phone)
        times[cam_id] = [t for t, _ in frames]
        poses[cam_id] = [pose_estimator.estimate(img) for _, img in frames]
        proj[cam_id] = rig.cameras[cam_id].projection_matrix()

    out: list[ProcessedFrame] = []
    for fs in synchronized_frames(times, tol_ms=tol_ms, min_cams=min_views):
        views = []
        for cam_id, idx in fs.frames.items():
            res = poses[cam_id][idx]
            if res is None:
                continue
            views.append(View(cam_id, proj[cam_id], res.points, res.visibility))
        if len(views) < min_views:
            continue
        skel = reconstruct(views, vis_thresh=vis_thresh, min_views=min_views)
        out.append(ProcessedFrame(t_ms=fs.t_ms, skeleton=skel,
                                  cam_ids=[v.cam_id for v in views]))
    return out
