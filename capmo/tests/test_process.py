"""
End-to-end offline pipeline (process_session) with injected decode + pose.

A known skeleton is "filmed" by three calibrated cameras at slightly offset
times; the pipeline must synchronize the frames and triangulate the skeleton back
out. Fakes stand in for PyAV and MediaPipe so the orchestration is exercised with
no heavy deps.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from .. import synth
from ..calibration import CameraCalibration, RigCalibration
from ..pose import PoseResult
from ..process import process_session

ORDER = ["a", "b", "c"]
TIMES = {"a": [0.0, 33.0, 66.0], "b": [3.0, 35.0, 69.0], "c": [5.0, 38.0, 70.0]}


def _rig_and_skeleton():
    rng = np.random.default_rng(7)
    skeleton = rng.uniform([-0.4, -0.4, -0.3], [0.4, 0.4, 0.3], size=(33, 3))
    K = synth.pinhole_K(800.0, 800.0, 320.0, 240.0)
    eyes = {"a": (0, 0, 2.0), "b": (-1.1, 0.3, 1.6), "c": (1.1, 0.3, 1.6)}
    cams = {}
    for cid, eye in eyes.items():
        R, t = synth.look_at(eye)
        rvec, _ = cv2.Rodrigues(R)
        cams[cid] = CameraCalibration(cid, (640, 480), K, np.zeros(5), rvec.ravel(), t)
    return RigCalibration(cameras=cams), skeleton


def _tagged_image(cam_index: int) -> np.ndarray:
    img = np.zeros((480, 640, 3), np.uint8)
    img[0, 0, 0] = cam_index            # so the fake pose knows the camera
    return img


def _fake_reader(session_dir, phone):
    cid = phone["phone_id"]
    ci = ORDER.index(cid)
    return [(t, _tagged_image(ci)) for t in TIMES[cid]]


class _FakePose:
    """Projects the known skeleton with the tagged camera's matrix."""
    def __init__(self, rig, skeleton, drop=()):
        self._proj = {c: rig.cameras[c].projection_matrix() for c in rig.cameras}
        self._skel = skeleton
        self._drop = set(drop)        # (cam_id, frame_t) to simulate no detection
        self._seen: dict[str, int] = {}

    def estimate(self, img):
        cid = ORDER[int(img[0, 0, 0])]
        n = self._seen.get(cid, 0)
        self._seen[cid] = n + 1
        if (cid, TIMES[cid][n]) in self._drop:
            return None
        P = self._proj[cid]
        pts = np.array([(P @ np.append(X, 1.0))[:2] / (P @ np.append(X, 1.0))[2]
                        for X in self._skel])
        return PoseResult(points=pts, visibility=np.ones(self._skel.shape[0]))


def _write_manifest(tmp_path, extra_phones=()):
    phones = [{"phone_id": c, "slug": c, "h264": f"{c}.h264", "index": f"{c}.jsonl"}
              for c in ORDER]
    phones.extend(extra_phones)
    (tmp_path / "manifest.json").write_text(json.dumps({"phones": phones}))
    return str(tmp_path)


def test_pipeline_reconstructs_skeleton(tmp_path):
    rig, skeleton = _rig_and_skeleton()
    session = _write_manifest(tmp_path)
    frames = process_session(session, rig, pose_estimator=_FakePose(rig, skeleton),
                             frame_reader=_fake_reader, cam_id_for=lambda p: p["phone_id"])
    assert len(frames) == 3
    for pf in frames:
        assert sorted(pf.cam_ids) == ["a", "b", "c"]
        assert pf.skeleton.valid.all()
        err = np.max(np.linalg.norm(pf.skeleton.points - skeleton, axis=1))
        assert err < 1e-6, f"reconstruction off by {err}"


def test_uncalibrated_phone_is_ignored(tmp_path):
    rig, skeleton = _rig_and_skeleton()
    # 'd' appears in the manifest but has no calibration -> skipped entirely.
    session = _write_manifest(tmp_path, extra_phones=[
        {"phone_id": "d", "slug": "d", "h264": "d.h264", "index": "d.jsonl"}])
    frames = process_session(session, rig, pose_estimator=_FakePose(rig, skeleton),
                             frame_reader=_fake_reader, cam_id_for=lambda p: p["phone_id"])
    assert len(frames) == 3
    assert all("d" not in pf.cam_ids for pf in frames)


def test_missing_detection_drops_only_that_view(tmp_path):
    rig, skeleton = _rig_and_skeleton()
    session = _write_manifest(tmp_path)
    # camera 'c' finds no person in its first frame (t=70 set still has a+b).
    pose = _FakePose(rig, skeleton, drop={("c", 70.0)})
    frames = process_session(session, rig, pose_estimator=pose,
                             frame_reader=_fake_reader, cam_id_for=lambda p: p["phone_id"])
    assert len(frames) == 3
    last = [pf for pf in frames if pf.t_ms == 66.0][0]
    assert sorted(last.cam_ids) == ["a", "b"]          # c dropped
    assert last.skeleton.valid.all()                   # 2 views still triangulate
