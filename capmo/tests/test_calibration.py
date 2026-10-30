"""
End-to-end ChArUco calibration test on synthetically *rendered* multi-camera
images (capmo.synth renders the board as grey quads via a pinhole camera).

Proves the whole chain works on real detections, not hand-fed points:
  * intrinsics are recovered from many single-camera views,
  * each camera's pose is recovered from a shared board view, and
  * triangulating the detected corners from two solved cameras reconstructs the
    board's known 3D geometry.

Requires OpenCV's aruco module (skipped otherwise).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

_SAMPLE = os.path.join(os.path.dirname(__file__), "data", "charuco_sample.png")

from ..calibration import (
    BoardConfig, CameraCalibration, IntrinsicCalibrator, RigCalibration,
    detect, estimate_pose,
)
from ..geometry import triangulate_point, reprojection_error
from .. import synth

W, H = 640, 480
K_TRUE = synth.pinhole_K(700.0, 700.0, 320.0, 240.0)
ZERO_DIST = np.zeros(5)


def _render(mesh, eye):
    R, t = synth.look_at(eye)
    img = synth.rasterize(mesh, K_TRUE, R, t, W, H, bg=128)
    return img


def _corner_map(det) -> dict[int, tuple[float, float]]:
    ids = np.asarray(det.ids).ravel()
    corners = np.asarray(det.corners).reshape(-1, 2)
    return {int(i): (float(c[0]), float(c[1])) for i, c in zip(ids, corners)}


@pytest.fixture(scope="module")
def board():
    # Same 8x5 geometry as the production default (the physical sheet), but with
    # a 4x4 dictionary: a 6x6 marker is only ~2px/module in this low-res render
    # and won't decode, whereas the board *size* renders and detects fine. The
    # calibration pipeline is dictionary-agnostic (it just hands the board to
    # OpenCV), so this exercises the real board dimensions.
    return BoardConfig(squares_x=8, squares_y=5, dict_name="DICT_4X4_50")


@pytest.fixture(scope="module")
def mesh(board):
    return synth.charuco_board_mesh(board, px_per_square=30)


# ---------------------------------------------------------------------------

def test_detection_finds_all_corners_frontal(board, mesh):
    det = detect(_render(mesh, (0, 0, 0.55)), board.detector())
    assert det.count == (board.squares_x - 1) * (board.squares_y - 1)


def test_intrinsics_recovered_from_rendered_views(board, mesh):
    cal = IntrinsicCalibrator(board)
    # A spread of viewpoints (azimuth/elevation/distance) -- Zhang's method needs
    # orientation variety to condition the focal length.
    eyes = []
    for az in (-0.30, 0.0, 0.30):
        for el in (-0.22, 0.0, 0.22):
            for d in (0.5, 0.62):
                eyes.append((d * np.sin(az), d * np.sin(el), d * np.cos(az) * np.cos(el)))
    for eye in eyes:
        det = detect(_render(mesh, eye), board.detector())
        cal.add(det, (W, H))
    assert cal.count >= 12

    K, dist, rms = cal.solve()
    assert rms < 1.0, f"reprojection rms too high: {rms}"
    assert abs(K[0, 0] - 700.0) / 700.0 < 0.05    # fx
    assert abs(K[1, 1] - 700.0) / 700.0 < 0.05    # fy
    assert abs(K[0, 2] - 320.0) < 12              # cx
    assert abs(K[1, 2] - 240.0) < 12              # cy


def test_extrinsics_and_triangulation_reconstruct_board(board, mesh):
    # Three cameras around the board, all solved against the same board frame.
    eyes = {"camL": (-0.22, 0.05, 0.5), "camC": (0.0, 0.0, 0.55), "camR": (0.22, 0.05, 0.5)}
    cams: dict[str, CameraCalibration] = {}
    dets = {}
    for cid, eye in eyes.items():
        det = detect(_render(mesh, eye), board.detector())
        assert det.count >= 12, f"{cid} saw too few corners ({det.count})"
        pose = estimate_pose(det, K_TRUE, ZERO_DIST, board)
        assert pose is not None
        rvec, tvec = pose
        cams[cid] = CameraCalibration(cid, (W, H), K_TRUE, ZERO_DIST, rvec, tvec)
        dets[cid] = _corner_map(det)

    # Per-camera pose must reproject its own corners tightly.
    board_corners = board.board().getChessboardCorners()   # (24,3) in board frame
    for cid, cam in cams.items():
        P = cam.projection_matrix()
        errs = [reprojection_error(P, board_corners[i], uv) for i, uv in dets[cid].items()]
        assert max(errs) < 2.0, f"{cid} reprojection up to {max(errs):.2f}px"

    # Triangulate corners seen by camL+camR and compare to the known geometry.
    common = set(dets["camL"]) & set(dets["camR"])
    assert len(common) >= 10
    P_l, P_r = cams["camL"].projection_matrix(), cams["camR"].projection_matrix()
    errors = []
    for i in common:
        X = triangulate_point([P_l, P_r], [dets["camL"][i], dets["camR"][i]])
        errors.append(np.linalg.norm(X - board_corners[i]))
    # Board squares are 0.04 m; reconstruction should be sub-millimetre.
    assert max(errors) < 2e-3, f"triangulation error up to {max(errors) * 1000:.2f} mm"


def test_rig_calibration_serialization_round_trips(board):
    cam = CameraCalibration(
        "camA", (W, H), K_TRUE, ZERO_DIST,
        rvec=np.array([0.1, -0.2, 0.05]), tvec=np.array([0.0, 0.0, 0.5]))
    rig = RigCalibration(board=board, cameras={"camA": cam})
    d = rig.to_dict()
    back = RigCalibration.from_dict(d)
    assert back.board.squares_x == board.squares_x
    got = back.cameras["camA"]
    assert np.allclose(got.K, K_TRUE)
    assert np.allclose(got.projection_matrix(), cam.projection_matrix())


def test_legacy_board_needs_legacy_pattern():
    """A legacy-layout board (like a pre-OpenCV-4.6 print) detects all markers but
    interpolates ZERO corners under the new pattern -- and full corners under the
    legacy one. This is the exact failure capmo's UI auto-detects and flips."""
    # Pin 5x8 (a size where the legacy/new mismatch cleanly yields 0 corners).
    legacy_cfg = BoardConfig(squares_x=5, squares_y=8, legacy=True)
    new_cfg = BoardConfig(squares_x=5, squares_y=8, legacy=False)
    sx, sy = legacy_cfg.squares_x, legacy_cfg.squares_y
    img = legacy_cfg.board().generateImage((sx * 120, sy * 120), marginSize=40)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    wrong = detect(bgr, new_cfg.detector())        # new pattern on a legacy board
    assert wrong.n_markers == (sx * sy) // 2       # all 20 markers still decode
    assert wrong.count == 0                        # but no corners interpolate

    right = detect(bgr, legacy_cfg.detector())     # the auto-detected fix
    assert right.n_markers == (sx * sy) // 2
    assert right.count == (sx - 1) * (sy - 1)      # full 28 corners


def test_transposed_board_dimensions_fail_to_interpolate():
    """An 8x5 board (the physical sheet) modelled as 5x8 detects all markers but
    interpolates ZERO corners -- the real bug. The matching 8x5 model recovers
    the full 28 corners. (The UI auto-detect tries this dimension swap.)"""
    real = BoardConfig(squares_x=8, squares_y=5)            # the printed board
    wrongdims = BoardConfig(squares_x=5, squares_y=8)
    img = real.board().generateImage((8 * 120, 5 * 120), marginSize=40)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    bad = detect(bgr, wrongdims.detector())
    assert bad.n_markers == 20 and bad.count == 0          # markers ok, no corners
    good = detect(bgr, real.detector())
    assert good.n_markers == 20 and good.count == (8 - 1) * (5 - 1)   # 28


@pytest.mark.skipif(not os.path.exists(_SAMPLE), reason="sample board photo absent")
def test_default_config_detects_real_board_photo():
    """A real phone photo of the physical board must detect under the production
    default (8x5, DICT_6X6_250) -- the exact case that was failing in the field."""
    img = cv2.imread(_SAMPLE)
    assert img is not None
    cfg = BoardConfig()
    det = detect(img, cfg.detector())
    assert det.n_markers == 20
    assert det.count == (cfg.squares_x - 1) * (cfg.squares_y - 1)   # 28


def test_triangulate_point_pure_dlt():
    # Two simple cameras 1 m apart looking down +z; a point in front.
    K = synth.pinhole_K(500, 500, 320, 240)
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([np.eye(3), np.array([[-1.0], [0], [0]])])   # shifted +x by 1
    X_true = np.array([0.2, -0.1, 3.0])
    def proj(P):
        p = P @ np.append(X_true, 1.0)
        return p[:2] / p[2]
    X = triangulate_point([P0, P1], [proj(P0), proj(P1)])
    assert np.linalg.norm(X - X_true) < 1e-6
