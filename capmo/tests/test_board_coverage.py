"""
The live board-coverage tracker (:mod:`capmo.core.board_coverage`) and the
shared board helpers (:mod:`capmo.core.board`). No Qt: detections are synthetic
projections of the real ChArUco corner geometry, frames are warped renders of
the real board image.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):  # pragma: no cover
    pytest.skip("cv2.aruco missing", allow_module_level=True)

from ..core import board as B  # noqa: E402
from ..core.board_coverage import GOOD, IDLE, SCANNING, LiveBoardTracker, Thresholds  # noqa: E402

CFG = B.BoardConfig(7, 5, 0.04, 0.03)
W, H = 640, 480
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])


def det_at(cx, cy, tilt=0.0, axis=(0, 1, 0), dist=0.9, ids=None):
    """A Detection of the whole board centred at pixel (cx, cy), tilted by
    *tilt* degrees about *axis* (board frame)."""
    c3 = CFG.corners3d()
    ctr = np.array([CFG.squares_x * CFG.square_len / 2, CFG.squares_y * CFG.square_len / 2, 0])
    R, _ = cv2.Rodrigues(np.asarray(axis, float) * np.radians(tilt))
    X = (c3 - ctr) @ R.T
    t = np.array([(cx - 320) / 600 * dist, (cy - 240) / 600 * dist, dist])
    p = (X + t) @ K.T
    p = p[:, :2] / p[:, 2:]
    sel = np.arange(len(c3)) if ids is None else np.asarray(ids)
    return B.Detection(p[sel].astype(np.float32).reshape(-1, 1, 2),
                       sel.astype(np.int32).reshape(-1, 1), n_markers=17)


GRID9 = list(itertools.product([110, 320, 530], [90, 240, 390]))


def feed_good(tr, cam, t0=0.0, dt=1.0, n=18, partner=None):
    """Enough novel, tilted views spread over the image to be 'good'."""
    t = t0
    for i in range(n):
        cx, cy = GRID9[i % 9]
        cx += 25 * (i // 9)
        tilt = (-28, 28)[i % 2]
        axis = (0, 1, 0) if (i // 2) % 2 else (1, 0, 0)
        t += dt
        tr.ingest(cam, t, det_at(cx, cy, tilt, axis), (W, H))
        if partner:
            tr.ingest(partner, t + 0.01, det_at(cx, cy, tilt, axis), (W, H))
    return t


# ------------------------------------------------------------- core.board
def test_board_coverage_is_the_calibrate_board_metric():
    from ..tools import calibrate_board as cb
    views = [cb.View(0, 0.0, np.arange(4), np.array([[10.0, 10], [630, 10], [10, 470], [320, 240]]),
                     np.zeros((4, 3)))]
    assert cb.coverage(views, (W, H)) == B.coverage([v.img for v in views], (W, H))
    assert cb.coverage(views, (W, H)) == pytest.approx(4 / 48)
    assert cb.WARN_COVERAGE == B.GOOD_COVERAGE and cb.COVERAGE_GRID == B.COVERAGE_GRID


def _render(blur=0.0, motion=0, legacy=False, squares=(7, 5)):
    """The real board image, perspective-warped into a 640x480 frame."""
    cfg = B.BoardConfig(squares[0], squares[1], 0.04, 0.03, legacy=legacy)
    tex = cfg.board().generateImage((squares[0] * 60 + 40, squares[1] * 60 + 40), marginSize=20)
    th, tw = tex.shape
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    dst = np.float32([[100, 80], [500, 90], [490, 380], [110, 370]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    img = np.full((H, W), 110, np.uint8)
    warped = cv2.warpPerspective(tex, Hm, (W, H))
    mask = cv2.warpPerspective(np.full_like(tex, 255), Hm, (W, H))
    img[mask > 0] = warped[mask > 0]
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if motion:
        k = np.zeros((motion, motion))
        k[motion // 2, :] = 1.0 / motion
        img = cv2.filter2D(img, -1, k)
    img = np.clip(img + np.random.default_rng(0).normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_sharpness_drops_with_blur_and_motion():
    sharp = _render()
    det = B.detect(sharp, CFG.detector())
    assert det.count == 24
    s0 = B.sharpness(sharp, det.points())
    s1 = B.sharpness(_render(blur=2.0), det.points())
    s2 = B.sharpness(_render(motion=7), det.points())
    assert s0 > 0.25 and s1 < 0.16 and s2 < 0.16
    assert s0 > Thresholds().min_sharpness > max(s1, s2)


def test_board_detector_resolves_layout():
    img = _render(legacy=True, squares=(8, 6))            # layouts differ for even rows
    plain = B.BoardDetector(B.BoardConfig(8, 6, 0.04, 0.03))
    assert B.detect(img, plain.detector).count == 0
    det = plain(img)
    assert det.count > 0 and plain.cfg.legacy and plain.switched == 1
    # an 8x5 print configured as 5x8 needs the swap (UI only)
    img2 = _render(squares=(5, 7))
    no_swap = B.BoardDetector(CFG)
    assert no_swap(img2).count == 0
    swap = B.BoardDetector(CFG, try_swap=True, lock=True)
    assert swap(img2).count > 0
    assert (swap.cfg.squares_x, swap.cfg.squares_y) == (5, 7) and swap.locked


# ------------------------------------------------------------- the tracker
def test_idle_until_the_board_is_seen():
    tr = LiveBoardTracker(CFG)
    tr.set_cameras(["camA", "camB"])
    st = tr.statuses()
    assert [s.state for s in st.values()] == [IDLE, IDLE]
    assert st["camA"].hints == ["show the board to this camera"]
    assert st["camA"].views == 0 and st["camA"].coverage == 0


def test_views_need_corners_area_novelty_and_time():
    tr = LiveBoardTracker(CFG)
    assert not tr.ingest("camA", 0.0, det_at(320, 240, ids=range(5)), (W, H)).usable   # < 8
    assert not B.non_degenerate(CFG.corners3d()[:6], CFG.square_len)                  # one row
    assert B.non_degenerate(CFG.corners3d()[:12], CFG.square_len)
    res = tr.ingest("camA", 1.0, det_at(320, 240), (W, H))
    assert res.accepted and tr.status("camA").views == 1
    assert not tr.ingest("camA", 1.2, det_at(500, 240), (W, H)).accepted   # too soon
    assert not tr.ingest("camA", 2.0, det_at(325, 240), (W, H)).accepted   # not novel
    assert tr.ingest("camA", 3.0, det_at(500, 240), (W, H)).accepted       # moved
    st = tr.status("camA")
    assert st.views == 2 and st.state == SCANNING and st.detections >= 4


def test_coverage_hint_points_at_the_empty_region():
    tr = LiveBoardTracker(CFG)
    t = 0.0
    for cx, cy in [(450, 330), (530, 390), (320, 390), (530, 240), (420, 250)]:
        t += 1
        tr.ingest("camA", t, det_at(cx, cy, dist=1.4), (W, H))
    st = tr.status("camA")
    assert st.coverage < 0.5
    assert st.hints[0] == "cover the top-left"
    expect = B.coverage([det_at(cx, cy, dist=1.4).points() for cx, cy in
                         [(450, 330), (530, 390), (320, 390), (530, 240), (420, 250)]], (W, H))
    assert st.coverage == pytest.approx(expect)


def test_single_camera_goes_good_with_coverage_views_and_tilt():
    tr = LiveBoardTracker(CFG)
    t = 0.0
    for i in range(18):                                  # frontal only: no tilt
        cx, cy = GRID9[i % 9]
        t += 1
        tr.ingest("camA", t, det_at(cx + 25 * (i // 9), cy), (W, H))
    st = tr.status("camA")
    assert st.coverage >= 0.5 and st.views >= 15 and st.state == SCANNING
    assert any(h.startswith("tilt more") for h in st.hints)
    tr2 = LiveBoardTracker(CFG)
    feed_good(tr2, "camA")
    st2 = tr2.status("camA")
    assert st2.tilt_max_deg >= 20 and st2.tilt_directions >= 2
    assert st2.state == GOOD and st2.hints == []
    assert "good coverage" in st2.readout()


def test_pairs_link_cameras_and_name_the_partner():
    tr = LiveBoardTracker(CFG)
    tr.set_cameras(["camA", "camB", "camC"])
    feed_good(tr, "camA", partner="camB")                 # A+B together
    t = feed_good(tr, "camC", t0=100.0)                   # C always alone
    st = tr.statuses()
    assert st["camA"].pairs["camB"] >= 3 and "camC" not in st["camA"].pairs
    assert st["camA"].state == GOOD and st["camB"].state == GOOD
    assert st["camC"].state == SCANNING and not st["camC"].linked
    assert "show to camC together with camA" in st["camC"].hints
    # C now co-sees the board with B a few times -> linked -> all good
    for k in range(3):
        t += 1.0
        tr.ingest("camB", t, det_at(200 + 80 * k, 240), (W, H))
        tr.ingest("camC", t + 0.02, det_at(400 - 80 * k, 240), (W, H))
    st = tr.statuses()
    assert st["camC"].pairs["camB"] == 3 and st["camC"].linked
    assert all(s.state == GOOD for s in st.values())
    summ = tr.summary()
    assert summ["all_good"] and summ["pairs"]["camB-camC"] == 3
    assert summ["board"]["squares_x"] == 7 and summ["thresholds"]["min_shared"] == 3


def test_pairs_need_near_simultaneous_frames_and_are_rate_limited():
    tr = LiveBoardTracker(CFG)
    tr.ingest("camA", 10.0, det_at(320, 240), (W, H))
    tr.ingest("camB", 10.2, det_at(320, 240), (W, H))      # 200 ms apart: not together
    assert tr.pair_counts() == {}
    tr.ingest("camB", 10.03, det_at(300, 240), (W, H))     # within 60 ms (out of order ok)
    assert tr.pair_counts() == {"camA-camB": 1}
    tr.ingest("camA", 10.25, det_at(320, 240), (W, H))     # pairs camB@10.2, but < 0.5 s
    assert tr.pair_counts() == {"camA-camB": 1}
    tr.ingest("camA", 11.0, det_at(320, 240), (W, H))
    tr.ingest("camB", 11.01, det_at(320, 240), (W, H))
    assert tr.pair_counts() == {"camA-camB": 2}


def test_ingest_frame_detects_and_rejects_blur():
    tr = LiveBoardTracker(CFG)
    det, res = tr.ingest_frame("camA", 0.0, _render(blur=2.0))
    assert det.count >= 8 and res.blurry and not res.usable
    st = tr.status("camA")
    assert st.state == IDLE and st.blurry == 1 and st.hints[0].startswith("hold the board still")
    det, res = tr.ingest_frame("camA", 1.0, _render())
    assert det.count == 24 and res.accepted and res.sharpness > 0.25
    assert tr.status("camA").state == SCANNING


def test_ingest_frame_adopts_the_matched_layout():
    tr = LiveBoardTracker(CFG)
    det, res = tr.ingest_frame("camA", 0.0, _render(squares=(5, 7)))
    assert det.count > 0 and res.usable
    assert (tr.board_cfg.squares_x, tr.board_cfg.squares_y) == (5, 7)
    tr.reset()                                           # keeps the matched board
    assert tr.board_cfg.squares_x == 5 and tr.statuses() == {}
