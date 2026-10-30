"""
Live board-coverage / sharpness indicator: is the ChArUco board segment that is
being recorded good enough for ``calibrate_board``?

:class:`LiveBoardTracker` is UI-agnostic (no Qt, no threads): feed it board
detections (or raw frames, it then runs the shared detector of
:mod:`capmo.core.board`) per camera with a timestamp, ask it for a per-camera
:class:`CameraStatus`. It solves nothing -- the calibration itself happens
offline (``calibrate_board``) on the recording; this only tells the operator,
while recording, what the offline solve will be missing.

What it tracks per camera, mirroring what ``calibrate_board`` needs and reports
(``capmo/tools/README.md``, "Reading the report"):

* **coverage** -- fraction of the 8x6 image grid that has seen a board corner
  (``coverage`` in the report, same function, :func:`capmo.core.board.coverage`).
  Good: >= :data:`~capmo.core.board.GOOD_COVERAGE` (0.5).
* **views** -- distinct board views: >= ``min_corners`` (8) corners spanning an
  area, sharp, and *novel* (the board moved by > 6 % of the image width or a
  quarter of the visible corner ids changed since the last kept view, at most
  one every 0.5 s). Good: >= :data:`~capmo.core.board.GOOD_VIEWS` (15 --
  ``n_views_intrinsics``).
* **tilt** -- board tilt vs the image plane per kept view (homography with a
  guessed focal, :func:`capmo.core.board.rough_tilt`). Good: max >= 20 deg
  (``tilt_deg.max``); the hint also asks for a second tilt direction.
* **sharpness** -- edge steepness of the board region
  (:func:`capmo.core.board.sharpness`); a detection below ``min_sharpness``
  (0.17: edges smeared over ~2 px, motion blur or focus) is counted as blurry
  and not used.
* **pairs** -- simultaneous views with every other camera: two cameras' usable
  detections whose timestamps differ by <= ``sync_tol_s`` (default 0.06 s:
  the live timestamps are frame arrival times with network jitter; the offline
  solve pairs to half a frame after offsets, which is why the board should be
  held still) count once per ``pair_interval_s``. A pair with >=
  :data:`~capmo.core.board.MIN_SHARED_VIEWS` (3, ``--min-shared``) such views
  is an edge of the camera graph; with two or more cameras each one must be
  connected through edges to the reference camera (the first one), which
  itself needs at least one edge -- ``calibrate_board`` chains the cameras
  along those edges.

Status: ``idle`` (no usable board view yet), ``scanning``, ``good`` (all of the
above met). ``hints`` are short operator instructions ("cover the top-left",
"tilt more", "show to camC together with camA", "hold still").
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from . import board as _board
from .sync.frames import nearest

IDLE, SCANNING, GOOD = "idle", "scanning", "good"

_REGIONS = [("top-left", 0, 0), ("top", 1, 0), ("top-right", 2, 0),
            ("left", 0, 1), ("centre", 1, 1), ("right", 2, 1),
            ("bottom-left", 0, 2), ("bottom", 1, 2), ("bottom-right", 2, 2)]


@dataclass
class Thresholds:
    grid: tuple = _board.COVERAGE_GRID
    good_coverage: float = _board.GOOD_COVERAGE
    good_views: int = _board.GOOD_VIEWS
    good_max_tilt_deg: float = _board.GOOD_MAX_TILT_DEG
    min_corners: int = _board.MIN_CORNERS
    min_shared: int = _board.MIN_SHARED_VIEWS
    min_sharpness: float = 0.17
    sync_tol_s: float = 0.06
    pair_interval_s: float = 0.5
    novelty_shift: float = 0.06          # fraction of the image width
    novelty_jaccard: float = 0.75        # kept if fewer ids in common than this
    min_accept_interval_s: float = 0.5
    tilt_dir_min_deg: float = 12.0       # a view counts for a tilt direction above this
    history_s: float = 10.0              # how long detections are kept for pairing


@dataclass
class CameraStatus:
    cam_id: str
    state: str = IDLE
    coverage: float = 0.0
    views: int = 0
    tilt_max_deg: float = 0.0
    tilt_directions: int = 0             # of 4 (right/down/left/up), tilt >= tilt_dir_min_deg
    sharpness: float | None = None       # last usable-or-blurry detection
    blurry: int = 0                      # detections rejected as blurry
    detections: int = 0                  # usable detections (incl. non-novel)
    pairs: dict = field(default_factory=dict)    # other cam -> simultaneous views
    linked: bool = True                  # connected to the reference camera
    hints: list = field(default_factory=list)
    last_corners: int = 0
    last_markers: int = 0

    @property
    def coverage_pct(self) -> int:
        return int(round(100 * self.coverage))

    def readout(self, good_views: int = _board.GOOD_VIEWS) -> str:
        """One-line tile text."""
        if self.state == IDLE:
            head = "board check: no board yet"
        elif self.state == GOOD:
            head = f"✓ good coverage ({self.coverage_pct}%, {self.views} views)"
        else:
            head = f"scanning… {self.coverage_pct}% · {self.views}/{good_views} views"
        if self.pairs:
            head += " · " + " ".join(f"{k}:{v}" for k, v in sorted(self.pairs.items()))
        return head + (f" — {self.hints[0]}" if self.hints and self.state != GOOD else "")

    def to_dict(self) -> dict:
        return {"state": self.state, "coverage": round(self.coverage, 3),
                "views": self.views, "tilt_max_deg": round(self.tilt_max_deg, 1),
                "tilt_directions": self.tilt_directions,
                "sharpness": None if self.sharpness is None else round(self.sharpness, 3),
                "blurry": self.blurry, "detections": self.detections,
                "pairs": dict(self.pairs), "linked": self.linked, "hints": list(self.hints)}


@dataclass
class IngestResult:
    usable: bool = False                 # counted for coverage / pairs
    accepted: bool = False               # kept as a new distinct view
    blurry: bool = False
    sharpness: float | None = None
    reason: str = ""


class _Cam:
    def __init__(self, cam_id: str, grid) -> None:
        self.cam_id = cam_id
        gx, gy = grid
        self.occ = np.zeros((gy, gx), bool)
        self.size = None
        self.views = 0
        self.tilts: list[tuple[float, float]] = []
        self.kept_centroid = None
        self.kept_ids: set = set()
        self.kept_t = -np.inf
        self.det_t: list[float] = []      # usable detection times (sorted)
        self.detections = 0
        self.blurry = 0
        self.last_blurry = False
        self.sharpness = None
        self.last_corners = 0
        self.last_markers = 0


class LiveBoardTracker:
    """Per-camera board coverage / view / tilt / sharpness / co-view tracker.

    *board_cfg*: the :class:`~capmo.core.board.BoardConfig` (gives the corner
    geometry for tilt / degeneracy and the detector for :meth:`ingest_frame`);
    *corners3d* overrides its geometry (tests). Thread-unsafe by design: call it
    from one thread (the UI's GUI thread).
    """

    def __init__(self, board_cfg: _board.BoardConfig | None = None, *,
                 thresholds: Thresholds | None = None, corners3d: np.ndarray | None = None,
                 try_swap: bool = True) -> None:
        self.th = thresholds or Thresholds()
        self._cfg = board_cfg or _board.BoardConfig()
        self._corners3d = None if corners3d is None else np.asarray(corners3d, np.float64)
        self._detector = None
        self._try_swap = try_swap
        self._cams: dict[str, _Cam] = {}
        self._order: list[str] = []
        self._pairs: dict[tuple[str, str], int] = {}
        self._pair_t: dict[tuple[str, str], float] = {}

    # ---------------------------------------------------------------- config
    @property
    def board_cfg(self) -> _board.BoardConfig:
        """The board actually matched (layout auto-resolved by :meth:`ingest_frame`)."""
        return self._detector.cfg if self._detector is not None else self._cfg

    @property
    def detector(self) -> _board.BoardDetector:
        if self._detector is None:
            self._detector = _board.BoardDetector(self._cfg, try_swap=self._try_swap, lock=True)
        return self._detector

    def _geometry(self) -> np.ndarray | None:
        if self._corners3d is None:
            try:
                self._corners3d = self.board_cfg.corners3d()
            except Exception:                 # no cv2.aruco: no tilt / degeneracy test
                return None
        return self._corners3d

    def set_cameras(self, cam_ids) -> None:
        """Declare the cameras that should see the board (keeps their order; the
        first is the reference). Unknown cameras show as idle."""
        for c in cam_ids:
            self._cam(c)
        self._order = list(cam_ids) + [c for c in self._order if c not in cam_ids]

    def reset(self) -> None:
        self._cams.clear()
        self._order.clear()
        self._pairs.clear()
        self._pair_t.clear()
        if self._detector is not None:
            self._cfg = self._detector.cfg
            self._detector = None

    def _cam(self, cam_id: str) -> _Cam:
        c = self._cams.get(cam_id)
        if c is None:
            c = self._cams[cam_id] = _Cam(cam_id, self.th.grid)
            if cam_id not in self._order:
                self._order.append(cam_id)
        return c

    # ---------------------------------------------------------------- ingest
    def ingest_frame(self, cam_id: str, t: float, image: np.ndarray):
        """Detect the board in *image* (shared detector, layout auto-resolve)
        and :meth:`ingest` it. Returns ``(Detection, IngestResult)``."""
        det = self.detector(image)
        h, w = image.shape[:2]
        return det, self.ingest(cam_id, t, det, (w, h), image=image)

    def ingest(self, cam_id: str, t: float, det, image_size, image=None) -> IngestResult:
        """One detection of camera *cam_id* at time *t* (seconds, any clock
        shared by all cameras). *image* (optional) enables the blur check."""
        th = self.th
        c = self._cam(cam_id)
        c.size = tuple(image_size)
        c.last_corners = det.count
        c.last_markers = int(getattr(det, "n_markers", 0))
        if det.count < th.min_corners:
            return IngestResult(reason="too few corners")
        pts = det.points()
        ids = det.id_array()
        geo = self._geometry()
        obj = None
        if geo is not None:
            ok = (ids >= 0) & (ids < len(geo))
            pts, ids = pts[ok], ids[ok]
            if len(ids) < th.min_corners:
                return IngestResult(reason="too few corners")
            obj = geo[ids]
            if not _board.non_degenerate(obj, self.board_cfg.square_len):
                return IngestResult(reason="corners in a line")
        sharp = None
        if image is not None:
            try:
                sharp = _board.sharpness(image, pts)
            except Exception:
                sharp = None
        c.sharpness = sharp
        if sharp is not None and sharp < th.min_sharpness:
            c.blurry += 1
            c.last_blurry = True
            return IngestResult(blurry=True, sharpness=sharp, reason="blurry")
        c.last_blurry = False
        # usable: coverage + simultaneous views
        c.detections += 1
        c.occ |= _board.coverage_cells(pts, c.size, th.grid)
        self._pair_up(cam_id, t)
        bisect.insort(c.det_t, float(t))
        while c.det_t and c.det_t[0] < t - th.history_s:
            c.det_t.pop(0)
        res = IngestResult(usable=True, sharpness=sharp)
        # distinct view?
        if t - c.kept_t < th.min_accept_interval_s or not self._novel(c, pts, ids):
            res.reason = "not novel"
            return res
        c.views += 1
        c.kept_t = float(t)
        c.kept_centroid = pts.mean(0)
        c.kept_ids = set(ids.tolist())
        if obj is not None:
            try:
                c.tilts.append(_board.rough_tilt(obj, pts, c.size))
            except Exception:
                pass
        res.accepted = True
        return res

    def _novel(self, c: _Cam, pts, ids) -> bool:
        if c.kept_centroid is None:
            return True
        w = c.size[0] if c.size else 0
        shift = float(np.linalg.norm(pts.mean(0) - c.kept_centroid))
        if w > 0 and shift > self.th.novelty_shift * w:
            return True
        cur = set(ids.tolist())
        union = cur | c.kept_ids
        return bool(union) and len(cur & c.kept_ids) / len(union) < self.th.novelty_jaccard

    def _pair_up(self, cam_id: str, t: float) -> None:
        for other, oc in self._cams.items():
            if other == cam_id or not oc.det_t:
                continue
            j = nearest(oc.det_t, t)
            if abs(oc.det_t[j] - t) > self.th.sync_tol_s:
                continue
            key = tuple(sorted((cam_id, other)))
            if t - self._pair_t.get(key, -np.inf) < self.th.pair_interval_s:
                continue
            self._pair_t[key] = float(t)
            self._pairs[key] = self._pairs.get(key, 0) + 1

    # ---------------------------------------------------------------- status
    def pair_counts(self) -> dict:
        """``{"camA-camB": n}`` simultaneous views per camera pair."""
        return {f"{a}-{b}": n for (a, b), n in sorted(self._pairs.items())}

    def _components(self) -> dict[str, int]:
        comp = {c: i for i, c in enumerate(self._order)}
        changed = True
        while changed:
            changed = False
            for (a, b), n in self._pairs.items():
                if n >= self.th.min_shared and comp[a] != comp[b]:
                    lo = min(comp[a], comp[b])
                    hi = max(comp[a], comp[b])
                    for k in comp:
                        if comp[k] == hi:
                            comp[k] = lo
                    changed = True
        return comp

    def status(self, cam_id: str) -> CameraStatus:
        return self.statuses()[cam_id]

    def statuses(self) -> dict[str, CameraStatus]:
        th = self.th
        comp = self._components()
        ref = self._order[0] if self._order else None
        multi = len(self._order) >= 2
        size: dict[int, int] = {}
        for k in comp.values():
            size[k] = size.get(k, 0) + 1
        out: dict[str, CameraStatus] = {}
        for cid in self._order:
            c = self._cams[cid]
            tilts = [a for a, _ in c.tilts]
            dirs = {int(((d + np.pi / 4) % (2 * np.pi)) // (np.pi / 2))
                    for a, d in c.tilts if a >= th.tilt_dir_min_deg}
            pairs = {}
            for (a, b), n in self._pairs.items():
                if cid in (a, b):
                    pairs[b if a == cid else a] = n
            st = CameraStatus(cid, coverage=float(c.occ.mean()), views=c.views,
                              tilt_max_deg=float(max(tilts)) if tilts else 0.0,
                              tilt_directions=len(dirs), sharpness=c.sharpness,
                              blurry=c.blurry, detections=c.detections, pairs=pairs,
                              linked=(not multi) or (comp[cid] == comp[ref] and size[comp[ref]] > 1),
                              last_corners=c.last_corners, last_markers=c.last_markers)
            st.hints = self._hints(c, st, comp, ref)
            ok = (st.coverage >= th.good_coverage and st.views >= th.good_views
                  and st.tilt_max_deg >= th.good_max_tilt_deg and st.linked)
            st.state = GOOD if ok else (SCANNING if c.detections or c.views else IDLE)
            out[cid] = st
        return out

    def _hints(self, c: _Cam, st: CameraStatus, comp, ref) -> list[str]:
        th = self.th
        hints = []
        if c.last_blurry:
            hints.append("hold the board still (blurry)")
        if not c.detections:
            hints.append("show the board to this camera")
            return hints
        if st.coverage < th.good_coverage:
            gy, gx = c.occ.shape
            xs = np.array_split(np.arange(gx), 3)
            ys = np.array_split(np.arange(gy), 3)
            frac = [(c.occ[np.ix_(ys[ry], xs[rx])].mean(), name)
                    for name, rx, ry in _REGIONS]
            worst = min(frac, key=lambda f: f[0])
            hints.append(f"cover the {worst[1]}")
        if st.tilt_max_deg < th.good_max_tilt_deg:
            hints.append(f"tilt more (max {st.tilt_max_deg:.0f}°, need {th.good_max_tilt_deg:.0f}°)")
        elif st.tilt_directions < 2:
            hints.append("tilt the other way too")
        if st.views < th.good_views:
            hints.append(f"more board positions ({st.views}/{th.good_views})")
        if not st.linked and ref is not None:
            # a camera of the reference's group (or, while the reference has no
            # partner yet, any other camera): the one it co-saw most / the busiest
            others = [o for o in self._order if comp[o] != comp[c.cam_id]]
            pref = [o for o in others if comp[o] == comp[ref]] or others
            if pref:
                partner = max(pref, key=lambda o: (st.pairs.get(o, 0), self._cams[o].views))
                hints.append(f"show to {c.cam_id} together with {partner}")
        return hints

    def summary(self) -> dict:
        """JSON-able snapshot (written into the session manifest as ``board_check``)."""
        sts = self.statuses()
        return {"board": self.board_cfg.to_dict(),
                "all_good": bool(sts) and all(s.state == GOOD for s in sts.values()),
                "cameras": {k: s.to_dict() for k, s in sts.items()},
                "pairs": self.pair_counts(),
                "thresholds": {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in self.th.__dict__.items()}}
