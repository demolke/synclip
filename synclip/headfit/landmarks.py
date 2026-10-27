"""MediaPipe FaceLandmarker wrapper + 3D landmark extraction on meshes.

Models are downloaded on first use to ``~/.cache/synclip/`` (never the repo):

* ``face_landmarker.task``          - 478 landmarks (Tasks API, IMAGE mode)
* ``blaze_face_short_range.tflite`` - face detector, only used for its score
  (FaceLandmarker does not expose a confidence).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np
import trimesh

from .render import Camera, cast, frame_camera, render, rotation_y

CACHE_DIR = Path(os.environ.get("SYNCLIP_CACHE", Path.home() / ".cache" / "synclip"))
_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

GNM_NPZ = Path(__file__).resolve().parent.parent / "gnm" / "gnm" / "shape" / "data" / "versions" / "v3_0" / "gnm_head.npz"

# --- MediaPipe indices that are reliable on eyeless / closed-mouth scans ---
_NOSE = [1, 2, 4, 5, 6, 19, 94, 168, 195, 197, 48, 278, 64, 294, 98, 327, 129, 358, 49, 279]
_BROWS = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55,  # one side
          300, 293, 334, 296, 336, 276, 283, 282, 295, 285]  # other side
_FOREHEAD = [9, 8, 151, 10, 108, 337, 109, 338]
_OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
               291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
_CHIN_JAW = [152, 148, 377, 176, 400, 175, 199, 200, 18, 83, 313, 201, 421, 32, 262,
             149, 378, 150, 379, 136, 365]
_CHEEKS = [50, 280, 205, 425, 187, 411, 123, 352, 117, 346, 118, 347, 101, 330,
           36, 266, 142, 371, 207, 427, 216, 436, 206, 426, 203, 423, 100, 329]
STABLE_IDX: list[int] = sorted(set(_NOSE + _BROWS + _FOREHEAD + _OUTER_LIPS + _CHIN_JAW + _CHEEKS))
# Inner-lip contour: unstable for alignment (moves with the mouth opening), but the
# detail fit uses it together with the outer contour to pin the vermilion border.
_INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
               308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
OUTER_LIP_IDX: list[int] = list(_OUTER_LIPS)  # closed contour, in order
INNER_LIP_IDX: list[int] = list(_INNER_LIPS)
LIP_IDX: list[int] = OUTER_LIP_IDX + INNER_LIP_IDX

_lock = threading.Lock()
_landmarker = None
_detector = None


def _ensure(url: str, name: str) -> Path:
    path = CACHE_DIR / name
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def _get_models():
    global _landmarker, _detector
    with _lock:
        if _landmarker is None:
            from mediapipe.tasks.python import BaseOptions, vision

            _landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(_ensure(_LANDMARKER_URL, "face_landmarker.task"))),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.1,
                    min_face_presence_confidence=0.1,
                )
            )
            _detector = vision.FaceDetector.create_from_options(
                vision.FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=str(_ensure(_DETECTOR_URL, "blaze_face_short_range.tflite"))),
                    running_mode=vision.RunningMode.IMAGE,
                    min_detection_confidence=0.05,
                )
            )
            # concurrent.futures shuts its executors down in a threading-atexit
            # hook, before ``atexit`` handlers; MediaPipe's close() needs them.
            reg = getattr(threading, "_register_atexit", None)
            try:
                (reg or atexit.register)(_close)
            except Exception:
                atexit.register(_close)
    return _landmarker, _detector


def _close():
    global _landmarker, _detector
    for m in (_landmarker, _detector):
        try:
            if m is not None:
                m.close()
        except Exception:
            pass
    _landmarker = _detector = None


def detect(image: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """Run FaceLandmarker on an RGB uint8 image.

    Returns ``(coords, confidence)``: coords is (478,2) pixel coordinates
    (column, row) or None when no face was found; confidence is the score of
    the BlazeFace detection containing the landmark centroid (None if the
    detector found nothing there although the landmarker did).
    """
    import mediapipe as mp

    landmarker, detector = _get_models()
    img = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
    h, w = img.shape[:2]
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    res = landmarker.detect(mpimg)
    if not res.face_landmarks:
        return None, None
    lm = res.face_landmarks[0]
    coords = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
    c = coords.mean(0)
    conf = None
    for det in detector.detect(mpimg).detections:
        bb = det.bounding_box
        if bb.origin_x <= c[0] <= bb.origin_x + bb.width and bb.origin_y <= c[1] <= bb.origin_y + bb.height:
            s = float(det.categories[0].score)
            conf = s if conf is None else max(conf, s)
    return coords, conf


# --------------------------------------------------------------------------- 3D

def detect_view(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    size: int = 512,
    zoom: bool = True,
) -> tuple[np.ndarray | None, float, dict]:
    """Render ``mesh`` seen through ``rotation`` (world->camera), detect, and
    back-project the 478 landmarks onto the surface.

    A second, zoomed render around the first detection improves precision.
    Returns ``(points (478,3) world with nan for misses, confidence, info)``.
    """
    cam = frame_camera(mesh, rotation, size)
    r = render(mesh, camera=cam)
    uv, conf = detect(r.image)
    info = {"image": r.image, "camera": cam, "uv": uv}
    if uv is None:
        return None, 0.0, info
    if zoom:
        lo, hi = uv.min(0), uv.max(0)
        mid = (lo + hi) / 2
        span = float(max(hi - lo)) * 1.9
        if span < 0.8 * size:
            # Re-centre the camera on the face in its own image plane.
            R = np.asarray(cam.rotation)
            cam_mid = (cam.center @ R.T) + np.array(
                [(mid[0] - size / 2) * cam.pixel_size, -(mid[1] - size / 2) * cam.pixel_size, 0.0]
            )
            zcam = Camera(rotation=R, center=cam_mid @ R, extent=span * cam.pixel_size, size=size)
            rz = render(mesh, camera=zcam)
            uvz, confz = detect(rz.image)
            if uvz is not None:
                cam, uv, info = zcam, uvz, {"image": rz.image, "camera": zcam, "uv": uvz}
                if confz is not None:
                    conf = confz if conf is None else max(conf, confz)
    hits = cast(mesh, cam, uv)
    return hits.points, float(conf or 0.05), info


def multiview_landmarks(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    yaws: tuple[float, ...] = (0.0, -25.0, 25.0),
    size: int = 512,
) -> tuple[np.ndarray | None, float, list[dict]]:
    """Per-landmark median of back-projected points over yawed views.

    ``rotation`` is the frontal world->camera rotation; yaw turns the head
    about the camera's vertical axis.  Confidence is that of the yaw-0 view.
    """
    pts, conf0, infos = [], 0.0, []
    for i, yaw in enumerate(yaws):
        R = rotation_y(yaw) @ np.asarray(rotation)
        p, conf, info = detect_view(mesh, R, size)
        infos.append(info)
        if i == 0:
            if p is None:
                return None, 0.0, infos
            conf0 = conf
        if p is not None:
            pts.append(p)
    stack = np.stack(pts)  # (V,478,3)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(stack, axis=0) if len(pts) > 1 else stack[0]
    return med, conf0, infos


def surface_bary(mesh: trimesh.Trimesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest surface point of each point -> (triangle index, barycentric)."""
    closest, _, tri = trimesh.proximity.closest_point(mesh, points)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[tri], closest)
    return np.asarray(tri, dtype=np.int64), bary


def bary_points(vertices: np.ndarray, faces: np.ndarray, lm: dict[int, tuple[int, np.ndarray]],
                idx: list[int] | None = None) -> np.ndarray:
    """Evaluate ``{mp_idx: (tri, bary)}`` on a mesh -> (len(idx),3)."""
    idx = sorted(lm) if idx is None else idx
    out = np.empty((len(idx), 3))
    for k, i in enumerate(idx):
        t, b = lm[i]
        out[k] = np.asarray(b) @ vertices[faces[t]]
    return out


def load_gnm_mean(npz_path: str | os.PathLike | None = None) -> trimesh.Trimesh:
    """GNM neutral mean head (template, bind pose) as a trimesh.

    Triangle indices match the npz ``triangles`` array (``process=False``).
    """
    d = np.load(npz_path or GNM_NPZ)
    return trimesh.Trimesh(d["template_vertex_positions"].astype(np.float64), d["triangles"], process=False)


def gnm_mean_landmarks(
    npz_path: str | os.PathLike | None = None,
    use_cache: bool = True,
) -> dict[int, tuple[int, np.ndarray]]:
    """MediaPipe landmarks of the GNM mean head as ``{mp_idx: (tri, bary(3,))}``.

    Uses the same render + detect + back-project pipeline as the target
    (frontal = identity rotation in the GNM frame, plus +-25 deg yaw views).
    Cached as JSON in ``~/.cache/synclip/``.
    """
    path = Path(npz_path or GNM_NPZ)
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|v1".encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"gnm_mean_landmarks_{key}.json"
    if use_cache and cache.exists():
        try:
            data = json.loads(cache.read_text())
            return {int(k): (int(v[0]), np.array(v[1:], dtype=np.float64)) for k, v in data["landmarks"].items()}
        except (ValueError, KeyError, TypeError, IndexError):
            pass  # corrupt / partial cache: recompute

    mesh = load_gnm_mean(path)
    pts, conf, _ = multiview_landmarks(mesh, np.eye(3))
    if pts is None:
        raise RuntimeError("MediaPipe found no face on the GNM mean head render")
    ok = np.isfinite(pts).all(1)
    idx = np.nonzero(ok)[0]
    tri, bary = surface_bary(mesh, pts[ok])
    out = {int(i): (int(t), b) for i, t, b in zip(idx, tri, bary)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.part")
    tmp.write_text(json.dumps({
        "npz": str(path), "confidence": conf,
        "landmarks": {str(i): [t, *map(float, b)] for i, (t, b) in out.items()},
    }))
    tmp.replace(cache)  # atomic: concurrent runs never read a half-written file
    return out
