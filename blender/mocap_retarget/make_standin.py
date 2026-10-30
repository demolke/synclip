"""Convert MediaPipe *world* landmarks (one camera) into the mocap contract npz.

Development stand-in only.  MediaPipe world landmarks are hip-centred, metres,
camera-aligned with x right / y down / z away from the camera.  We convert to
Blender Z-up with the performer facing -Y, remove the camera pitch (estimate
"up" robustly from the ankle->hip and hip->shoulder vectors), and put the
lowest foot point on the floor every frame (root translation is otherwise 0).

Contract written:
    joints (T,33,3) float32, valid (T,33) bool, fps float, t (T,), names (33,)

Usage (plain numpy, no Blender needed):
    /home/user/venv/bin/python make_standin.py \
        /home/user/data/kp/kp/take1/camA.npz /home/user/data/mocap/standin_joints3d.npz
"""
import sys
import numpy as np

NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]


def rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def main(src, dst):
    d = np.load(src, allow_pickle=True)
    w = d["world"].astype(np.float64)                     # (T,33,3)
    fps = float(d["fps"]) if "fps" in d.files else 30.0
    t = d["t"] if "t" in d.files else np.arange(len(w)) / fps
    vis = d["vis"] if "vis" in d.files else np.ones(w.shape[:2])
    ok = np.isfinite(w).all(axis=2)
    if "detected" in d.files:
        ok &= d["detected"][:, None]
    ok &= vis > 0.0

    # MediaPipe (x right, y down, z away from camera) -> Blender (X, Y=z, Z=-y)
    j = np.stack([w[..., 0], w[..., 2], -w[..., 1]], axis=-1)

    # remove camera pitch/roll: robust median "up" over valid frames
    frame_ok = ok[:, [11, 12, 23, 24, 27, 28]].all(axis=1)
    hip = j[frame_ok][:, [23, 24]].mean(1)
    sho = j[frame_ok][:, [11, 12]].mean(1)
    ank = j[frame_ok][:, [27, 28]].mean(1)
    v1 = hip - ank
    v2 = sho - hip
    v = np.concatenate([v1 / np.linalg.norm(v1, axis=1, keepdims=True),
                        v2 / np.linalg.norm(v2, axis=1, keepdims=True)])
    up = np.median(v, axis=0)
    R = rot_between(up, np.array([0.0, 0.0, 1.0]))
    j = j @ R.T
    print("camera tilt removed: %.1f deg" % np.degrees(np.arccos(np.clip(up[2] / np.linalg.norm(up), -1, 1))))

    # MediaPipe world landmarks are garbage when the performer is seen from behind
    # (trunk upside-down / lying while actually walking).  Flag those frames invalid:
    # the retargeter gap-fills them.  (The performer never bends > 65 deg in this take.)
    tr = (j[:, 11] + j[:, 12]) / 2 - (j[:, 23] + j[:, 24]) / 2
    tilt = np.degrees(np.arccos(np.clip(tr[:, 2] / (np.linalg.norm(tr, axis=1) + 1e-9), -1, 1)))
    bad = ~(tilt < 65.0) & ok.all(1)
    ok &= ~bad[:, None]
    print("implausible (trunk tilt > 65 deg) frames flagged invalid:", int(bad.sum()))

    # ground: lowest foot point on the floor each frame (lightly smoothed)
    feet = j[:, [27, 28, 29, 30, 31, 32], 2]
    fok = ok[:, [27, 28, 29, 30, 31, 32]]
    low = np.where(fok, feet, np.inf).min(axis=1)
    good = np.isfinite(low)
    idx = np.arange(len(low))
    low = np.interp(idx, idx[good], low[good])
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    k /= k.sum()
    lowp = np.pad(low, 6, mode="edge")
    low = np.convolve(lowp, k, mode="valid")
    j[..., 2] -= low[:, None]

    j[~ok] = np.nan
    np.savez(dst, joints=j.astype(np.float32), valid=ok, fps=np.float64(fps),
             t=np.asarray(t, np.float64), names=np.array(NAMES))
    print("wrote", dst, j.shape, "valid frames", int(ok.all(1).sum()), "fps", fps)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/user/data/kp/kp/take1/camA.npz",
         sys.argv[2] if len(sys.argv) > 2 else "/home/user/data/mocap/standin_joints3d.npz")
