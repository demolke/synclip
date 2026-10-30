"""
Export a capmo character animation + camera rig to GLB and re-parse it.

Pure numpy/stdlib: we build a tiny skeleton and pose track directly, a
:class:`RigCalibration` with cameras placed at known world poses (rotation
matrices built by hand -- no cv2 dependency for placement), export to a temp
GLB, then re-parse the container and assert the structure, the camera centres,
the animation sampler/frame counts, the skeleton hierarchy, and the OpenCV ->
glTF -Z/+Y camera flip.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..calibration import CameraCalibration, RigCalibration
from ..export_glb import (
    _RX_180,
    _read_glb,
    export_animation_glb,
)
from ..retarget import (
    IDENTITY_QUAT,
    default_skeleton,
    forward_kinematics,
    retarget_pose,
)
from ..retarget import quat_to_matrix


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _bind_landmarks(skel) -> np.ndarray:
    """(33, 3) landmarks whose joints equal the skeleton's bind pose."""
    from .. import retarget as R
    pts = np.zeros((33, 3))
    for ji, (_, src) in enumerate(R.JOINTS):
        for s in src:
            pts[s] = skel.bind_joints[ji]
    return pts


def _bent_landmarks(skel) -> np.ndarray:
    """A second frame: shift the whole body and bend the left elbow."""
    pts = _bind_landmarks(skel)
    pts += np.array([0.3, 0.1, -0.2])          # translate root
    # move the left wrist (landmark 15) so the forearm bends
    pts[15] = pts[13] + np.array([0.0, 0.25, 0.0])
    return pts


def _make_poses(skel):
    valid = np.ones(33, bool)
    p0 = retarget_pose(skel, _bind_landmarks(skel), valid, t_ms=0.0)
    p1 = retarget_pose(skel, _bent_landmarks(skel), valid, t_ms=33.0)
    p2 = retarget_pose(skel, _bent_landmarks(skel), valid, t_ms=66.0)
    return [p0, p1, p2]


def _cam_at(cam_id, C, R, *, w=640, h=480, fx=500.0, fy=500.0):
    """A CameraCalibration whose world->cam is X_c = R @ X_w + t (t from C)."""
    C = np.asarray(C, np.float64)
    R = np.asarray(R, np.float64)
    t = -R @ C                                  # since C = -R.T @ t
    K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1.0]])
    # Store rotation as a matrix via a tiny shim so we avoid cv2.Rodrigues.
    cam = CameraCalibration(
        cam_id=cam_id, image_size=(w, h), K=K, dist=np.zeros(5),
        rvec=np.zeros(3), tvec=t,
    )
    cam.rotation_matrix = lambda R=R: R         # type: ignore[method-assign]
    return cam


def _look_at_rotation(C, target=(0.0, 0.0, 0.0), up=(0.0, -1.0, 0.0)) -> np.ndarray:
    """OpenCV world->cam rotation R for a camera at C looking at *target*.

    OpenCV camera looks down +Z with +Y down. R's rows are the world-frame
    camera axes (x_cam, y_cam, z_cam).
    """
    C = np.asarray(C, np.float64)
    target = np.asarray(target, np.float64)
    z = target - C                              # +Z_cam points at the target
    z /= np.linalg.norm(z)
    up = np.asarray(up, np.float64)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])                 # rows = camera axes in world


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_glb_is_valid_container(tmp_path):
    skel = default_skeleton()
    poses = _make_poses(skel)
    out = tmp_path / "anim.glb"
    export_animation_glb(str(out), skel, poses, rig=None)

    raw = out.read_bytes()
    import struct
    magic, ver, total = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67
    assert ver == 2
    assert total == len(raw)

    j, bin_data = _read_glb(str(out))
    assert j["asset"]["version"] == "2.0"
    assert len(bin_data) == j["buffers"][0]["byteLength"]
    assert "animations" in j and "nodes" in j and "scenes" in j


def test_skeleton_node_hierarchy(tmp_path):
    skel = default_skeleton()
    poses = _make_poses(skel)
    out = tmp_path / "anim.glb"
    export_animation_glb(str(out), skel, poses, rig=None, include_bone_geometry=False)
    j, _ = _read_glb(str(out))

    # One root node + one node per bone.
    bone_nodes = [n for n in j["nodes"] if n.get("name") in skel.bone_names]
    assert len(bone_nodes) == skel.num_bones

    # The pelvis root is a scene root.
    names = [j["nodes"][i].get("name") for i in j["scenes"][0]["nodes"]]
    assert "pelvis_root" in names

    # Parent relationships: every bone's node must be a child of its FK parent's
    # node (root for parent < 0).
    name_to_idx = {n.get("name"): i for i, n in enumerate(j["nodes"])}
    children_of = {i: set(n.get("children", [])) for i, n in enumerate(j["nodes"])}
    root_idx = name_to_idx["pelvis_root"]
    for b, bname in enumerate(skel.bone_names):
        node_idx = name_to_idx[bname]
        par = skel.parent[b]
        parent_idx = root_idx if par < 0 else name_to_idx[skel.bone_names[par]]
        assert node_idx in children_of[parent_idx], f"{bname} not under its parent"


def test_animation_frames_and_seconds(tmp_path):
    skel = default_skeleton()
    poses = _make_poses(skel)
    out = tmp_path / "anim.glb"
    export_animation_glb(str(out), skel, poses, rig=None, include_bone_geometry=False)
    j, bin_data = _read_glb(str(out))

    anim = j["animations"][0]
    # samplers: root translation + root rotation + one per bone.
    assert len(anim["samplers"]) == 2 + skel.num_bones
    assert len(anim["channels"]) == 2 + skel.num_bones

    # The time accessor (sampler input) must be in seconds: t_ms/1000.
    in_acc = anim["samplers"][0]["input"]
    acc = j["accessors"][in_acc]
    assert acc["count"] == len(poses)
    view = j["bufferViews"][acc["bufferView"]]
    off = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    times = np.frombuffer(bin_data, np.float32, count=acc["count"], offset=off)
    assert np.allclose(times, [p.t_ms / 1000.0 for p in poses], atol=1e-6)


def test_cameras_named_and_centred(tmp_path):
    skel = default_skeleton()
    poses = _make_poses(skel)

    placements = {
        "camA": (np.array([2.0, 0.0, 0.0]), _look_at_rotation([2.0, 0.0, 0.0])),
        "camB": (np.array([0.0, 1.5, -2.0]), _look_at_rotation([0.0, 1.5, -2.0])),
        "camC": (np.array([-1.0, 0.5, 1.0]), _look_at_rotation([-1.0, 0.5, 1.0])),
    }
    rig = RigCalibration()
    for cid, (C, R) in placements.items():
        rig.cameras[cid] = _cam_at(cid, C, R)

    out = tmp_path / "anim.glb"
    export_animation_glb(str(out), skel, poses, rig=rig)
    j, _ = _read_glb(str(out))

    assert "cameras" in j and len(j["cameras"]) == len(placements)

    # One camera node per rig camera, named by cam_id.
    cam_nodes = {n["name"]: n for n in j["nodes"] if "camera" in n}
    assert set(cam_nodes) == set(placements)

    for cid, (C, R) in placements.items():
        node = cam_nodes[cid]
        assert np.allclose(node["translation"], C, atol=1e-6), cid
        # perspective camera present
        cam = j["cameras"][node["camera"]]
        assert cam["type"] == "perspective"
        assert cam["perspective"]["yfov"] > 0


def test_camera_z_flip_points_at_scene(tmp_path):
    """A camera placed looking at the origin must, in glTF, point its -Z toward
    the origin (the OpenCV +Z look flipped to glTF -Z)."""
    skel = default_skeleton()
    poses = _make_poses(skel)

    C = np.array([3.0, 0.0, 0.0])
    R = _look_at_rotation(C, target=(0.0, 0.0, 0.0))
    rig = RigCalibration()
    rig.cameras["cam"] = _cam_at("cam", C, R)

    out = tmp_path / "anim.glb"
    export_animation_glb(str(out), skel, poses, rig=rig)
    j, _ = _read_glb(str(out))

    node = next(n for n in j["nodes"] if n.get("name") == "cam")
    x, y, z, w = node["rotation"]
    R_node = quat_to_matrix(np.array([w, x, y, z]))   # node local->world
    forward_world = R_node @ np.array([0.0, 0.0, -1.0])   # glTF camera looks -Z
    to_origin = (np.zeros(3) - C)
    to_origin /= np.linalg.norm(to_origin)
    assert np.dot(forward_world, to_origin) > 0.99


def test_node_transforms_reproduce_forward_kinematics(tmp_path):
    """Walking the exported node tree with the animated TRS must reproduce the
    same joint world positions as forward_kinematics for each frame."""
    skel = default_skeleton()
    poses = _make_poses(skel)
    out = tmp_path / "anim.glb"
    j = export_animation_glb(str(out), skel, poses, rig=None,
                             include_bone_geometry=False)

    name_to_idx = {n.get("name"): i for i, n in enumerate(j["nodes"])}
    nodes = j["nodes"]

    def node_local(idx, rot_xyzw, trans):
        M = np.eye(4)
        x, y, z, w = rot_xyzw
        M[:3, :3] = quat_to_matrix(np.array([w, x, y, z]))
        M[:3, 3] = trans
        return M

    for fi, pose in enumerate(poses):
        # Build per-node animated TRS from the pose (mirror what the sampler
        # outputs would be at this keyframe).
        root_idx = name_to_idx["pelvis_root"]
        rp = pose.root_pos if np.all(np.isfinite(pose.root_pos)) else \
            skel.bind_joints[0]
        # root node world matrix
        rr = pose.root_rot
        Mroot = node_local(root_idx, [rr[1], rr[2], rr[3], rr[0]], rp)

        # bone node world matrices, top-down
        world = {root_idx: Mroot}
        order = list(range(skel.num_bones))
        bone_world = {}
        for b in order:
            par = skel.parent[b]
            parent_idx = root_idx if par < 0 else name_to_idx[skel.bone_names[par]]
            node = nodes[name_to_idx[skel.bone_names[b]]]
            lr = pose.local_rot[b]
            local = node_local(0, [lr[1], lr[2], lr[3], lr[0]],
                               node["translation"])
            Mw = world[parent_idx] @ local
            world[name_to_idx[skel.bone_names[b]]] = Mw
            bone_world[b] = Mw

        # The exported skeleton reproduces FK joint *directions* and the chained
        # head positions. Check each bone node origin equals the FK head of that
        # bone.
        fk = forward_kinematics(skel, pose)
        for b in range(skel.num_bones):
            head_node = bone_world[b][:3, 3]
            fk_head = fk[skel.head_joint[b]]
            assert np.allclose(head_node, fk_head, atol=1e-6), \
                f"frame {fi} bone {skel.bone_names[b]} head mismatch"
            # tail = head + R_node @ (rest_len * +Y_local?) -- instead verify the
            # bone *direction*: node local +Y rotated to world should match the
            # FK bone direction.
            # (geometry box is along +Y in a child node; here just check head.)
