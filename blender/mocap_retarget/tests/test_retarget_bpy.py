"""Regression tests of retarget.py / validate.py in Blender (bpy module).

    /opt/bpyenv/bin/python -m pytest blender/mocap_retarget/tests
"""
import json
import math
import os
import sys
import warnings

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

import retarget as mr  # noqa: E402
from conftest import NPZ, quiet_params  # noqa: E402

CHECK_BONES = ("torso", "hips", "chest", "head", "upper_arm_fk.L", "forearm_fk.R", "hand_fk.L",
               "thigh_fk.L", "shin_fk.R", "foot_fk.L")


def _world_hips(rig, frames):
    sc = rig.users_scene[0]
    out = []
    for f in frames:
        sc.frame_set(f)
        out.append(np.array(rig.matrix_world @ rig.pose.bones["torso"].head))
    return np.array(out)


def _pose(rig, frames):
    """{frame: {bone: 4x4 armature-space matrix}}, frames evaluated in the given order."""
    sc = rig.users_scene[0]
    out = {}
    for f in frames:
        sc.frame_set(f)
        out[f] = {n: np.array(rig.pose.bones[n].matrix) for n in CHECK_BONES}
    return out


# ------------------------------------------------------------------ root / world / live
def test_world_space_root_ignores_rig_placement(rig):
    """mocap_space WORLD: the mocap coordinates are world coordinates, so moving / turning
    the rig object does not move the performance (the root scale is applied around the
    mocap world origin, not the rig origin). OBJECT: the performance is attached to the rig
    object, so it moves by exactly the rig's transform."""
    frames = (1, 40, 80, 120)
    prm = quiet_params(foot_lock=True)
    mr.retarget(rig, NPZ, prm)
    h0 = _world_hips(rig, frames)
    rig.location = (5.0, 0.0, 0.0)
    rig.rotation_euler = (0.0, 0.0, math.radians(90))
    bpy.context.view_layer.update()
    mr.retarget(rig, NPZ, prm)
    h1 = _world_hips(rig, frames)
    assert np.abs(h1 - h0).max() < 1e-4
    # OBJECT: the same rig transform carries the performance along
    rig.location = (0.0, 0.0, 0.0)
    rig.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    prm_o = quiet_params(foot_lock=True, mocap_space="OBJECT")
    mr.retarget(rig, NPZ, prm_o)
    o0 = _world_hips(rig, frames)
    rig.location = (5.0, 0.0, 0.0)
    rig.rotation_euler = (0.0, 0.0, math.radians(90))
    bpy.context.view_layer.update()
    mr.retarget(rig, NPZ, prm_o)
    o1 = _world_hips(rig, frames)
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.abs(o1 - (o0 @ Rz.T + [5.0, 0.0, 0.0])).max() < 1e-4


@pytest.mark.parametrize("root_motion", ["FULL", "IN_PLACE", "FROM_FEET"])
def test_live_pose_equals_baked_pose(rig, root_motion):
    """Live mode solves each frame like the bake (shared root target, frame range and
    observed-joint mask). Compared with the bake's temporal steps off (foot lock, rotation
    smoothing ...), which live mode does not have; FROM_FEET: live keeps the hips in place
    horizontally (no foot integration), so only rotations and height are compared."""
    prm = quiet_params(root_motion=root_motion, frame_start=10, frame_end=99, scene_start=1)
    mr.retarget(rig, NPZ, prm)
    frames = list(range(1, 91))
    baked = _pose(rig, frames)
    s = rig.mocap_retarget
    mr.settings_from_params(s, prm, NPZ)
    s.live = True                               # unassigns the action, installs the handler
    assert rig.animation_data.action is None
    live = _pose(rig, frames)                   # scrubbed in order: same solver state
    for f in (1, 30, 60, 90):
        for n in CHECK_BONES:
            A, B = baked[f][n], live[f][n]
            assert np.abs(A[:3, :3] - B[:3, :3]).max() < 1e-4, (f, n)
            if root_motion != "FROM_FEET":
                assert np.abs(A[:3, 3] - B[:3, 3]).max() < 1e-4, (f, n)
            else:
                assert abs(A[2, 3] - B[2, 3]) < 1e-4, (f, n)


def test_frame_range_errors(rig):
    mocap = mr.load_mocap(NPZ)
    model = mr.build_model(rig)
    for kw, msg in ((dict(frame_start=500), "frame_start 500 is outside"),
                    (dict(frame_end=500), "frame_end 500 is outside"),
                    (dict(frame_start=50, frame_end=20), "frame_end 20 is before frame_start 50")):
        with pytest.raises(ValueError, match=msg):
            mr.solve_sequence(model, mocap, mr.Params(**kw))


def test_live_off_restores_the_action_and_slot(rig):
    mr.retarget(rig, NPZ, quiet_params())
    act, slot = rig.animation_data.action, rig.animation_data.action_slot
    s = rig.mocap_retarget
    mr.settings_from_params(s, quiet_params(), NPZ)
    s.live = True
    assert rig.animation_data.action is None
    bpy.context.scene.frame_set(5)
    assert mr._live_key(rig) in mr._LIVE and rig.name not in mr._LIVE
    s.live = False
    assert rig.animation_data.action == act
    assert rig.animation_data.action_slot.identifier == slot.identifier
    assert "mocap_live_prev_action" not in rig.keys()


def test_live_render_shows_the_rendered_frame(rig, tmp_path):
    """F12 / animation renders in live mode: the pose of the frame being rendered (the live
    solve runs on frame_change_pre, before the render depsgraph evaluates the frame)."""
    from bpy_extras.object_utils import world_to_camera_view
    sc = bpy.context.scene
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.hide_render = True
    bpy.ops.mesh.primitive_cube_add(size=0.2)
    cube = bpy.context.active_object
    cube.parent, cube.parent_type, cube.parent_bone = rig, "BONE", "torso"
    mat = bpy.data.materials.new("glow")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[1].default_value = 5.0
    nt.links.new(em.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs[0])
    cube.data.materials.append(mat)
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type, cam_d.ortho_scale = "ORTHO", 4.0
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.location = (0.0, 0.0, 10.0)                  # top view: the walk is in XY
    sc.world = bpy.data.worlds.new("black")
    sc.render.engine = "CYCLES"
    sc.cycles.samples, sc.cycles.device = 1, "CPU"
    sc.render.resolution_x = sc.render.resolution_y = 64
    sc.render.resolution_percentage = 100
    s = rig.mocap_retarget
    mr.settings_from_params(s, quiet_params(), NPZ)
    s.live = True
    assert mr._live_handler in bpy.app.handlers.frame_change_pre
    sc.frame_set(120)                       # the render starts elsewhere than its first frame
    sc.render.filepath = str(tmp_path / "f####")
    sc.render.image_settings.file_format = "PNG"
    sc.frame_start, sc.frame_end, sc.frame_step = 1, 120, 119
    bpy.ops.render.render(animation=True)
    exp = {}
    for f in (1, 120):
        sc.frame_set(f)
        p = world_to_camera_view(sc, cam, cube.matrix_world.translation)
        exp[f] = np.array([p.x, p.y]) * 64
    assert np.linalg.norm(exp[120] - exp[1]) > 5          # a stale pose would be visible
    for f in (1, 120):
        img = bpy.data.images.load(str(tmp_path / ("f%04d.png" % f)))
        a = np.array(img.pixels[:]).reshape(64, 64, 4)[..., 0]
        ys, xs = np.nonzero(a > 0.5)
        got = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
        assert np.linalg.norm(got - exp[f]) < 1.5, (f, got, exp[f])


# ------------------------------------------------------------- params / registration
def test_params_from_dict_drops_unknown_keys():
    d = mr.asdict(mr.Params(smooth_sigma=2.5))
    d["renamed_old_param"] = 1
    with pytest.warns(UserWarning, match="renamed_old_param"):
        prm = mr.Params.from_dict(d)
    assert prm.smooth_sigma == 2.5


def test_settings_from_action_with_unknown_stored_key(rig):
    mr.retarget(rig, NPZ, quiet_params(smooth_sigma=2.5))
    act = rig.animation_data.action
    d = json.loads(act["mocap_params"])
    d["renamed_old_param"] = 1
    act["mocap_params"] = json.dumps(d)
    rig.mocap_retarget.smooth_sigma = 0.5
    bpy.context.view_layer.objects.active = rig
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with bpy.context.temp_override(active_object=rig, object=rig):
            assert bpy.ops.mocap_retarget.settings_from_action() == {"FINISHED"}
    assert rig.mocap_retarget.smooth_sigma == pytest.approx(2.5)


def _module_copy(name, version):
    """retarget.py loaded as another module (an installed add-on / embedded text copy)."""
    import types
    with open(mr.__file__) as fh:
        src = fh.read()
    old = '"version": %s,' % (tuple(mr.bl_info["version"]),)
    assert old in src
    src = src.replace(old, '"version": %s,' % (version,), 1)
    mod = types.ModuleType(name)
    mod.__file__ = mr.__file__
    sys.modules[name] = mod
    exec(compile(src, mr.__file__, "exec"), mod.__dict__)
    return mod


def test_registration_keeps_an_equal_or_newer_copy(rig):
    cur = tuple(mr.bl_info["version"])
    try:
        older = _module_copy("mr_older_copy", (1, 0, 0))
        assert older.register() is False                     # embedded older text module
        assert mr._registered_settings_class().mocap_retarget_version == cur
        same = _module_copy("mr_same_copy", cur)
        assert same.register() is False
        newer = _module_copy("mr_newer_copy", (99, 0, 0))
        assert newer.register() is True                      # a newer add-on replaces it
        assert mr._registered_settings_class().mocap_retarget_version == (99, 0, 0)
        assert mr.register() is False                        # ... and is not overridden again
        assert hasattr(bpy.types.Object, "mocap_retarget")
    finally:
        cls = mr._registered_settings_class()
        if cls is not None and cls is not mr.MocapRetargetSettings:
            sys.modules[cls.__module__].unregister()
        mr.register()
        for n in ("mr_older_copy", "mr_same_copy", "mr_newer_copy"):
            sys.modules.pop(n, None)
    assert mr._registered_settings_class() is mr.MocapRetargetSettings


# ---------------------------------------------------------------------- input checks
def test_load_mocap_refuses_pickled_arrays(tmp_path):
    d = dict(np.load(NPZ))
    assert d["names"].dtype.kind == "U"                       # the contract: no object arrays
    p = tmp_path / "pickled.npz"
    np.savez(p, joints=np.array(list(d["joints"]), dtype=object), valid=d["valid"], fps=d["fps"])
    with pytest.raises(ValueError, match="allow_pickle"):
        mr.load_mocap(str(p))
    assert mr.load_mocap(NPZ)["joints"].shape == (120, 33, 3)


def test_never_observed_joint_is_an_error(tmp_path):
    d = dict(np.load(NPZ))
    d["valid"] = d["valid"].copy()
    d["valid"][:, mr.MP["left_index"]] = False
    p = tmp_path / "noindex.npz"
    np.savez(p, **d)
    with pytest.raises(ValueError, match="never observed.*left_index"):
        mr.preprocess(mr.load_mocap(str(p)), mr.Params())


# ----------------------------------------------------------------------- bake output
def test_bake_sets_fps_base_on_the_rigs_scene(rig, tmp_path):
    d = dict(np.load(NPZ))
    d["fps"] = np.float64(30000 / 1001)
    p = tmp_path / "ntsc.npz"
    np.savez(p, **d)
    home = rig.users_scene[0]
    other = bpy.data.scenes.new("Other")
    other.frame_end = 999
    with bpy.context.temp_override(scene=other):
        mr.retarget(rig, str(p), quiet_params())
    assert other.frame_end == 999 and other.render.fps_base == 1.0
    assert home.render.fps == 30 and home.render.fps_base == pytest.approx(1.001)
    assert home.render.fps / home.render.fps_base == pytest.approx(30000 / 1001)
    assert (home.frame_start, home.frame_end) == (1, 120)


def test_step_warns_that_planted_feet_are_lost(rig):
    with pytest.warns(UserWarning, match="foot lock"):
        mr.retarget(rig, NPZ, mr.Params(step=2))


def test_validate_reports_sampled_worst_flip_frames(rig, tmp_path):
    """validate.py --step 2: the worst-flip frames are frames that were evaluated, and an
    unknown stored parameter does not break it."""
    import validate
    mr.retarget(rig, NPZ, mr.Params())
    act = rig.animation_data.action
    d = json.loads(act["mocap_params"])
    d["renamed_old_param"] = 1
    act["mocap_params"] = json.dumps(d)
    act["mocap_npz"] = NPZ
    blend = str(tmp_path / "baked.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    out = str(tmp_path / "val.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            validate.main(["--blend", blend, "--json", out, "--start", "1", "--end", "61",
                           "--step", "2"])
        except SystemExit:
            pass
    rep = json.load(open(out))
    worst = rep["e_flips"]["worst_excess_over_mocap"]
    assert worst and all(w["frame"] in range(3, 62, 2) for w in worst)
