"""Two Rigify characters with different proportions, one mocap take, one self-contained .blend.

    /opt/bpyenv/bin/python make_two_characters.py \
        --npz /home/user/data/mocap/take1_joints3d.npz \
        --out /home/user/data/mocap/deliver/two_characters_take1.blend [--spacing 2.0] [--control-mode LEGS_IK_ARMS_FK|FK|BOTH]

* rig_adult  : 1.75 m, legs x1.08, arms x0.95 (the same rig as target_rig.blend), blue/orange.
* rig_stocky : build_rig.py --preset stocky (1.2 m, big torso and head, short legs, long
               arms, wide shoulders/hips), purple/teal.
* Each rig is retargeted with retarget.py (its own action take1_<rig>, its own settings in
  Object.mocap_retarget, Params.mocap_space = OBJECT so the performance moves with the rig
  object: the rigs stand at x = -spacing/2 and +spacing/2).  --control-mode (default
  LEGS_IK_ARMS_FK): FK keys plus matching IK keys, legs switched to IK, arms to FK
  (FK = FK keys only, every limb in FK; BOTH = both key sets, every limb in FK).
* Self-contained: retarget.py is stored as the text "mocap_retarget.py" with Register on (the
  Mocap Retarget N-panel appears when the file is opened with Python auto-run allowed, or after
  Text > Run Script), and the npz is embedded as the text "take1_joints3d.npz" (the settings
  point to text:take1_joints3d.npz), so Re-bake / Live work without the original files.
* A camera tracks the midpoint of the two characters (keyed every frame, smoothed, framing
  both over the whole take), a sun light, Workbench / solid look, checker floor.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_rig  # noqa: E402
import retarget as mr  # noqa: E402

CHARACTERS = [
    # name, build_character kwargs
    ("rig_adult", dict(height=1.75, leg_scale=1.08, arm_scale=0.95, palette="default")),
    ("rig_stocky", dict(build_rig.preset_kwargs("stocky"))),
]


def _char_collection(name, rig):
    coll = bpy.data.collections.get(name + "_proxies")
    coll.name = name
    meta = bpy.data.objects.get(name + "_metarig")
    for ob in (rig, meta):
        if ob is None:
            continue
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        coll.objects.link(ob)
    return coll


def _sample_points(rigs, frames):
    """World positions of every DEF bone head/tail of all rigs, per frame: (F, N, 3)."""
    sc = bpy.context.scene
    out = []
    for f in frames:
        sc.frame_set(int(f))
        pts = []
        for rig in rigs:
            Mw = rig.matrix_world
            for pb in rig.pose.bones:
                if pb.name.startswith("DEF-"):
                    pts.append(Mw @ pb.head)
                    pts.append(Mw @ pb.tail)
        out.append([tuple(p) for p in pts])
    return np.array(out)


def add_tracking_camera(rigs, res=(960, 540), yaw=18.0, pitch=10.0, lens=35.0, margin=0.08, sigma=20.0):
    """A camera whose target follows the (smoothed) midpoint of all characters, at the
    smallest distance that keeps every DEF bone of every rig inside the frame (with margin)
    on every frame of the take."""
    sc = bpy.context.scene
    frames = np.arange(sc.frame_start, sc.frame_end + 1)
    P = _sample_points(rigs, frames)                      # (F, N, 3)
    ctr = 0.5 * (P.min(axis=1) + P.max(axis=1))           # bbox centre per frame
    ctr[:, :2] = np.stack([mr._gauss1d(ctr[:, i], sigma) for i in range(2)], axis=1)
    zc = 0.5 * float(np.percentile(P[..., 2].max(axis=1), 95))
    ctr[:, 2] = zc
    # camera orientation: looking along +Y (the performer faces -Y), turned by yaw, tilted down
    rot = Euler((math.radians(90 - pitch), 0, math.radians(yaw)), "XYZ").to_matrix()
    R = np.array(rot)
    fwd = -R[:, 2]
    W, H = res
    sensor = 36.0
    tx = sensor / 2 / lens * (1 - margin)                 # half-width tan
    ty = tx * H / W

    def fits(dist):
        C = ctr - fwd * dist                              # camera centres (F, 3)
        rel = P - C[:, None, :]
        cam = rel @ R                                     # camera coords (x right, y up, -z forward)
        z = -cam[..., 2]
        if (z <= 0.1).any():
            return False
        return bool((np.abs(cam[..., 0] / z) <= tx).all() and (np.abs(cam[..., 1] / z) <= ty).all())
    lo, hi = 0.5, 40.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    dist = hi
    tgt = bpy.data.objects.new("cam_target", None)
    sc.collection.objects.link(tgt)
    tgt.animation_data_create()
    act = bpy.data.actions.new("cam_target_track")
    tgt.animation_data.action = act
    slot = act.slots.new(id_type="OBJECT", name=tgt.name)
    tgt.animation_data.action_slot = slot
    from bpy_extras import anim_utils
    cb = anim_utils.action_ensure_channelbag_for_slot(act, slot)
    for i in range(3):
        fc = cb.fcurves.new("location", index=i)
        fc.keyframe_points.add(len(frames))
        co = np.empty(2 * len(frames))
        co[0::2] = frames
        co[1::2] = ctr[:, i]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [1] * len(frames))
        fc.update()
    cam_data = bpy.data.cameras.new("two_characters_cam")
    cam_data.lens = lens
    cam_data.sensor_width = sensor
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start, cam_data.clip_end = 0.05, 200.0
    cam = bpy.data.objects.new("two_characters_cam", cam_data)
    sc.collection.objects.link(cam)
    cam.parent = tgt
    cam.matrix_parent_inverse = Matrix.Identity(4)
    cam.matrix_basis = Matrix.Translation(Vector(-fwd * dist)) @ rot.to_4x4()
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = res
    print("camera: distance %.2f m, lens %g mm, centre z %.2f" % (dist, lens, zc))
    return cam


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="/home/user/data/mocap/take1_joints3d.npz")
    p.add_argument("--out", default="/home/user/data/mocap/deliver/two_characters_take1.blend")
    p.add_argument("--spacing", type=float, default=2.0, help="m between the two rig origins (along X)")
    p.add_argument("--take", default="take1", help="prefix of the action names")
    p.add_argument("--set", nargs="*", help="retarget param=value overrides for both rigs")
    p.add_argument("--no-embed", action="store_true", help="reference the npz path instead of embedding it")
    p.add_argument("--control-mode", default="LEGS_IK_ARMS_FK", choices=mr.CONTROL_MODES,
                   help="retarget control_mode of both rigs (default LEGS_IK_ARMS_FK: FK + IK keys, legs in IK, "
                        "arms in FK; FK = FK keys only)")
    a = p.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    mr.register()
    sc = bpy.context.scene
    npz = a.npz if a.no_embed else mr.embed_npz(a.npz)
    rigs = []
    for i, (name, kw) in enumerate(CHARACTERS):
        kw = dict(kw)
        height = kw.pop("height")
        x = (i - (len(CHARACTERS) - 1) / 2) * a.spacing
        rig = build_rig.build_character(name, height, location=(x, 0, 0), **kw)
        _char_collection(name, rig)
        prm = mr._parse_set(a.set, mr.Params(mocap_space="OBJECT", action_name="%s_%s" % (a.take, name),
                                             control_mode=a.control_mode))
        info = mr.retarget(rig, npz, prm)
        mr.settings_from_params(rig.mocap_retarget, prm, npz)
        print("%s: %s" % (name, {k: info[k] for k in ("action", "scale", "n_frames", "contact_frames")}))
        rigs.append(rig)
    sc.mocap_retarget.rig = rigs[0]
    mr.settings_from_params(sc.mocap_retarget, mr.params_from_settings(rigs[0].mocap_retarget), npz)
    floor = bpy.data.collections.new("floor")
    sc.collection.children.link(floor)
    build_rig.make_floor(floor, size=12.0)
    build_rig.enable_workbench_look(sc)
    sh = sc.display.shading
    sh.show_shadows = True
    sh.shadow_intensity = 0.6
    sc.display.light_direction = (0.35, -0.45, 0.82)
    sh.background_type = "VIEWPORT"
    sh.background_color = (0.08, 0.09, 0.11)
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.0
    sun_ob = bpy.data.objects.new("sun", sun)
    sun_ob.rotation_euler = Euler((math.radians(40), math.radians(15), math.radians(30)), "XYZ")
    sc.collection.objects.link(sun_ob)
    add_tracking_camera(rigs)
    # the 3D viewports of the saved file show solid / object colours too (when opened in the GUI)
    for scr in bpy.data.screens:
        for area in scr.areas:
            for sp in area.spaces:
                if sp.type == "VIEW_3D":
                    sp.shading.type = "SOLID"
                    sp.shading.color_type = "OBJECT"
    sc.frame_set(sc.frame_start)
    mr.embed_text_module()
    bpy.ops.file.pack_all()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=True)
    print("saved", a.out, "rigs:", [r.name for r in rigs], "frames %d-%d @ %d fps" % (
        sc.frame_start, sc.frame_end, sc.render.fps))


if __name__ == "__main__":
    main(sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1:])
