"""Workbench preview render of a retargeted rig (headless).

    /opt/bpyenv/bin/python render_preview.py --blend retargeted.blend --out preview.mp4 \
        [--start 1 --end 300 --step 1] [--res 852x480]
        [--view front34|track34|side|front|scene | --calib /home/user/data/mocap/rig.json:camB]
        [--ghost take.npz --ghost-mode overlay|side] [--overlay] [--no-stamp]
        [--png-dir DIR] [--keep-png]

* --view: a camera that follows the rig's torso (XY only, fixed height); track34 is a
  tighter front-3/4.  --calib FILE:CAM instead uses a calibrated camera (OpenCV K, rvec,
  tvec in the npz's world frame, see capmo rig.json), so the render has the same perspective
  as that camera's video (resolution is forced to the camera's aspect).
* --overlay: transparent background, no floor (RGBA PNGs to composite over the video).
* --ghost NPZ: the processed mocap joints as small dots, mapped exactly like the rig's root
  (overlay mode: on top of the rig) or 1 m to the side with sticks (side mode).  A joint is
  hidden on frames where it was not observed.
* Object colours: left = blue, right = orange, fins = green.  PNGs -> MP4 via ffmpeg.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_camera(rig, view, height):
    sc = bpy.context.scene
    tgt = bpy.data.objects.new("cam_target", None)
    sc.collection.objects.link(tgt)
    c = tgt.constraints.new("COPY_LOCATION")
    c.target = rig
    c.subtarget = "torso"
    c.use_z = False
    c.use_offset = False
    off = {"front34": (2.2, -3.3, 0.55, 40), "track34": (1.5, -2.3, 0.45, 42),
           "side": (3.8, 0.0, 0.45, 40), "front": (0.0, -3.9, 0.45, 40)}[view]
    cam_data = bpy.data.cameras.new("preview_cam")
    cam_data.lens = off[3]
    cam = bpy.data.objects.new("preview_cam", cam_data)
    sc.collection.objects.link(cam)
    cam.parent = tgt
    s = height / 1.75
    cam.location = (off[0] * s, off[1] * s, (0.9 + off[2]) * s)
    tr = cam.constraints.new("TRACK_TO")
    aim = bpy.data.objects.new("cam_aim", None)
    sc.collection.objects.link(aim)
    aim.parent = tgt
    aim.location = (0, 0, 0.85 * s)
    tr.target = aim
    tr.track_axis = "TRACK_NEGATIVE_Z"
    tr.up_axis = "UP_Y"
    sc.camera = cam
    return cam


def setup_calib_camera(spec, width):
    """Blender camera from an OpenCV calibration (x = K [R|t] X, world -> camera)."""
    path, cam_id = spec.rsplit(":", 1)
    c = json.load(open(path))["cameras"][cam_id]
    K = np.array(c["K"], float)
    W, H = c["image_size"]
    rv = Vector(c["rvec"])
    R = Matrix.Rotation(rv.length, 3, rv.normalized()) if rv.length > 1e-12 else Matrix.Identity(3)
    t = Vector(c["tvec"])
    C = -(R.transposed() @ t)
    # OpenCV camera axes (x right, y down, z forward) -> Blender camera (x right, y up, -z forward)
    Rbl = R.transposed() @ Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    cam_data = bpy.data.cameras.new("calib_" + cam_id)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = K[0, 0] / W * 36.0
    cam_data.shift_x = -(K[0, 2] - W / 2) / W
    cam_data.shift_y = (K[1, 2] - H / 2) / W
    cam_data.clip_start, cam_data.clip_end = 0.05, 100.0
    cam = bpy.data.objects.new("calib_" + cam_id, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.matrix_world = Matrix.Translation(C) @ Rbl.to_4x4()
    bpy.context.scene.camera = cam
    height = int(round(width * H / W / 2)) * 2
    if (K[0, 0] - K[1, 1]) > 1e-3 * K[0, 0]:
        print("warning: fx != fy, pixel aspect not modelled")
    return cam, (width, height)


def _keys(ob, path, frames, vals, constant=False):
    from bpy_extras import anim_utils
    if ob.animation_data is None or ob.animation_data.action is None:
        a = bpy.data.actions.new(ob.name)
        ob.animation_data_create()
        ob.animation_data.action = a
        ob.animation_data.action_slot = ob.animation_data.action_slot or a.slots.new(id_type="OBJECT", name=ob.name)
    a = ob.animation_data.action
    cb = anim_utils.action_ensure_channelbag_for_slot(a, ob.animation_data.action_slot)
    for i in range(vals.shape[1]):
        fc = cb.fcurves.new(path, index=i)
        fc.keyframe_points.add(len(frames))
        co = np.empty(2 * len(frames))
        co[0::2] = frames
        co[1::2] = vals[:, i]
        fc.keyframe_points.foreach_set("co", co)
        if constant:
            fc.keyframe_points.foreach_set("interpolation", [0] * len(frames))
        fc.update()


def add_ghost(npz, rig, mode="overlay", offset=(1.1, 0.0, 0.0)):
    """Mocap joints (processed exactly like the bake: swap repair, outliers, smoothing),
    mapped to the rig's space with the same root mapping as the bake (xy * scale, height
    scaled above the stance-ankle level).  overlay: small dots on the rig; side: dots +
    sticks 1 m beside it.  Unobserved joints are hidden."""
    import retarget as mr
    act = rig.animation_data.action
    prm = mr.Params(**json.loads(act["mocap_params"]))
    info = json.loads(act["mocap_info"])
    J, vm = mr.preprocess(mr.load_mocap(npz), prm, return_mask=True)
    f0 = max(0, prm.frame_start)
    f1 = len(J) - 1 if prm.frame_end < 0 else min(prm.frame_end, len(J) - 1)
    J, vm = J[f0:f1 + 1], vm[f0:f1 + 1]
    s = info["scale"]
    G = J.copy()
    if prm.root_motion == "FULL":
        G[..., :2] *= s
    else:  # in place / from feet: follow the rig's evaluated hip centre
        sc = bpy.context.scene
        hc = (J[:, 23] + J[:, 24]) / 2
        hip_rig = np.zeros((len(J), 3))
        for k in range(len(J)):
            sc.frame_set(int(prm.scene_start + k))
            hip_rig[k] = (np.array(rig.pose.bones["DEF-thigh.L"].head) +
                          np.array(rig.pose.bones["DEF-thigh.R"].head)) / 2
        G[..., :2] = (J[..., :2] - hc[:, None, :2]) * s + hip_rig[:, None, :2]
    G[..., 2] = info.get("rig_ankle_height", 0.0) + s * (J[..., 2] - info.get("mocap_ankle_height", 0.0))
    if mode == "side":
        G += np.array(offset)
    if getattr(prm, "mocap_space", "WORLD") == "OBJECT":   # mocap frame attached to the rig object
        Mw = np.array(rig.matrix_world)
        G = G @ Mw[:3, :3].T + Mw[:3, 3]
    frames = prm.scene_start + np.arange(len(J), dtype=np.float64)
    import bmesh
    me = bpy.data.meshes.new("ghost_dot")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.022 if mode == "side" else 0.028)
    bm.to_mesh(me)
    bm.free()
    coll = bpy.data.collections.new("ghost")
    bpy.context.scene.collection.children.link(coll)
    used = [0, 7, 8] + list(range(11, 17)) + list(range(23, 29)) + [31, 32]
    for k in used:
        ob = bpy.data.objects.new("ghost_%02d" % k, me)
        coll.objects.link(ob)
        ob.color = (0.98, 0.98, 0.25, 1) if mode == "overlay" else (
            (0.2, 0.45, 0.95, 1) if k % 2 == 1 else (0.95, 0.5, 0.15, 1))
        ob.show_in_front = True
        _keys(ob, "location", frames, G[:, k])
        sc_ = np.repeat(vm[:, k:k + 1].astype(float), 3, axis=1)
        _keys(ob, "scale", frames, sc_, constant=True)
    if mode != "side":
        return
    segs = [(11, 12), (23, 24), (11, 23), (12, 24), (11, 13), (13, 15), (12, 14), (14, 16),
            (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (7, 8), (0, 7), (0, 8)]
    r = 0.008
    verts, faces, owner = [], [], []
    P0 = G[0]
    for a_, b_ in segs:
        for ax in (np.array([r, 0, 0]), np.array([0, r, 0]), np.array([0, 0, r])):
            k0 = len(verts)
            verts += [P0[a_] - ax, P0[a_] + ax, P0[b_] + ax, P0[b_] - ax]
            owner += [a_, a_, b_, b_]
            faces.append((k0, k0 + 1, k0 + 2, k0 + 3))
    me2 = bpy.data.meshes.new("ghost_sticks")
    me2.from_pydata([tuple(v) for v in verts], [], faces)
    ob2 = bpy.data.objects.new("ghost_sticks", me2)
    ob2.color = (0.85, 0.85, 0.85, 1)
    coll.objects.link(ob2)
    for k in sorted(set(owner)):
        mod = ob2.modifiers.new("hook_%02d" % k, "HOOK")
        mod.object = bpy.data.objects["ghost_%02d" % k]
        mod.vertex_indices_set([i for i, o in enumerate(owner) if o == k])
        mod.matrix_inverse = Matrix.Translation(tuple(P0[k])).inverted()
        mod.center = tuple(P0[k])


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default="/home/user/data/mocap/retargeted.blend")
    p.add_argument("--out", default="/home/user/data/mocap/preview.mp4")
    p.add_argument("--rig", default="rig")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--res", default="852x480", help="WxH (with --calib only W is used)")
    p.add_argument("--view", default="front34", choices=["front34", "track34", "side", "front", "scene"],
                   help="scene = use the .blend's own active camera (e.g. two_characters_take1.blend)")
    p.add_argument("--calib", default=None, help="rig.json:camB - render from that calibrated camera")
    p.add_argument("--ghost", default=None, help="npz: show the processed mocap joints")
    p.add_argument("--ghost-mode", default="overlay", choices=["overlay", "side"])
    p.add_argument("--overlay", action="store_true", help="transparent background, no floor (RGBA)")
    p.add_argument("--no-stamp", action="store_true")
    p.add_argument("--png-dir", default=None)
    p.add_argument("--keep-png", action="store_true")
    a = p.parse_args(argv)

    bpy.ops.wm.open_mainfile(filepath=a.blend)
    sc = bpy.context.scene
    rig = bpy.data.objects[a.rig]
    height = float(rig.get("mocap_height", 1.75))
    w, h = (int(x) for x in a.res.split("x"))
    if a.calib:
        _, (w, h) = setup_calib_camera(a.calib, w)
    elif a.view == "scene":
        if sc.camera is None:
            raise SystemExit("--view scene: the .blend has no active camera")
    else:
        setup_camera(rig, a.view, height)
    if a.ghost:
        add_ghost(a.ghost, rig, a.ghost_mode)
    rig.hide_render = True
    for ob in bpy.data.objects:
        if ob.type == "ARMATURE" or ob.name.startswith("WGT"):
            ob.hide_render = True
        if a.overlay and ob.name.startswith("floor_"):
            ob.hide_render = True
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "OBJECT"
    sh.show_cavity = False
    sh.show_shadows = not a.overlay
    sh.shadow_intensity = 0.6
    sc.display.shadow_focus = 0.2
    sc.display.light_direction = (0.35, -0.45, 0.82)
    sh.background_type = "VIEWPORT"
    sh.background_color = (0.08, 0.09, 0.11)
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = a.overlay
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA" if a.overlay else "RGB"
    sc.render.use_stamp = not a.no_stamp
    for k in ("date", "time", "render_time", "memory", "camera", "lens", "scene", "filename", "hostname"):
        setattr(sc.render, "use_stamp_" + k, False)
    sc.render.use_stamp_frame = True
    sc.render.stamp_font_size = max(12, h // 30)
    start = a.start if a.start is not None else sc.frame_start
    end = a.end if a.end is not None else sc.frame_end
    png_dir = a.png_dir or tempfile.mkdtemp(prefix="mocap_preview_")
    os.makedirs(png_dir, exist_ok=True)
    n = 0
    for f in range(start, end + 1, a.step):
        sc.frame_set(f)
        sc.render.filepath = os.path.join(png_dir, "f_%06d.png" % f)
        bpy.ops.render.render(write_still=True)
        os.replace(os.path.join(png_dir, "f_%06d.png" % f), os.path.join(png_dir, "seq_%06d.png" % n))
        n += 1
    fps = sc.render.fps / a.step
    if a.out:
        if a.overlay:   # keep alpha: PNG-in-MOV (qtrle) for compositing
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "%g" % fps,
                   "-i", os.path.join(png_dir, "seq_%06d.png"), "-c:v", "qtrle", a.out]
        else:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "%g" % fps,
                   "-i", os.path.join(png_dir, "seq_%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-crf", "20", a.out]
        subprocess.run(cmd, check=True)
        print("wrote", a.out, n, "frames @", fps, "fps")
    if not a.keep_png and not a.png_dir:
        shutil.rmtree(png_dir, ignore_errors=True)
    else:
        print("pngs in", png_dir)


if __name__ == "__main__":
    main(sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1:])
