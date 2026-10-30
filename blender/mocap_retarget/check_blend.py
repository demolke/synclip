"""Re-open a retargeted .blend and check it is complete (exit code 0 = OK).

    /opt/bpyenv/bin/python check_blend.py --blend /home/user/data/mocap/deliver/two_characters_take1.blend \
        [--rebake rig_stocky]

* opens the file with bpy.ops.wm.open_mainfile and Python auto-run allowed (as a trusted file
  in the GUI), so the embedded text module "mocap_retarget.py" (Register on) runs;
* every Rigify rig: its action, keyed controls, frame range, IK_FK switches per control_mode
  (and the IK keys when control_mode is not FK), its own panel
  settings (Object.mocap_retarget), the mocap source (file or embedded text: npz);
* the Mocap Retarget panel / operators are registered by the embedded module (not by
  importing retarget.py from disk);
* --rebake RIG: runs the panel's Retarget / Re-bake operator on that rig from its stored
  settings (the file is not saved) and checks the new action matches the stored one.
"""
import argparse
import json
import sys

import bpy
import numpy as np
from bpy_extras import anim_utils


def fcurves(rig):
    ad = rig.animation_data
    cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    return list(cb.fcurves) if cb else []


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default="/home/user/data/mocap/deliver/two_characters_take1.blend")
    p.add_argument("--rebake", default=None, help="rig name: re-bake it with the panel operator")
    a = p.parse_args(argv)
    bpy.context.preferences.filepaths.use_scripts_auto_execute = True   # "trusted" file
    bpy.ops.wm.open_mainfile(filepath=a.blend, load_ui=True)
    sc = bpy.context.scene
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    print("file %s: frames %d-%d @ %d fps, camera %s" % (a.blend, sc.frame_start, sc.frame_end, sc.render.fps,
                                                       sc.camera.name if sc.camera else None))
    txt = bpy.data.texts.get("mocap_retarget.py")
    check(txt is not None and txt.use_module, "text module mocap_retarget.py embedded with Register on")
    check(hasattr(bpy.types, "VIEW3D_PT_mocap_retarget") and hasattr(bpy.ops.mocap_retarget, "bake")
          and hasattr(bpy.types.Object, "mocap_retarget"), "Mocap Retarget panel + operators registered on load")
    rigs = [o for o in bpy.data.objects if o.type == "ARMATURE" and "torso" in o.data.bones and "thigh_fk.L" in o.data.bones]
    check(len(rigs) >= 1, "Rigify rigs: %s" % [r.name for r in rigs])
    actions = set()
    for rig in rigs:
        act = rig.animation_data.action if rig.animation_data else None
        check(act is not None, "%s: action %s" % (rig.name, act.name if act else None))
        if act is None:
            continue
        actions.add(act.name)
        fcs = fcurves(rig)
        n_keys = {len(fc.keyframe_points) for fc in fcs}
        rng = act.frame_range
        check(len(fcs) >= 95 and rng[0] <= sc.frame_start and rng[1] >= sc.frame_end,
              "%s: %d f-curves, keys/curve %s, action range %d-%d" % (rig.name, len(fcs), sorted(n_keys), rng[0], rng[1]))
        prm_ = json.loads(act["mocap_params"])
        mode = json.loads(act["mocap_info"]).get("control_mode", prm_.get("control_mode", "FK"))
        want = {"arm": 1.0, "leg": 1.0} if mode in ("FK", "BOTH") else {"arm": 1.0, "leg": 0.0}
        sw = {n: (k, rig.pose.bones[n]["IK_FK"]) for k, n in
              [(k, p_ + "." + s_) for s_ in "LR" for k, p_ in (("arm", "upper_arm_parent"), ("leg", "thigh_parent"))]
              if n in rig.pose.bones}
        check(len(sw) == 4 and all(v == want[k] for k, v in sw.values()),
              "%s: control_mode %s, Rigify IK_FK %s" % (rig.name, mode, {n: v for n, (k, v) in sw.items()}))
        if mode != "FK":
            paths = {fc.data_path for fc in fcs}
            ik = ["hand_ik", "foot_ik", "toe_ik", "upper_arm_ik_target", "thigh_ik_target"]
            check(all(any('"%s.%s"' % (n, s_) in p_ for p_ in paths) for n in ik for s_ in "LR"),
                  "%s: IK controls keyed (%s)" % (rig.name, ", ".join(ik)))
        s = rig.mocap_retarget
        src = s.npz_path
        emb = src.startswith("text:") and src[5:] in bpy.data.texts
        check(bool(src), "%s: own settings (npz %s%s, mocap_space %s, action_name %s)" % (
            rig.name, src, " embedded" if emb else "", s.mocap_space, s.action_name))
        prm = json.loads(act["mocap_params"])
        check(prm.get("mocap_space", "WORLD") == s.mocap_space and prm["action_name"] == s.action_name,
              "%s: panel settings match the baked action's parameters" % rig.name)
    check(len(actions) == len(rigs), "one action per rig: %s" % sorted(actions))
    if a.rebake:
        rig = bpy.data.objects[a.rebake]
        old = rig.animation_data.action
        fc_old = {(fc.data_path, fc.array_index): np.array([k.co[1] for k in fc.keyframe_points])
                  for fc in fcurves(rig)}
        old.name = old.name + "_before"
        for o in bpy.data.objects:
            o.select_set(o is rig)
        bpy.context.view_layer.objects.active = rig
        win = bpy.context.window_manager.windows[0] if bpy.context.window_manager.windows else None
        area = next((ar for ar in win.screen.areas if ar.type == "VIEW_3D"), None) if win else None
        if area is not None:
            with bpy.context.temp_override(window=win, area=area, active_object=rig, object=rig):
                res = bpy.ops.mocap_retarget.bake()
        else:
            with bpy.context.temp_override(active_object=rig, object=rig):
                res = bpy.ops.mocap_retarget.bake()
        new = rig.animation_data.action
        fc_new = {(fc.data_path, fc.array_index): np.array([k.co[1] for k in fc.keyframe_points])
                  for fc in fcurves(rig)}
        diff = max(float(np.abs(fc_new[k] - v).max()) for k, v in fc_old.items())
        check(res == {"FINISHED"} and new is not old and diff < 1e-4,
              "%s: panel Re-bake -> %s, max key difference vs stored %.2e" % (rig.name, new.name, diff))
    print("CHECK:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1:]))
