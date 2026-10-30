"""Export one retargeted Rigify character (+ its baked action) as FBX or glTF for a game engine.

    /opt/bpyenv/bin/python export_anim.py --blend retargeted.blend --rig rig --out take1.fbx
    /opt/bpyenv/bin/python export_anim.py --blend two_characters_take1.blend --rig rig_stocky \
        --out stocky_take1.glb [--all-bones] [--no-meshes] [--at-origin]

* Only the rig's deform (DEF-) bones are exported by default: that is the game skeleton;
  Rigify's controls / mechanism bones are not needed once the motion is baked (the exporters
  sample the evaluated pose every frame, so constraints and the FK controls are baked in).
* The meshes bone-parented to that rig (the proxy boxes) are exported with it unless
  --no-meshes.  Format from the extension: .fbx, .glb or .gltf.
* --at-origin: temporarily move the rig object to the world origin (characters placed side
  by side in one scene keep their offset otherwise).
* Root motion is whatever the action has: bake with --set root_motion=IN_PLACE (retarget.py)
  for an in-place clip (the torso keeps only its height), FULL for root motion.
"""
import argparse
import os
import sys

import bpy


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--rig", default="rig")
    p.add_argument("--out", required=True, help=".fbx, .glb or .gltf")
    p.add_argument("--all-bones", action="store_true", help="also export Rigify's control / mechanism bones")
    p.add_argument("--no-meshes", action="store_true")
    p.add_argument("--at-origin", action="store_true")
    a = p.parse_args(argv)
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    rig = bpy.data.objects[a.rig]
    if rig.animation_data is None or rig.animation_data.action is None:
        raise SystemExit("%s has no action - retarget it first" % rig.name)
    sc = bpy.context.scene
    meshes = [] if a.no_meshes else [o for o in bpy.data.objects if o.type == "MESH" and o.parent is rig]
    if a.at_origin:
        rig.location = (0, 0, 0)
    for o in bpy.data.objects:
        o.hide_set(False)
        o.select_set(False)
    for o in [rig] + meshes:
        o.hide_viewport = False
        o.select_set(True)
    bpy.context.view_layer.objects.active = rig
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ext = os.path.splitext(a.out)[1].lower()
    if ext == ".fbx":
        bpy.ops.export_scene.fbx(filepath=a.out, use_selection=True, object_types={"ARMATURE", "MESH"},
                                 use_armature_deform_only=not a.all_bones, add_leaf_bones=False,
                                 bake_anim=True, bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                                 bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0,
                                 bake_anim_step=1.0)
    elif ext in (".glb", ".gltf"):
        bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB" if ext == ".glb" else "GLTF_SEPARATE",
                                  use_selection=True, export_def_bones=not a.all_bones,
                                  export_animations=True, export_animation_mode="ACTIVE_ACTIONS",
                                  export_force_sampling=True, export_frame_range=True)
    else:
        raise SystemExit("unknown format %r (use .fbx, .glb or .gltf)" % ext)
    print("exported %s: rig %s, action %s, frames %d-%d, %d meshes, %s bones -> %s (%.1f MB)" % (
        ext, rig.name, rig.animation_data.action.name, sc.frame_start, sc.frame_end, len(meshes),
        "all" if a.all_bones else "deform", a.out, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main(sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1:])
