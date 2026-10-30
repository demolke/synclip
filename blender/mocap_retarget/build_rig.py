"""Build a Rigify humanoid target rig with low-poly proxy meshes.

    /opt/bpyenv/bin/python build_rig.py --out /home/user/data/mocap/target_rig.blend \
        [--height 1.75] [--leg-scale 1.0] [--arm-scale 1.0] [--keep-face]
        [--preset default|stocky] [--spine S --neck S --head S --upper-arm S --forearm S
         --hand S --thigh S --shin S --foot S --shoulder-width S --hip-width S]
        [--girth torso=1.2,head=1.3,arm=1.1,leg=1.2,hand=1,foot=1] [--palette default|stocky]
        [--rig-name rig]

* Human metarig -> (optional proportion changes) -> scaled to --height ->
  Rigify generate.  The face is removed by default (fewer bones, faster).
* Proportions: every bone vector is scaled by its segment factor and the children
  follow (hierarchically), so --thigh 0.6 shortens only the thighs and moves the rest
  of the leg up.  --shoulder-width / --hip-width scale the lateral (X) offset of the
  arm / leg chains.  --leg-scale / --arm-scale (old options) multiply the
  thigh/shin/foot and upper-arm/forearm/hand factors.  Finally everything is scaled
  uniformly so the top of the head is at --height.
* --preset stocky: a ~1.2 m cartoon character (big torso and head, short legs, long
  arms, wide shoulders and hips, chunky proxies, purple/teal colours).  Explicit
  options override the preset's values.
* build_character() builds into the current file (several characters per scene,
  see make_two_characters.py); build() is the one-character file entry point.
* Mocap landmark positions on the target (ears / eyes / nose / heel ...) are
  stored as the custom property ``mocap_landmarks`` (JSON, armature space) on
  the generated rig, so retarget.py can build matching rest frames.
* Every DEF bone gets a flat box proxy (bone-parented) with a small "fin" on
  its front/up side so twist is visible in a render.  Left side = blue,
  right side = orange, centre = grey.  A checker floor is added.
"""
import argparse
import json
import sys

import bpy
import addon_utils
from mathutils import Vector, Matrix


def enable_rigify():
    mod = addon_utils.enable("rigify", default_set=True)
    if mod is None:  # Blender may ship it as an extension in future builds
        for m in addon_utils.modules():
            if m.__name__.endswith("rigify"):
                addon_utils.enable(m.__name__, default_set=True)
                break
    if not hasattr(bpy.types, "RIGIFY_OT_generate") and not hasattr(bpy.ops.pose, "rigify_generate"):
        raise RuntimeError("Rigify could not be enabled")


# ---------------------------------------------------------------- metarig edit
SEGMENTS = ("spine", "neck", "head", "upper_arm", "forearm", "hand", "thigh", "shin", "foot")


def _segment_of(name):
    """Proportion segment a metarig bone starts (None = inherit the parent's factor)."""
    base = name.split(".")[0]
    if name in ("spine", "spine.001", "spine.002", "spine.003"):
        return "spine"
    if name in ("spine.004", "spine.005"):
        return "neck"
    if name == "spine.006":
        return "head"
    if base in ("upper_arm", "forearm", "hand", "thigh", "shin", "foot"):
        return base
    if base == "shoulder":
        return "clavicle"
    return None


def edit_metarig(meta, height, leg_scale, arm_scale, keep_face, seg=None, shoulder_width=1.0,
                 hip_width=1.0):
    """Returns landmark dict (armature space, after all edits).

    seg: {segment: factor} for SEGMENTS (missing = 1).  Each bone's vector (head->tail)
    is scaled by its segment factor (children of a bone that starts no segment inherit
    the parent's), every child head keeps its offset from the parent head scaled the
    same way, so the chain stays connected and only that segment changes length."""
    bpy.context.view_layer.objects.active = meta
    bpy.ops.object.mode_set(mode="EDIT")
    eb = meta.data.edit_bones
    # Work on a snapshot: with connected bones, moving a child's head moves the
    # parent's tail, so positions must be computed first and assigned afterwards.
    pos = {b.name: [b.head.copy(), b.tail.copy()] for b in eb}
    parent = {b.name: (b.parent.name if b.parent else None) for b in eb}
    f = {k: 1.0 for k in SEGMENTS}
    f.update(seg or {})
    f["clavicle"] = 1.0
    for k in ("thigh", "shin", "foot"):
        f[k] *= leg_scale
    for k in ("upper_arm", "forearm", "hand"):
        f[k] *= arm_scale
    factor = {}
    new = {}

    def xs(v, sx):
        return Vector((v.x * sx, v.y, v.z))

    def place(n):
        if n in new:
            return
        p = parent[n]
        sg = _segment_of(n)
        if p is not None:
            place(p)
        factor[n] = f[sg] if sg is not None else (factor[p] if p is not None else 1.0)
        h0, t0 = pos[n]
        if p is None:
            h = h0.copy()
        else:
            ph0 = pos[p][0]
            r = (h0 - ph0) * factor[p]
            if n.startswith("upper_arm."):
                r = xs(r, shoulder_width)
            if n.startswith("thigh."):
                r = xs(r, hip_width)
            h = new[p][0] + r
        v = (t0 - h0) * factor[n]
        if n.startswith("shoulder."):
            v = xs(v, shoulder_width)
        if n.startswith("pelvis."):
            v = xs(v, hip_width)
        new[n] = [h, h + v]

    for n in pos:
        place(n)
    pos = new
    zmin = min(min(h.z, t.z) for h, t in pos.values())
    ztop = pos["spine.006"][1].z - zmin
    s = height / ztop
    for n, ht in pos.items():
        pos[n] = [Vector((v.x, v.y, v.z - zmin)) * s for v in ht]
    for _ in range(2):  # twice: connected neighbours may have been dragged on the first pass
        for b in eb:
            b.head, b.tail = pos[b.name]
    eb = meta.data.edit_bones

    def mid(*pts):
        return sum((Vector(p) for p in pts), Vector()) / len(pts)

    lm = {}
    if "face" in eb:
        lm["nose"] = eb["nose.001"].tail.copy()
        for side, mp in (("L", "left"), ("R", "right")):
            lm[mp + "_eye"] = eb["eye." + side].head.copy()
            lm[mp + "_ear"] = mid(eb["ear." + side].head, eb["ear." + side].tail)
            lm[mp + "_mouth"] = eb["lip.T." + side + ".001"].tail.copy() if ("lip.T." + side + ".001") in eb else None
    for side, mp in (("L", "left"), ("R", "right")):
        lm[mp + "_heel"] = mid(eb["heel.02." + side].head, eb["heel.02." + side].tail)
        lm[mp + "_index"] = eb["f_index.01." + side].tail.copy()
        lm[mp + "_pinky"] = eb["f_pinky.01." + side].tail.copy()
        lm[mp + "_thumb"] = eb["thumb.02." + side].tail.copy()
    if not keep_face and "face" in eb:
        face = eb["face"]
        kill = [face] + list(face.children_recursive)
        for b in kill:
            eb.remove(b)
    bpy.ops.object.mode_set(mode="OBJECT")
    return {k: list(v) for k, v in lm.items() if v is not None}


# ---------------------------------------------------------------- proxies
def _box(verts, faces, c, side, front, a, l0, l1, w, d):
    """Append a box spanning [l0,l1] along unit axis a, width w along side, depth d along front."""
    base = len(verts)
    for l in (l0, l1):
        for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(c + a * l + side * (sx * w / 2) + front * (sz * d / 2))
    faces += [(base + i for i in f) for f in
              ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))]


def _girth_group(n):
    for pre, g in (("thigh", "leg"), ("shin", "leg"), ("upper_arm", "arm"), ("forearm", "arm"),
                   ("hand", "hand"), ("foot", "foot"), ("toe", "foot"), ("spine.006", "head"),
                   ("spine.004", "neck"), ("spine.005", "neck"), ("spine", "torso"), ("shoulder", "torso"),
                   ("pelvis", "torso"), ("breast", "torso")):
        if n.startswith(pre):
            return g
    return None


def bone_dims(name, length, girth=None):
    """Proxy box (width, depth) in metres; girth = {group: factor} (torso, head, neck,
    arm, leg, hand, foot) thickens the boxes of a stocky character."""
    w, d = _bone_dims(name, length)
    g = (girth or {}).get(_girth_group(name[4:]), 1.0)
    return w * g, d * g


def _bone_dims(name, length):
    n = name[4:]
    if n.startswith(("thigh",)):
        return 0.14, 0.12
    if n.startswith("shin"):
        return 0.10, 0.09
    if n.startswith("upper_arm"):
        return 0.085, 0.075
    if n.startswith("forearm"):
        return 0.075, 0.055
    if n.startswith("hand"):
        return 0.08, 0.03
    if n.startswith("foot"):
        return 0.08, 0.06
    if n.startswith("toe"):
        return 0.08, 0.035
    if n.startswith("spine.006"):
        return 0.16, 0.19
    if n.startswith(("spine.004", "spine.005")):
        return 0.08, 0.08
    if n.startswith("spine"):
        return 0.30, 0.18
    if n.startswith("shoulder"):
        return 0.07, 0.07
    if n.startswith("pelvis"):
        return 0.06, 0.06
    if n.startswith("breast"):
        return 0.08, 0.06
    return max(0.012, min(0.03, length * 0.4)), max(0.01, min(0.02, length * 0.3))


PALETTES = {
    # left, right, centre, fin
    "default": ((0.20, 0.45, 0.95, 1.0), (0.95, 0.50, 0.15, 1.0), (0.75, 0.75, 0.72, 1.0), (0.15, 0.85, 0.30, 1.0)),
    "stocky": ((0.62, 0.30, 0.90, 1.0), (0.10, 0.78, 0.72, 1.0), (0.88, 0.32, 0.36, 1.0), (1.00, 0.90, 0.20, 1.0)),
}


def make_proxies(rig, coll, girth=None, palette="default", prefix="PRX_"):
    left, right, centre, fin_col = PALETTES[palette] if isinstance(palette, str) else palette
    made = []
    for b in rig.data.bones:
        if not b.use_deform or not b.name.startswith("DEF-"):
            continue
        M = b.matrix_local
        head = M.translation.copy()
        L = b.length
        a = (b.tail_local - b.head_local).normalized()
        if abs(a.z) > 0.7:          # vertical bone: front = -Y
            front = Vector((0, -1, 0))
        else:                       # horizontal bone: "front" = up
            front = Vector((0, 0, 1))
        front = (front - a * front.dot(a)).normalized()
        side = a.cross(front).normalized()
        w, d = bone_dims(b.name, L, girth)
        verts, faces = [], []
        gap = min(0.01, L * 0.05)
        _box(verts, faces, head, side, front, a, gap, L - gap, w, d)
        fverts, ffaces = [], []
        if b.name[4:].startswith(("thigh", "shin", "upper_arm", "forearm", "foot", "hand", "spine.006")):
            # fin on the front / top side to make twist visible
            fw = w * 0.25
            fd = d * 0.6
            _box(fverts, ffaces, head + front * (d / 2 + fd / 2 - 0.002), side, front, a,
                 L * 0.3, L * 0.8, fw, fd)
        if b.name == "DEF-spine.006":  # nose
            ng = (girth or {}).get("head", 1.0)
            _box(fverts, ffaces, head + Vector((0, -0.11 * ng, 0)), side, front, a, L * 0.35, L * 0.5,
                 0.03 * ng, 0.05 * ng)
        to_bone = Matrix.Translation((0, -L, 0)) @ M.inverted()
        for vs, fs, col, suffix in ((verts, faces, None, ""), (fverts, ffaces, fin_col, "_fin")):
            if not vs:
                continue
            me = bpy.data.meshes.new(prefix + b.name + suffix)
            me.from_pydata([to_bone @ v for v in vs], [], [tuple(f) for f in fs])
            me.update()
            ob = bpy.data.objects.new(prefix + b.name + suffix, me)
            coll.objects.link(ob)
            ob.parent = rig
            ob.parent_type = "BONE"
            ob.parent_bone = b.name
            ob.matrix_parent_inverse = Matrix.Identity(4)
            ob.matrix_basis = Matrix.Identity(4)
            if col is None:
                col = left if b.name.endswith(".L") or ".L." in b.name else (
                    right if b.name.endswith(".R") or ".R." in b.name else centre)
            ob.color = col
            made.append(ob)
    return made


def make_floor(coll, size=8.0, tile=0.5):
    n = int(size / tile)
    for parity, col in ((0, (0.55, 0.55, 0.55, 1)), (1, (0.35, 0.35, 0.37, 1))):
        verts, faces = [], []
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 != parity:
                    continue
                x0 = -size / 2 + i * tile
                y0 = -size / 2 + j * tile
                k = len(verts)
                verts += [(x0, y0, 0), (x0 + tile, y0, 0), (x0 + tile, y0 + tile, 0), (x0, y0 + tile, 0)]
                faces.append((k, k + 1, k + 2, k + 3))
        me = bpy.data.meshes.new("floor_%d" % parity)
        me.from_pydata(verts, [], faces)
        ob = bpy.data.objects.new("floor_%d" % parity, me)
        ob.color = col
        coll.objects.link(ob)


PRESETS = {
    "default": {},
    # short, stocky cartoon character: big torso and head, short legs, long arms, wide
    # shoulders and hips (factors are relative to Rigify's human metarig, before the
    # uniform scale to --height)
    "stocky": dict(height=1.2, seg=dict(spine=1.0, neck=0.6, head=1.7, upper_arm=1.05, forearm=1.05,
                                        hand=1.15, thigh=0.6, shin=0.6, foot=1.1),
                   shoulder_width=1.35, hip_width=1.3, palette="stocky",
                   girth=dict(torso=1.2, head=1.35, neck=1.4, arm=1.15, leg=1.3, hand=1.2, foot=1.25)),
}


def enable_workbench_look(sc):
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.color_type = "OBJECT"
    sc.display.shading.light = "STUDIO"
    for wgt in bpy.data.collections:
        if wgt.name.startswith("WGTS"):
            wgt.hide_render = True
            wgt.hide_viewport = True


def build_character(name="rig", height=1.75, leg_scale=1.0, arm_scale=1.0, keep_face=False, seg=None,
                    shoulder_width=1.0, hip_width=1.0, girth=None, palette="default", location=(0, 0, 0),
                    collection=None):
    """Build one Rigify character into the current file.  Returns the generated rig
    (object named `name`; metarig `name`_metarig; proxies in `collection`)."""
    enable_rigify()
    sc = bpy.context.scene
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.armature_human_metarig_add()
    meta = bpy.context.active_object
    meta.name = name + "_metarig" if name != "rig" else "metarig"
    lm = edit_metarig(meta, height, leg_scale, arm_scale, keep_face, seg, shoulder_width, hip_width)
    bpy.context.view_layer.objects.active = meta
    meta.select_set(True)
    meta.data.rigify_target_rig = None
    bpy.ops.pose.rigify_generate()
    rig = meta.data.rigify_target_rig or next(o for o in bpy.data.objects if o.type == "ARMATURE" and o != meta)
    rig.name = name
    rig.data.name = name
    rig["mocap_landmarks"] = json.dumps(lm)
    rig["mocap_height"] = height
    rig["mocap_proportions"] = json.dumps(dict(leg_scale=leg_scale, arm_scale=arm_scale, seg=seg or {},
                                              shoulder_width=shoulder_width, hip_width=hip_width,
                                              girth=girth or {}, palette=palette))
    # FK mode for the limbs, no neck/head follow mixing (clean FK hierarchy)
    for pb in rig.pose.bones:
        if "IK_FK" in pb.keys():
            pb["IK_FK"] = 1.0
    meta.hide_set(True)
    meta.hide_render = True
    if collection is None:
        collection = bpy.data.collections.new("proxies" if name == "rig" else name + "_proxies")
        sc.collection.children.link(collection)
    made = make_proxies(rig, collection, girth, palette, "PRX_" if name == "rig" else "PRX_%s_" % name)
    if any(location):
        for ob in (rig, meta):
            ob.location = Vector(location)
    bpy.context.view_layer.update()
    print("built %s: height=%.2f proxies=%d" % (name, height, len(made)))
    return rig


def build(out, height=1.75, leg_scale=1.0, arm_scale=1.0, keep_face=False, **kw):
    """One character + checker floor in a new file (the original entry point)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig = build_character(kw.pop("name", "rig"), height, leg_scale, arm_scale, keep_face, **kw)
    coll = bpy.data.collections.get("proxies") or next(c for c in bpy.data.collections if c.name.endswith("proxies"))
    make_floor(coll)
    enable_workbench_look(bpy.context.scene)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print("saved %s: rig=%s height=%.2f" % (out, rig.name, height))
    return rig


def preset_kwargs(preset="default", **over):
    """Merge a preset with explicit overrides (None = keep the preset's value)."""
    kw = json.loads(json.dumps(PRESETS[preset]))  # deep copy
    kw.setdefault("height", 1.75)
    kw.setdefault("seg", {})
    for k, v in over.items():
        if v is None:
            continue
        if k == "seg":
            kw["seg"].update(v)
        elif k == "girth":
            kw.setdefault("girth", {}).update(v)
        else:
            kw[k] = v
    return kw


def _kv(s):
    return {k: float(v) for k, v in (it.split("=") for it in s.split(",") if it)}


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/mocap.blend")
    p.add_argument("--preset", default="default", choices=sorted(PRESETS))
    p.add_argument("--rig-name", default="rig", help="name of the generated rig object")
    p.add_argument("--height", type=float, default=None, help="m, top of the head (default 1.75 / preset)")
    p.add_argument("--leg-scale", type=float, default=1.0, help="multiplies thigh, shin and foot")
    p.add_argument("--arm-scale", type=float, default=1.0, help="multiplies upper arm, forearm and hand")
    for sgm in SEGMENTS:
        p.add_argument("--" + sgm.replace("_", "-"), type=float, default=None, dest="seg_" + sgm,
                       help="length factor of the %s segment" % sgm)
    p.add_argument("--shoulder-width", type=float, default=None)
    p.add_argument("--hip-width", type=float, default=None)
    p.add_argument("--girth", type=_kv, default=None,
                   help="proxy thickness: torso=1.2,head=1.3,neck=1,arm=1.1,leg=1.2,hand=1,foot=1")
    p.add_argument("--palette", default=None, choices=sorted(PALETTES))
    p.add_argument("--keep-face", action="store_true")
    a = p.parse_args([])
    seg = {sgm: getattr(a, "seg_" + sgm) for sgm in SEGMENTS if getattr(a, "seg_" + sgm) is not None}
    kw = preset_kwargs(a.preset, height=a.height, seg=seg, shoulder_width=a.shoulder_width,
                       hip_width=a.hip_width, girth=a.girth, palette=a.palette)
    height = kw.pop("height")
    build(a.out, height, a.leg_scale, a.arm_scale, a.keep_face, name=a.rig_name, **kw)


if __name__ == "__main__":
    main(sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1:])
