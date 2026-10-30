"""Quantitative validation of a retargeted Rigify rig (evaluates the depsgraph per frame).

    /opt/bpyenv/bin/python validate.py --blend retargeted.blend [--npz take.npz] \
        [--json report.json] [--start N --end M --step K]

Checks
  a) stretch    : every DEF bone's posed length vs rest (max relative deviation)
  b) direction  : angle between retargeted DEF segments and the (processed) mocap
                  segments: limbs, feet, hands, trunk, shoulder/hip lines, head
  c) twist      : forearm-vs-upper-arm, upper-arm-vs-chest, chest-vs-pelvis,
                  shin-vs-thigh, wrist (hand-vs-forearm) twist relative to rest (flag > 100 deg)
  d) hinge      : elbow / knee out-of-plane angle and hyper-extension
  e) flips      : frame-to-frame angular velocity spikes of every DEF bone
  f) feet       : floor penetration, sliding speed during contacts
  a)-f) run on the file as saved (its IK_FK switches); control_mode BOTH also with every
  limb in IK.  Files with IK keys (control_mode LEGS_IK_ARMS_FK / BOTH) add:
  g) IK stretch : each limb in IK with its hand_ik / foot_ik pushed 15% out of reach (action
                  unassigned for the test): its DEF bones must not stretch (< 1e-3)
  h) FK == IK   : all limbs FK vs all limbs IK: DEF limb directions (arms / legs: mean < 0.5,
                  p95 < 2 deg) and wrist / ankle / toe-tip positions (< 1 mm)
Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retarget as mr  # noqa: E402

THRESH = {
    "stretch_max_rel": 1e-3,
    "dir_limb_mean_deg": 6.0,
    "dir_limb_p95_deg": 15.0,
    "twist_flag_deg": 100.0,
    "hinge_offplane_max_deg": 2.0,
    "hyperextension_max_deg": 10.0,
    "flip_deg_per_frame": 45.0,
    "penetration_max_m": 0.015,
    "slide_p95_m_per_frame": 0.01,
    # IK modes (control_mode != FK)
    "ik_stretch_max_rel": 1e-3,          # g) limb stretch with the IK end control 15% out of reach
    "match_dir_mean_deg": 0.5,           # h) DEF limb directions, all-FK vs all-IK evaluation (per arms / legs)
    "match_dir_p95_deg": 2.0,            #    (differences: hyper-extended frames, which IK with a pole straightens)
    "match_end_max_mm": 1.0,             #    wrist / ankle / toe-tip position
}
SWITCHES = [L["prop"] + "." + s for s, _ in mr.SIDES for L in mr.IK_LIMBS.values()]
LIMB_KIND = {L["prop"] + "." + s: k for s, _ in mr.SIDES for k, L in mr.IK_LIMBS.items()}


def set_switch(rig, switch):
    """Set IK_FK per limb and make the depsgraph re-evaluate the switch drivers."""
    for n, v in switch.items():
        rig.pose.bones[n]["IK_FK"] = float(v)
    rig.update_tag()
    bpy.context.view_layer.update()


def sample_limbs(rig, frames, switch):
    """DEF limb head/tail and end points per frame with the given switches."""
    set_switch(rig, switch)
    sc = bpy.context.scene
    pbs = rig.pose.bones
    names = [b.name for b in rig.data.bones if b.name.startswith("DEF-") and
             b.name[4:].split(".")[0] in ("upper_arm", "forearm", "hand", "thigh", "shin", "foot", "toe")]
    ends = ["ORG-hand.L", "ORG-hand.R", "ORG-foot.L", "ORG-foot.R"]
    H, T, E = [], [], []
    for f in frames:
        sc.frame_set(f)
        H.append([np.array(pbs[n].head) for n in names])
        T.append([np.array(pbs[n].tail) for n in names])
        E.append([np.array(pbs[n].head) for n in ends] + [np.array(pbs["ORG-toe." + x].tail) for x in "LR"])
    return names, np.array(H), np.array(T), np.array(E)


def fk_ik_match(rig, frames):
    """h) the FK and IK keys describe the same motion: all limbs FK vs all limbs IK."""
    names, H0, T0, E0 = sample_limbs(rig, frames, {n: 1.0 for n in SWITCHES})
    _, H1, T1, E1 = sample_limbs(rig, frames, {n: 0.0 for n in SWITCHES})
    d = ang(T0 - H0, T1 - H1)
    out = {}
    for g, keys in (("arms", ("upper_arm", "forearm", "hand")), ("legs", ("thigh", "shin", "foot", "toe"))):
        idx = [j for j, n in enumerate(names) if n[4:].split(".")[0] in keys]
        out["dir_deg_" + g] = stats(d[:, idx])
    out["end_pos_mm"] = stats(1000 * np.linalg.norm(E0 - E1, axis=-1))
    out["frames_dir_gt_1deg"] = int((d.max(1) > 1).sum())
    out["pass"] = bool(all(out["dir_deg_" + g]["mean"] < THRESH["match_dir_mean_deg"] and
                           out["dir_deg_" + g]["p95"] < THRESH["match_dir_p95_deg"] for g in ("arms", "legs"))
                       and out["end_pos_mm"]["max"] < THRESH["match_end_max_mm"])
    return out


def ik_stretch_test(rig, frame, limbs):
    """g) IK setup + stress test: each limb in IK, its end control pushed 15% beyond reach
    (action unassigned for the test); the limb's DEF bones must not stretch."""
    sc = bpy.context.scene
    pbs, bones = rig.pose.bones, rig.data.bones
    ad = rig.animation_data
    act, slot = ad.action, ad.action_slot
    sc.frame_set(frame)
    saved = {n: pbs[n]["IK_FK"] for n in SWITCHES}
    out = {"settings": {}, "reach_115pct_max_rel": {}}
    ad.action = None
    try:
        for prop in limbs:
            s = prop[-1]
            L = mr.IK_LIMBS[LIMB_KIND[prop]]
            pb = pbs[prop]
            chain = [pbs[n + "." + s].ik_stretch for n in L["chain"]]
            out["settings"][prop] = {"IK_Stretch": float(pb.get("IK_Stretch", -1)), "pole_vector": bool(pb.get("pole_vector")),
                                     "bone_ik_stretch": chain}
            set_switch(rig, {prop: 0.0})
            up = "ORG-upper_arm." + s if LIMB_KIND[prop] == "arm" else "ORG-thigh." + s
            lo = "ORG-forearm." + s if LIMB_KIND[prop] == "arm" else "ORG-shin." + s
            reach = bones[up].length + bones[lo].length
            end = pbs[L["end"] + "." + s]
            M0 = end.matrix.copy()
            root = np.array(pbs[up].head)
            hd = np.array(end.head)
            M = M0.copy()
            M.translation = M0.translation + type(M0.translation)((root + (hd - root) / np.linalg.norm(hd - root) * reach * 1.15) - hd)
            end.matrix = M
            rig.update_tag()
            bpy.context.view_layer.update()
            defs = [b.name for b in bones if b.name.startswith("DEF-") and b.name.endswith(("." + s, "." + s + ".001"))
                    and b.name[4:].split(".")[0] in ("upper_arm", "forearm", "hand", "thigh", "shin", "foot", "toe")]
            rel = {n: abs(np.linalg.norm(np.array(pbs[n].tail) - np.array(pbs[n].head)) / bones[n].length - 1) for n in defs}
            # limb chain (upper / lower segment) vs the end (hand / foot / toe) bones
            lim = [n for n in defs if n[4:].split(".")[0] in ("upper_arm", "forearm", "thigh", "shin")]
            out["reach_115pct_max_rel"][prop] = float(max(rel[n] for n in lim))
            out.setdefault("reach_115pct_end_bones_max_rel", {})[prop] = {n: round(float(v), 4) for n, v in rel.items()
                                                                         if n not in lim and v > 1e-3}
            end.matrix = M0
    finally:
        ad.action = act
        ad.action_slot = slot
        set_switch(rig, saved)
    out["pass"] = bool(all(v < THRESH["ik_stretch_max_rel"] for v in out["reach_115pct_max_rel"].values()))
    return out


def ang(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def rot_angle(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def stats(x):
    x = np.asarray(x, float)
    return {"mean": float(np.mean(x)), "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default="/home/user/data/mocap/retargeted.blend")
    p.add_argument("--npz", default=None, help="defaults to the path stored on the action")
    p.add_argument("--rig", default="rig")
    p.add_argument("--json", default=None)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    a = p.parse_args(argv)

    bpy.ops.wm.open_mainfile(filepath=a.blend)
    sc = bpy.context.scene
    rig = bpy.data.objects[a.rig]
    act = rig.animation_data.action
    prm = mr.Params.from_dict(json.loads(act["mocap_params"]))
    info = json.loads(act["mocap_info"])
    npz = a.npz or act["mocap_npz"]
    mocap = mr.load_mocap(npz)
    J, vmask = mr.preprocess(mocap, prm, return_mask=True)
    observed = vmask[:, mr.CORE].all(1)
    Wi = mr.mocap_to_rig(np.array(rig.matrix_world), prm)
    J = J @ Wi[:3, :3].T + Wi[:3, 3]
    model = mr.build_model(rig)
    start = a.start if a.start is not None else sc.frame_start
    end = a.end if a.end is not None else sc.frame_end
    frames = list(range(start, end + 1, a.step))
    mi = [prm.frame_start + (f - prm.scene_start) for f in frames]

    bones = rig.data.bones
    defs = [b.name for b in bones if b.name.startswith("DEF-")]
    rest_len = {n: bones[n].length for n in defs}
    rest_M = {n: np.array(bones[n].matrix_local) for n in defs}
    pbs = rig.pose.bones
    orgs = [b.name for b in bones if b.name.startswith("ORG-") and b.name[4:].split(".")[0] in
            ("upper_arm", "forearm", "hand", "thigh", "shin", "shoulder", "spine")]
    RM = {n: np.array(bones[n].matrix_local)[:3, :3] for n in defs + orgs}
    HH, TT = {}, {}
    # landmarks attached to DEF bones (for head / hand / foot checks)
    lm = model.lm

    def posed(n, M, p):
        return (M[n] @ np.linalg.inv(rest_M[n]) @ np.append(p, 1.0))[:3]

    def evaluate(label, switch):
        """All per-frame checks with the IK_FK switches set to `switch` ({prop bone: value})."""
        set_switch(rig, switch)
        N = len(frames)
        rec = {k: [] for k in ("stretch",)}
        seg_err = {}
        twist = {}
        hinge_off = {}
        hyper = {}
        angvel = {n: [] for n in defs}
        foot_low = {"L": [], "R": []}
        ankle = {"L": [], "R": []}
        Rprev = {}
        mprev, mw = {}, {}
        for k, f in enumerate(frames):
            sc.frame_set(f)
            P = J[mi[k]]
            M = {n: np.array(pbs[n].matrix) for n in defs}   # armature space
            H = {n: np.array(pbs[n].head) for n in defs}
            T = {n: np.array(pbs[n].tail) for n in defs}
            R = {n: M[n][:3, :3] / np.linalg.norm(M[n][:3, :3], axis=0) for n in defs}
            # a) stretch
            rec["stretch"].append(max(abs(np.linalg.norm(T[n] - H[n]) / rest_len[n] - 1) for n in defs))
            # e) angular velocity
            for n in defs:
                if n in Rprev:
                    angvel[n].append(rot_angle(Rprev[n].T @ R[n]))
                Rprev[n] = R[n]

            for n in orgs:
                Mo = np.array(pbs[n].matrix)
                R[n] = Mo[:3, :3] / np.linalg.norm(Mo[:3, :3], axis=0)
                HH[n] = np.array(pbs[n].head)
                TT[n] = np.array(pbs[n].tail)

            def rel(child, parent):
                Dc = R[child] @ RM[child].T
                Dp = R[parent] @ RM[parent].T
                axis = RM[child][:, 1]
                _, tw = mr.swing_twist(Dp.T @ Dc, axis / np.linalg.norm(axis))
                return abs(math.degrees(tw))

            vk = vmask[mi[k]]
            SEGJ = {"upper_arm": (11, 13), "forearm": (13, 15), "thigh": (23, 25), "shin": (25, 27),
                    "foot": (27, 31), "hand": (15, 17, 19), "trunk": (11, 12, 23, 24),
                    "shoulder_line": (11, 12), "hip_line": (23, 24), "head_ear_line": (7, 8)}

            def seg(name, rig_vec, mc_vec):
                # direction errors only where this segment's joints were actually observed
                # (unobserved segments are filled in rotation space, not from interpolated joints)
                base, _, sd = name.partition(".")
                js = SEGJ[base]
                if sd == "R":
                    js = tuple(i + 1 for i in js)
                if vk[list(js)].all():
                    seg_err.setdefault(name, []).append(float(ang(rig_vec, mc_vec)))
            for s, mp in mr.SIDES:
                j = lambda n: P[mr.MP[mp + "_" + n]]
                seg("upper_arm." + s, T["DEF-upper_arm.%s.001" % s] - H["DEF-upper_arm." + s], j("elbow") - j("shoulder"))
                seg("forearm." + s, T["DEF-forearm.%s.001" % s] - H["DEF-forearm." + s], j("wrist") - j("elbow"))
                seg("thigh." + s, T["DEF-thigh.%s.001" % s] - H["DEF-thigh." + s], j("knee") - j("hip"))
                seg("shin." + s, T["DEF-shin.%s.001" % s] - H["DEF-shin." + s], j("ankle") - j("knee"))
                seg("foot." + s, T["DEF-toe." + s] - H["DEF-foot." + s], j("foot_index") - j("ankle"))
                hm = (posed("DEF-hand." + s, M, lm[mp + "_index"]) + posed("DEF-hand." + s, M, lm[mp + "_pinky"])) / 2
                seg("hand." + s, hm - H["DEF-hand." + s], (j("index") + j("pinky")) / 2 - j("wrist"))
                # c) twist / d) hinge are measured on the ORG bones (= the FK chain).  Rigify's
                # first DEF segment of each limb is anti-twist by design, so it is not a twist ref.
                twist.setdefault("forearm." + s, []).append(rel("ORG-forearm." + s, "ORG-upper_arm." + s))
                twist.setdefault("upper_arm." + s, []).append(rel("ORG-upper_arm." + s, "ORG-shoulder." + s))
                twist.setdefault("shin." + s, []).append(rel("ORG-shin." + s, "ORG-thigh." + s))
                twist.setdefault("thigh." + s, []).append(rel("ORG-thigh." + s, "ORG-spine"))
                twist.setdefault("DEF_forearm_split." + s, []).append(rel("DEF-forearm.%s.001" % s, "DEF-forearm." + s))
                # wrist twist: hand vs forearm (an arm in IK carries the forearm pronation here)
                twist.setdefault("wrist." + s, []).append(rel("ORG-hand." + s, "ORG-forearm." + s))
                for jn, up, lo in (("elbow." + s, "ORG-upper_arm." + s, "ORG-forearm." + s),
                                   ("knee." + s, "ORG-thigh." + s, "ORG-shin." + s)):
                    u0 = np.array(bones[up].tail_local) - np.array(bones[up].head_local)
                    l0 = np.array(bones[lo].tail_local) - np.array(bones[lo].head_local)
                    n0 = np.cross(u0, l0)
                    n0 /= np.linalg.norm(n0)
                    nn = R[up] @ RM[up].T @ n0
                    u = TT[up] - HH[up]
                    l_ = TT[lo] - HH[lo]
                    u /= np.linalg.norm(u)
                    l_ /= np.linalg.norm(l_)
                    off = math.degrees(math.asin(abs(np.clip(np.dot(l_, nn), -1, 1))))
                    flex = math.degrees(math.atan2(np.dot(np.cross(u, l_), nn), np.dot(u, l_)))
                    hinge_off.setdefault(jn, []).append(off)
                    hyper.setdefault(jn, []).append(max(0.0, -flex))
                # f) feet
                pts = [posed("DEF-foot." + s, M, lm[mp + "_heel"]), H["DEF-toe." + s], T["DEF-toe." + s]]
                foot_low[s].append(min(q_[2] for q_ in pts) - model.floor_rest)
                ankle[s].append(np.array(pts))
            mh = (H["DEF-thigh.L"] + H["DEF-thigh.R"]) / 2
            msh = (H["DEF-upper_arm.L"] + H["DEF-upper_arm.R"]) / 2
            seg("trunk", msh - mh, (P[11] + P[12]) / 2 - (P[23] + P[24]) / 2)
            seg("shoulder_line", H["DEF-upper_arm.L"] - H["DEF-upper_arm.R"], P[11] - P[12])
            seg("hip_line", H["DEF-thigh.L"] - H["DEF-thigh.R"], P[23] - P[24])
            el = posed("DEF-spine.006", M, lm["left_ear"])
            er = posed("DEF-spine.006", M, lm["right_ear"])
            seg("head_ear_line", el - er, P[7] - P[8])
            twist.setdefault("spine", []).append(rel("ORG-spine.003", "ORG-spine"))
            # mocap direction vectors per body group (for mocap-relative flip detection)
            mvec = {}
            for s_, mp in mr.SIDES:
                q = lambda n: P[mr.MP[mp + "_" + n]]
                mvec["arm" + s_] = [q("elbow") - q("shoulder")]
                mvec["fore" + s_] = [q("wrist") - q("elbow")]
                mvec["hand" + s_] = [(q("index") + q("pinky")) / 2 - q("wrist"), q("index") - q("pinky")]
                mvec["thigh" + s_] = [q("knee") - q("hip")]
                mvec["shin" + s_] = [q("ankle") - q("knee")]
                mvec["foot" + s_] = [q("foot_index") - q("ankle"), q("ankle") - q("heel")]
            mvec["trunk"] = [(P[11] + P[12]) / 2 - (P[23] + P[24]) / 2, P[11] - P[12], P[23] - P[24]]
            mvec["head"] = [P[7] - P[8], P[0] - (P[7] + P[8]) / 2]
            for g, vs in mvec.items():
                vs = [v / (np.linalg.norm(v) + 1e-12) for v in vs]
                if g in mprev:
                    mw.setdefault(g, []).append(max(float(ang(x, y)) for x, y in zip(vs, mprev[g])))
                mprev[g] = vs

        rep = {"blend": a.blend, "npz": npz, "frames": [frames[0], frames[-1], a.step], "n": N,
               "n_observed": int(sum(observed[i] for i in mi)),
               "retarget_info": info, "thresholds": THRESH}
        rep["a_stretch"] = {"max_rel": float(max(rec["stretch"]))}
        rep["b_direction_deg"] = {k: stats(v) for k, v in seg_err.items()}
        limb_keys = [k for k in seg_err if k.split(".")[0] in ("upper_arm", "forearm", "thigh", "shin")]
        allv = np.concatenate([seg_err[k] for k in limb_keys])
        rep["b_direction_limbs_all"] = stats(allv)
        rep["c_twist_deg"] = {k: dict(stats(v), flagged_frames=int((np.array(v) > THRESH["twist_flag_deg"]).sum()))
                              for k, v in twist.items()}
        rep["d_hinge"] = {k: {"offplane_max_deg": float(max(hinge_off[k])),
                              "hyperext_max_deg": float(max(hyper[k])),
                              "hyperext_frames_gt5": int((np.array(hyper[k]) > 5).sum())} for k in hinge_off}
        def group(n):
            b_ = n[4:]
            sd = "L" if ".L" in b_ else "R"
            for pre, g in (("upper_arm", "arm"), ("forearm", "fore"), ("hand", "hand"), ("f_", "hand"),
                           ("thumb", "hand"), ("palm", "hand"), ("thigh", "thigh"), ("shin", "shin"),
                           ("foot", "foot"), ("toe", "foot")):
                if b_.startswith(pre):
                    return g + sd
            if b_.startswith(("spine.004", "spine.005", "spine.006")):
                return "head"
            return "trunk"
        av = {n: np.array(v) for n, v in angvel.items() if len(v)}
        mwa = {g: np.array(v) for g, v in mw.items()}
        thr = THRESH["flip_deg_per_frame"] * a.step
        spikes, worst = {}, []
        for n, v in av.items():
            m = mwa[group(n)]
            bad = (v > thr) & (v > m + 20.0)
            if bad.any():
                spikes[n] = int(bad.sum())
            excess = v - m
            i = int(excess.argmax())
            worst.append((float(excess[i]), float(v[i]), n, frames[0] + (i + 1) * a.step))
        worst.sort(reverse=True)
        rep["e_flips"] = {"threshold_deg_per_frame": thr, "rule": "rig > thr and rig > mocap + 20 deg/frame",
                          "rig_max_deg_per_frame": float(max(v.max() for v in av.values())),
                          "bones_with_spikes": spikes,
                          "worst_excess_over_mocap": [{"excess": w, "deg": d_, "bone": n, "frame": fr}
                                                      for w, d_, n, fr in worst[:8]]}
        feet = {}
        for s in ("L", "R"):
            low = np.array(foot_low[s])
            pts = np.array(ankle[s])                      # (N,3 points,3)
            z = pts[..., 2] - model.floor_rest
            v = np.zeros(z.shape)
            v[1:] = np.linalg.norm(np.diff(pts[..., :2], axis=0), axis=-1) / a.step
            on = z < 0.01                                  # point on the floor (< 1 cm)
            con = np.zeros(z.shape, bool)
            con[1:] = on[1:] & on[:-1]
            # sustained contacts: runs of >= 4 frames on the floor, without their first / last
            # frame (touch-down and lift-off are landing, not sliding)
            sus = np.zeros(z.shape, bool)
            for j in range(z.shape[1]):
                k = 0
                while k < len(on):
                    if on[k, j]:
                        e = k
                        while e < len(on) and on[e, j]:
                            e += 1
                        if e - k >= 4:
                            sus[k + 1:e - 1, j] = True
                        k = e
                    else:
                        k += 1
            sl, ss = v[con], v[sus]
            feet[s] = {"min_height_m": float(low.min()),
                       "penetration_frames_gt1cm": int((low < -0.01).sum()),
                       "contact_point_frames": int(con.sum()),
                       "sustained_contact_point_frames": int(sus.sum()),
                       "slide_all_contacts_p95_m_per_frame": float(np.percentile(sl, 95)) if sl.size else 0.0,
                       "slide_mean_m_per_frame": float(ss.mean()) if ss.size else 0.0,
                       "slide_p95_m_per_frame": float(np.percentile(ss, 95)) if ss.size else 0.0}
        rep["f_feet"] = feet

        checks = {
            "a_zero_stretch": rep["a_stretch"]["max_rel"] < THRESH["stretch_max_rel"],
            "b_direction_limbs": rep["b_direction_limbs_all"]["mean"] < THRESH["dir_limb_mean_deg"]
            and rep["b_direction_limbs_all"]["p95"] < THRESH["dir_limb_p95_deg"],
            "c_twist": all(v["flagged_frames"] == 0 for v in rep["c_twist_deg"].values()),
            "d_hinge": all(v["offplane_max_deg"] < THRESH["hinge_offplane_max_deg"] and
                           v["hyperext_max_deg"] <= THRESH["hyperextension_max_deg"] for v in rep["d_hinge"].values()),
            "e_no_flips": not rep["e_flips"]["bones_with_spikes"],
            "f_feet": all(v["min_height_m"] > -THRESH["penetration_max_m"] and
                          v["slide_p95_m_per_frame"] < THRESH["slide_p95_m_per_frame"] for v in feet.values()),
        }
        rep["evaluation"] = label
        return rep, checks

    def show(rep, checks):
        print("==== mocap retarget validation (%d frames, %d with observed core) ====" % (rep["n"], rep["n_observed"]))
        print("a) max DEF stretch          : %.2e" % rep["a_stretch"]["max_rel"])
        print("b) limb direction err (deg) : mean %.2f  p95 %.2f  max %.2f" % tuple(
            rep["b_direction_limbs_all"][k] for k in ("mean", "p95", "max")))
        for k, v in rep["b_direction_deg"].items():
            print("     %-16s mean %6.2f  p95 %6.2f  max %6.2f" % (k, v["mean"], v["p95"], v["max"]))
        print("c) twist max (deg)          : " + ", ".join("%s %.0f" % (k, v["max"]) for k, v in rep["c_twist_deg"].items()))
        print("d) hinge                    : " + ", ".join("%s off %.2f hyper %.1f" % (k, v["offplane_max_deg"], v["hyperext_max_deg"])
                                                         for k, v in rep["d_hinge"].items()))
        print("e) flips                    : spikes %s worst %s" % (rep["e_flips"]["bones_with_spikes"], [(w["bone"], w["frame"], round(w["excess"], 1)) for w in rep["e_flips"]["worst_excess_over_mocap"][:3]]))
        print("f) feet                     : " + ", ".join("%s min %.3f sustained-contact slide mean %.4f p95 %.4f (%d pt-frames; all contacts p95 %.4f)" % (
            s, v["min_height_m"], v["slide_mean_m_per_frame"], v["slide_p95_m_per_frame"],
            v["sustained_contact_point_frames"], v["slide_all_contacts_p95_m_per_frame"]) for s, v in rep["f_feet"].items()))

    # which evaluations: the file as saved; control_mode BOTH also all limbs in IK
    mode = getattr(prm, "control_mode", "FK")
    ik_keyed = mode != "FK" and info.get("control_mode", mode) != "FK"
    saved = {n: float(rig.pose.bones[n]["IK_FK"]) for n in SWITCHES if n in rig.pose.bones}
    desc = ", ".join("%s %s" % (k, "FK" if all(saved[n] >= 0.5 for n in saved if LIMB_KIND[n] == k) else "IK")
                     for k in ("arm", "leg"))
    evals = [("as saved (%s)" % desc, saved)]
    if mode == "BOTH" and ik_keyed:
        evals.append(("all IK", {n: 0.0 for n in saved}))
    results = [(label,) + evaluate(label, sw) for label, sw in evals]
    rep, checks = dict(results[0][1]), dict(results[0][2])
    rep["control_mode"] = mode
    if len(results) > 1:
        rep["evaluations"] = {}
        for label, r, c in results:
            rep["evaluations"][label] = {"checks": c, "a_stretch": r["a_stretch"], "b_direction_limbs_all": r["b_direction_limbs_all"],
                                         "c_twist_deg": r["c_twist_deg"], "d_hinge": r["d_hinge"], "f_feet": r["f_feet"]}
        for label, r, c in results[1:]:
            checks.update({"%s [%s]" % (k, label): v for k, v in c.items()})
    if ik_keyed:
        rep["g_ik_stretch"] = ik_stretch_test(rig, frames[len(frames) // 2], list(saved))
        rep["h_fk_ik_match"] = fk_ik_match(rig, frames)
        set_switch(rig, saved)
        checks["g_ik_no_stretch"] = rep["g_ik_stretch"]["pass"]
        checks["h_fk_ik_match"] = rep["h_fk_ik_match"]["pass"]
    rep["checks"] = checks
    rep["pass"] = all(checks.values())
    out = a.json or os.path.splitext(a.blend)[0] + "_validation.json"
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("\ncontrol_mode %s" % mode)
    for label, r, c in results:
        print("\n---- evaluation: %s" % label)
        show(r, c)
    if ik_keyed:
        g, h = rep["g_ik_stretch"], rep["h_fk_ik_match"]
        print("g) IK stretch, end control 15%% out of reach: max rel %s; IK_Stretch %s, bone ik_stretch %s" % (
            ", ".join("%s %.1e" % (k, v) for k, v in g["reach_115pct_max_rel"].items()),
            sorted({v["IK_Stretch"] for v in g["settings"].values()}),
            sorted({x for v in g["settings"].values() for x in v["bone_ik_stretch"]})))
        endb = {n: v for d_ in g.get("reach_115pct_end_bones_max_rel", {}).values() for n, v in d_.items()}
        if endb:
            print("   (info, not checked: Rigify clamps the ankle but the toe follows foot_ik, so an out-of-reach "
                  "foot_ik stretches %s)" % ", ".join("%s %.0f%%" % (n, 100 * v) for n, v in endb.items()))
        print("h) FK vs IK keys (DEF dir deg) : arms mean %.3f p95 %.3f max %.2f | legs mean %.3f p95 %.3f max %.2f | "
              "end pos max %.3f mm" % (h["dir_deg_arms"]["mean"], h["dir_deg_arms"]["p95"], h["dir_deg_arms"]["max"],
                                       h["dir_deg_legs"]["mean"], h["dir_deg_legs"]["p95"], h["dir_deg_legs"]["max"],
                                       h["end_pos_mm"]["max"]))
    print()
    for k, v in checks.items():
        print("   %-34s %s" % (k, "PASS" if v else "FAIL"))
    print("OVERALL:", "PASS" if rep["pass"] else "FAIL", "->", out)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1:]))
