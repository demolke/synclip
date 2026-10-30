# mocap_retarget: MediaPipe-33 joint positions → Blender 5.x Rigify

This pipeline retargets the same movement onto a different skeleton. It is built for:

- zero bone stretch
- stable twist and no flips
- one-axis elbows and knees
- planted feet
- clean FK keys that can drive a character controller
- optionally matching Rigify IK keys (legs in IK for editing planted feet), switchable without a pop

Everything runs headless with the `bpy` module (Blender 5.2.3). Retargeting itself only needs numpy.

| file | what |
|---|---|
| `build_rig.py` | human metarig → proportion changes (presets, per-segment lengths, shoulder/hip width) → scale to height → Rigify generate; flat box proxies with a "fin" per DEF bone plus a checker floor; saves a `.blend`; stores the rig's mocap landmarks (`mocap_landmarks`) |
| `retarget.py` | add-on (3D view N-panel "Mocap Retarget", per-rig settings, live scrub mode) plus a pure API (`Params`, `retarget()`, `solve_sequence()`, `solve_pose()`, `ik_bases()`) and a CLI |
| `validate.py` | evaluates the depsgraph per frame and checks stretch, direction, twist, hinge, flips and feet; with IK keys also IK stretch and FK = IK; writes a JSON report and PASS/FAIL (exit code) |
| `render_preview.py` | Workbench render from a camera that follows the torso, a calibrated camera, or the scene's own camera; optional mocap "ghost"; PNG → MP4 |
| `side_by_side.sh` | ffmpeg hstack of a render and a reference video: same height, frame-synced, labelled |
| `compose_grid.sh` | 2x2 review video: calibrated-camera render, the camera's video, the rig composited over the video, and a tracking 3/4 view |
| `make_two_characters.py` | two characters with different proportions, both retargeted, in one self-contained `.blend` (camera, light, embedded add-on and mocap) |
| `check_blend.py` | re-opens a retargeted `.blend` like a trusted GUI file and checks rigs, actions, FK, per-rig settings and the embedded panel; can re-bake through the panel operator |
| `export_anim.py` | FBX / glTF export of one character (deform bones + baked action) for a game engine |
| `make_standin.py` | dev stand-in: MediaPipe *world* landmarks of one camera → the contract npz |

## Tutorial: from videos to an animated Rigify character

Every command below was run as written (with the paths of this machine). Set these once per shell, adapted to your machine:

```bash
REPO=/home/user/synclip                 # this repository
MR=$REPO/blender/mocap_retarget         # this folder
PY=/home/user/venv/bin/python           # Python with the capmo dependencies (step 0)
BPY=/opt/bpyenv/bin/python              # Python with the bpy module (step 0)
W=$HOME/mocap_work                      # your working folder
mkdir -p $W
```

### 0) Prerequisites

**capmo (keypoints + 3D reconstruction)**: Python 3.10–3.13.

```bash
python3 -m venv /home/user/venv
/home/user/venv/bin/pip install numpy av opencv-contrib-python mediapipe scipy matplotlib
```

On Linux MediaPipe also needs `libGLESv2` (`sudo apt install libgles2`). `matplotlib` is only for the validation images of step 3.

**Blender 5.x**, one of:
- the GUI app from blender.org (needed for the interactive part, step 5b). Headless, every script here also runs as `blender -b --factory-startup --python SCRIPT.py -- ARGS`; the scripts read their arguments after `--`, and Blender bundles numpy.
- the `bpy` wheel, which is Blender as a Python module. A wheel exists only for the Python version that Blender release is built with (Python 3.13 for 5.2):

  ```bash
  python3.13 -m venv /opt/bpyenv
  /opt/bpyenv/bin/pip install bpy==5.2.3      # pulls numpy
  ```

Rigify ships with Blender; `build_rig.py` enables it.

**ffmpeg** for the videos: `sudo apt install ffmpeg` (macOS: `brew install ffmpeg`). `side_by_side.sh` and `compose_grid.sh` also use `ffprobe` and `fc-match` (fontconfig).

### 1) Record (or collect) the videos

There are two recording paths; everything after step 2 is the same for both.

**A. Phones with capmo (recommended).** Each phone runs the StreamCam app and streams to capmo on the PC, which stores the H.264 stream verbatim plus per-frame timing and the phone clock offsets in a *session directory*. The cameras are then already synchronised to the PC clock (typically to a few ms), and step 2 reads the session directly.

```bash
cd $REPO
# CLI recorder: phones dial <PC IP>:12345; --start tells each phone to stream at the preset
$PY -m capmo.tools.record --out $W/sessions --start --preset 1080p30 \
    --board-squares 7x5 --board-square-len 0.10 --board-marker-len 0.075
#   Enter (or 's'+Enter) starts/stops a recording, 'q' quits; Ctrl+C stops cleanly.
#   --wait: start idle; --duration 90: stop after 90 s; --control-port 8765: HTTP control
#   (curl -X POST localhost:8765/start | /stop, curl localhost:8765/status); kill -USR1 toggles.
#   --convert also writes host-synced MP4s to <session>/videos/.
# or the Qt UI: python -m capmo --record $W/sessions --board-squares 7x5 \
#     --board-square-len 0.10 --board-marker-len 0.075
```

**The Qt UI** (`python -m capmo`) shows a live tile per phone and records with **● Record**. Before recording the board segment, switch **Board check** on and walk the board through the volume: each tile stays amber with a hint ("cover the top-left", "tilt more", "hold the board still", "show to camC together with camA") until that camera has what `calibrate_board` needs (image coverage ≥ 50 %, ≥ 15 distinct sharp views, ≥ 20° tilt, ≥ 3 views together with a linked camera), then turns green. Nothing is solved live: keep Board check on while recording so the board ends up in the session's manifest, then use the commands of steps 2–3 (or the UI's **Process…**, which runs the same chain with default options into `<session>/processed/`, and **View**, which plays the result on a test character; the Blender steps below need the `joints3d.npz`, wherever it came from).

Each recording is `$W/sessions/session_YYYYMMDD_HHMMSS/` with `manifest.json` (cameras, size, fps, clock offsets, the board), `<phone>.h264`, `<phone>.index.jsonl` and `<phone>.clock.jsonl`. The board options only document which board was shown (step 2 uses it by default). To get ordinary video files from a session (to watch or archive), remux it without re-encoding; the PTS of every file are on the common PC clock:

```bash
$PY -m capmo.tools.session_to_take $W/sessions/session_20261028_101500 --out $W/videos/take1 \
    --cam-ids camA,camB,camC,camD        # renames the phones in manifest order
```

**B. Any other cameras** (phones' own camera app, action cams, camcorders): record ordinary video files, one per camera. Follow the advice below; the time offsets between the files are estimated from the motion in step 3.

Advice for both paths:

- **Cameras.** Use 2–4 cameras, 4 is better. Place them around the performer, roughly 60–120° apart, at hip-to-chest height, 2.5–5 m away. The whole body, feet included, must stay in frame for the entire take, in at least two views at any time.
- **Settings.** Use the same frame rate everywhere (30 fps or more) and landscape orientation. Lock focus and exposure; switch off stabilisation, digital zoom and lens switching, because they change the camera intrinsics during the take. Phone rotation metadata is handled.
- **Sync.** The cameras don't need hardware sync. With capmo sessions the PC clock already aligns them and the motion-based estimate only refines it. `reconstruct_take` estimates the time offsets from the motion (epipolar scan, then a reprojection check). A sharp, visible event at the start, such as a clap and a small hop, makes this robust. Start every camera, do the event, perform, and stop the cameras afterwards. Keep the same camera order (`--cam-ids`) for every take.
- **Performance.** Start and end with a second of standing still (an A or T pose is fine). Avoid long stretches where both legs are hidden in all views: occluded limbs are interpolated, not measured (see Known limitations).
- **Board at the start of the videos (recommended).** Start all cameras, show a ChArUco board for 10–30 s, put it away, then perform; the board gives metric, accurate cameras for that take. Without a board, `reconstruct_take` self-calibrates from the body (see "Self-calibration fallback" in step 3).
  - *Board.* Print it flat and rigid, big enough that one square is at least ~20 px in the image: for cameras 3–5 m away use e.g. 7x5 squares of 10 cm (A1/A2), not the A4 default (8x5, 40 mm). Generate it with `$PY -c "from capmo.calibration import BoardConfig; import cv2; cv2.imwrite('board.png', BoardConfig(7, 5, 0.10, 0.075).board().generateImage((2300, 1700), marginSize=100))"`, print at 100 %, and measure a square with a ruler; the measured lengths go to the extractor.
  - *Moving it.* Keep it visible to **at least two cameras at once**, and **hold it still for about a second** in each position, because the cameras are not frame-synchronised. **Tilt** it 20–45° in several directions, and bring it close to each camera so it also covers the **image corners and edges**. Every pair of neighbouring cameras must see it together several times. Then put it out of view and note the time (here 20 s).

### 2) Extract 2D keypoints (locally, per take)

The flags are documented in `capmo/tools/README.md`. Run this from the repository root:

```bash
cd $REPO
# the board segment (video time 0-20 s): ChArUco detections on every frame
$PY -m capmo.tools.extract_keypoints --calib --start 0 --end 20 --calib-every 1 \
    --squares-x 7 --squares-y 5 --square-len 0.10 --marker-len 0.075 \
    --out $W/kp --take take1 --cam-ids camA,camB,camC,camD \
    take1_A.mp4 take1_B.mp4 take1_C.mp4 take1_D.mp4
# the performance (from 20 s): body keypoints
$PY -m capmo.tools.extract_keypoints --start 20 --out $W/kp --take take1 --model heavy \
    --cam-ids camA,camB,camC,camD --keyframes 6 --proxy \
    take1_A.mp4 take1_B.mp4 take1_C.mp4 take1_D.mp4
```

**From a capmo session** (path A of step 1) give the session directory instead of the video files. The cameras come from the manifest (`--cam-ids` renames them in manifest order), the board stored in the manifest is used unless board options are given, and `--start`/`--end` are seconds since the start of the session (the same instant for every camera):

```bash
S=$W/sessions/session_20261028_101500
$PY -m capmo.tools.extract_keypoints --calib --start 0 --end 20 --calib-every 1 \
    --out $W/kp --take take1 --cam-ids camA,camB,camC,camD $S
$PY -m capmo.tools.extract_keypoints --start 20 --out $W/kp --take take1 --model heavy \
    --cam-ids camA,camB,camC,camD --keyframes 6 --proxy $S
```

Each frame's `t` is then its capture time on the PC clock relative to the session start, so the cameras' `t` differ slightly (different start, dropped frames); `reconstruct_take` resamples them onto one grid and uses the clocks as its sync prior (step 3, `--timeline`).

Without a board, run only the second command, without `--start`. The output is `kp/take1/<cam>.npz` (2D landmarks, visibility, timestamps), `<cam>.calib.npz` (board corners), `meta.json`, 480p `*.proxy.mp4` and keyframe JPEGs. `t` and `frame_idx` always count from the start of the video. `--model lite|full|heavy` trades speed for accuracy; `full` is the default and `heavy` is recommended. `--every 2` halves the time. The first run downloads the model.

### 3) Calibrate the cameras and reconstruct 3D joints

**With the board** (step 1):

```bash
cd $REPO
$PY -m capmo.tools.calibrate_board $W/kp/take1 --out $W/kp/take1/rig.json \
    --report $W/kp/take1/rig_report.json --overlay-dir $W/kp/take1/rig_check
$PY -m capmo.tools.reconstruct_take $W/kp/take1 --out $W/mocap --rig $W/kp/take1/rig.json \
    --start 20 --compare-selfcal
```

`calibrate_board` solves each camera's intrinsics from its board views, matches the views of different cameras by frame index (add `--estimate-offsets`, or `--offsets camB=+2,...`, if the cameras were started far apart; for a take extracted from a capmo session the offsets come from the PC clock automatically), chains the cameras through the board views they share, and bundle-adjusts everything. It prints the camera layout and writes `rig.json` in the board's frame: metres, origin at a board corner. `reconstruct_take` then keeps that metric scale and only re-derives the floor, up and heading from the performer. `--start 20` drops the board segment if you extracted the whole video. The output `t` then starts at 0, and `t_source`/`frame_idx` map the rows back to the video. `--compare-selfcal` also self-calibrates from the body and reports the per-camera differences, as a sanity check: a few cm and under 1–2° is good.

Check `rig_report.json` (the `summary` block, then `warnings`):
- **Intrinsic RMS** per camera: good below 0.5 px, acceptable below 1 px.
- **BA RMS** per camera: good below 1 px, acceptable below 2 px. Higher means sync problems (a moving board with unsynchronised cameras) or blur.
- **`coverage`** above 0.5, and **`tilt_deg.max`** of 20–45°. With poor coverage (a small board far away), rerun with `--fix-principal-point` and/or `--ba-intrinsics`.
- **`graph.not_connected`** must be empty.
- In `rig_check/*.jpg`, the red circles (the reprojected board) must sit on the green crosses (the detected corners).

A camera that never shares the board with the others is reported by name. In that case re-record, use the self-calibration path below, or add `--fallback-selfcal`, which places only that camera from the body, anchored to the board cameras.

**Self-calibration fallback** (no board):

```bash
$PY -m capmo.tools.reconstruct_take $W/kp/take1 --out $W/mocap                  # self-calibrating, about 3 min
$PY -m capmo.tools.reconstruct_take $W/kp/take2 --out $W/mocap2 --rig $W/mocap/rig.json   # reuse it: same camera placement, about 1 min
```

The self-calibration takes its scale from an average adult's limb lengths, so the scale is only as good as that prior (a few %).

Outputs in `--out`:
- `take1_joints3d.npz`: **the input of the retargeting.** `joints` (T,33,3) in metres, Z up, floor at z=0, performer facing −Y; `valid` (T,33); `fps`; `t`, `t_source`, `frame_idx`.
- `take1_joints3d_raw.npz`: the triangulation alone.
- `rig.json`: the cameras, in the same world frame.
- `report.json` and `validation/`.

How to read the report: it prints a summary at the end, and the same values are in `report.json`.
- **Bone table.** Raw mean and CV% per bone. A raw CV above about 15% means poor triangulation. The final bones are made constant.
- **"frames ≥2/≥3 agreeing views".** Coverage: frames with fewer than 2 views can't be triangulated and are gap-filled.
- **Jitter** (raw → final) and **lowest foot z p5/p50/p95.** The floor fit: p50 should be about 0, and a p5 far below 0 means a bad floor or scale.
- **left/right ratios.** Should be about 1. `frac_nose_in_front_of_shoulders` / `frac_toes_in_front_of_hips` near 1 means no front/back flips.
- **`sync_*`.** The estimated frame offsets per camera.
- **`final_reprojection_px`.** The error in pixels.

The images in `validation/`:
- `reproj_<cam>.jpg`: the 3D joints projected onto the proxy videos. They must sit on the body.
- `skeleton3d_frames.png`: sample 3D poses.
- `layout.png`: the cameras around the performer.
- `timeseries.png`: bone lengths (raw dots vs final lines), foot and nose heights (feet should touch 0 between steps), and the number of agreeing views per frame. Gaps there are the frames whose joints are interpolated.

### 4) The target character: a Rigify rig

**Build one with `build_rig.py`.** It writes the human metarig with your proportions, generates the Rigify rig, adds box proxies and a floor, and saves a `.blend`:

```bash
cd $MR
# default 1.75 m adult (this repo's reference character: longer legs, shorter arms than the performer)
$BPY build_rig.py --out $W/target_rig.blend --height 1.75 --leg-scale 1.08 --arm-scale 0.95
# stocky cartoon preset: 1.2 m, big torso and head, short legs, long arms, wide shoulders/hips, purple/teal
$BPY build_rig.py --out $W/stocky_rig.blend --preset stocky
# custom proportions: segment length factors (relative to the metarig, before scaling to --height)
$BPY build_rig.py --out $W/custom_rig.blend --height 1.6 --spine 1.1 --neck 0.8 --head 1.3 \
    --upper-arm 1.1 --forearm 1.1 --hand 1.0 --thigh 0.8 --shin 0.85 --foot 1.0 \
    --shoulder-width 1.2 --hip-width 1.1 --girth torso=1.2,arm=1.1,leg=1.15 --palette stocky
```

How the proportion options work:
- Each segment (`--spine --neck --head --upper-arm --forearm --hand --thigh --shin --foot`) scales the length of its bones. The rest of the chain follows, so `--thigh 0.6` shortens only the thighs.
- `--shoulder-width` / `--hip-width` scale the sideways offset of the arm / leg chains.
- `--leg-scale` / `--arm-scale` multiply the whole leg / arm.
- Then everything is scaled uniformly so the top of the head is at `--height`.
- A preset fills in defaults; explicit options override them.
- `--girth` only thickens the proxy boxes, `--palette` picks the colours, and `--rig-name` names the rig object (default `rig`).

The stocky preset is `height 1.2, spine 1.0, neck 0.6, head 1.7, upper_arm/forearm 1.05, hand 1.15, thigh/shin 0.6, foot 1.1, shoulder_width 1.35, hip_width 1.3`. That gives legs (hip→ankle) of 0.42 m vs 0.91 m for the adult, a head that is 20% of the height vs 10%, and arms that are 32% of the height vs 26%.

**Or use your own Rigify character.** Requirements:
- A rig *generated* by Rigify from the **human** metarig (or a metarig with the same `spine`/`basic.super_limb`/`super_finger` layout), not the metarig itself. The face can be on or off.
- The standard control names must exist: `torso`, `spine_fk`, `spine_fk.001`–`.003`, `neck`, `head`, `shoulder.L/R`, `upper_arm_fk`, `forearm_fk`, `hand_fk`, `thigh_fk`, `shin_fk`, `foot_fk`, `toe_fk` (`.L/.R`), and the `ORG-` bones `upper_arm`, `forearm`, `hand`, `thigh`, `shin`, `foot`, `toe` and `spine.006`. `heel.02` and the `f_index.01` / `f_pinky.01` / `thumb.02` fingers are used when present.
- **A slightly bent elbow at rest.** It defines the hinge axis; the human metarig has one. The knees may be straight.
- The character faces −Y, as Rigify's default. Your mesh is skinned to the `DEF-` bones as usual; the retarget keys the FK controls and `torso` (plus the IK limb controls when `control_mode` is not `FK`, see 5c).
- Without `mocap_landmarks` on the rig, the ears, eyes and nose are estimated from the head bone. `build_rig.py` stores the exact ones; for your own rig, re-generate it from its metarig with `build_rig.edit_metarig`'s landmark code or accept the estimate.

### 5a) Retarget headless (CLI)

```bash
cd $MR
$BPY retarget.py --blend $W/target_rig.blend --npz $W/mocap/take1_joints3d.npz --out $W/retargeted_take1.blend
# any parameter: --set name=value ... (offsets: --set off_arm_L=10,0,0 in degrees)
$BPY retarget.py --blend $W/stocky_rig.blend --npz $W/mocap/take1_joints3d.npz --out $W/stocky_take1.blend \
    --embed --set smooth_sigma=2 action_name=take1_stocky
```

What the CLI does:
1. It opens the rig `.blend`, solves the whole take and bakes one action (`--set action_name=...`, default `mocap_retarget`) onto the FK controls (`IK_FK` = 1, i.e. FK). With `--set control_mode=LEGS_IK_ARMS_FK` or `BOTH` the action also holds matching IK keys (5c).
2. It stores the parameters on the rig (`Object.mocap_retarget`) and on the action (`mocap_params`, `mocap_info`, `mocap_npz`), so validation and re-bakes use the same parameters.
3. It saves `--out`.

`--embed` also stores the npz and the add-on in the `.blend` (see 5b). `--rig NAME` picks the rig object (default `rig`). It takes about 15–20 s for 1177 frames. The printed `scale` is the leg ratio rig/performer, used for the root translation.

### 5b) Retarget interactively in the Blender GUI

**Make the panel available.** Use one of these:
- **Install it as an add-on** (once per Blender install). Edit ▸ Preferences ▸ Add-ons ▸ the ▾ menu at the top right ▸ **Install from Disk…** ▸ choose `blender/mocap_retarget/retarget.py`, then tick **Mocap Retarget (MediaPipe -> Rigify)**.
- **Use a `.blend` made with `--embed` or `make_two_characters.py`.** It contains the text `mocap_retarget.py` with *Register* on, so the panel registers when the file is opened with Python auto-run allowed:
  - Blender asks on opening ▸ **Allow Execution**,
  - or Preferences ▸ Save & Load ▸ **Auto Run Python Scripts**,
  - or open the text in the Text Editor and **Run Script** once.

  Rigify's own `rig_ui.py` needs the same permission for its rig UI panel. If both the add-on and the embedded text are present, the newer version (`bl_info` version) is kept: an embedded copy does not replace an equal or newer add-on that is already registered, whatever the load order.

**Where the panel is.**
1. Select the Rigify rig, which is the armature, not a proxy mesh.
2. In the 3D Viewport press **N** and open the **Mocap Retarget** tab.

The panel always edits the **active rig's own settings**, so every character in a scene has its own parameters, npz and action name. If the active object isn't a Rigify rig, a "Rigify rig" field picks which one to edit.

Main controls:
- **Mocap .npz**: a file path, or `text:NAME` for an npz embedded in the `.blend`.
- **Retarget / Re-bake**: solves and bakes the whole take with the current settings, replacing that rig's action. It only touches that rig. It takes 15–20 s for 40 s of mocap.
- **Live (scrub)**: re-solves only the current frame (FK and, per **Controls**, IK controls) on every frame change (before the frame is evaluated, so renders show the frame being rendered) and unassigns the baked action while on. This gives instant feedback for angles, offsets and limits while scrubbing. The per-frame solve and the root are the bake's, but the whole-take steps (rotation smoothing / despike / speed cap, rotation gap fill, ground smoothing, foot lock, and `FROM_FEET`'s foot integration: live keeps the hips in place horizontally) only happen in a bake, so finish with **Re-bake**. Turning Live off puts the previously assigned action (and slot) back.
- `step` > 1 keys every N-th frame only (linear in between), so planted feet between the keys are lost; the bake warns.
- **Load settings from baked action**: copies the parameters stored on the rig's action into the panel. Use it for files made by an older version (stored parameters this version does not know are skipped with a warning).

The parameter groups, and when to touch them:

| group | key parameters | tweak when |
|---|---|---|
| Frames | `frame_start/end` (mocap frames), `step` (key every N), `scene_start`, `mocap_yaw`, `mocap_space` | Trim the take, key sparsely for hand editing (bake every frame for a controller), rotate the capture. Set `mocap_space` = **Rig object** to place a character anywhere by moving its rig object (several characters). **World** keeps the performance where it was captured. |
| Filtering | `smooth_sigma` (joint positions), `rot_smooth`, `rot_despike`, `max_ang_vel`, `despike`, `reject_outliers`, `gap_erode`, `infer_knees`, `fix_lr_swaps`, `gap_fill_rotations` | Jittery result: raise `smooth_sigma` (2–3) or `rot_smooth`. Fast moves look mushy: lower them. Single-frame pops: `rot_despike`. Limbs swap sides: keep `fix_lr_swaps`. |
| Trunk / head | `pelvis_follow`, `trunk_lateral`, `neck_share`, `head_face_weight`, `head_limit`, `clavicle_share` | The pelvis tilts too much or too little, the head nods wrongly (noisy face points: lower `head_face_weight`), the shoulders shrug too much or too little (`clavicle_share`). |
| Twist / hinge | `twist_mix`, forearm / upper-arm / arm-total / thigh / shin twist limits, `leg_twist_mix`, `twist_rate_limit`, `continuity`, `hinge_strict`, `hyperextend`, `elbow_max`, `knee_max`, `bend_lo/hi`, `hand_mix`, `hand_limit`, `foot_limit` | Candy-wrapper forearms: lower `forearm_twist_limit` / `arm_total_twist_limit`. A knee or elbow pops between bend directions: raise `continuity`. A limb looks hyper-extended: lower `hyperextend`. Hands look wild: lower `hand_mix`. |
| Root / ground | `root_auto_scale` (leg ratio), `root_scale`, `root_height_offset`, `root_motion` (Full / In place / From feet), `ground_mode` (Auto / Plant / Clamp / None), `ground_smooth` | The character covers too much or too little ground: `root_scale`. It floats or sinks: `root_height_offset`. For a controller clip use **In place**. For hip-centred input (single-camera world landmarks) use **From feet**. |
| Foot lock | `foot_lock`, `foot_lock_height`, `foot_lock_speed`, `foot_lock_blend`, `foot_flatten`, `foot_pole`, `foot_heading_from_leg`, `foot_pivot`, `foot_lock_hip_drop`, `swing_clearance`, `foot_lock_radius`, `foot_creep`, `heel_rise_max`, `foot_scale_ref_leg` | Feet slide: raise `foot_lock_height` / `foot_lock_speed`. Feet stick during steps: lower them. A planted foot doesn't turn with the body: lower `foot_pivot`. The knees bend more than the performer's in stance: raise `heel_rise_max`. Height / speed / radius / creep are for a 0.9 m leg (`foot_scale_ref_leg`) and scale with the rig's legs; set it to 0 to use absolute metres. |
| Per-limb offsets | Euler offsets (bone-local, degrees) for spine, head, arm, forearm, hand, leg, shin, foot (L/R) | A consistent pose difference, such as arms through a big belly or toes out: rotate that segment for the whole take. Check it in Live mode, then Re-bake. |
| Output | `control_mode` (FK / Legs IK, arms FK / FK + IK keys), `set_fk`, `ik_zero_bone_stretch`, `action_name` | Animators who edit planted feet: **Legs IK, arms FK** (5c). `set_fk` off leaves the `IK_FK` switches alone. Name the action per character or take. A name used by another rig is never overwritten (`_<rig>` is appended). |

A typical loop:
1. Re-bake with the defaults.
2. Run `validate.py` (step 6) or scrub the take.
3. Turn on Live and adjust offsets or limits at the bad frames.
4. Turn Live off and Re-bake.
5. Save.

### 5c) Control modes: FK, legs in IK, or both key sets

`control_mode` chooses which Rigify controls are keyed. Set it with `--set control_mode=...` on the CLI, or in the panel ▸ Output ▸ **Controls**:

| mode | keyed controls | `IK_FK` after the bake | use it for |
|---|---|---|---|
| `FK` (default of `retarget.py`) | FK controls + `torso` | every limb 1 (FK) | export / character controller; identical to the version-1.0 bake |
| `LEGS_IK_ARMS_FK` (default of `make_two_characters.py`) | FK keys **and** matching IK keys on all four limbs | legs 0 (IK), arms 1 (FK) | animators: planted feet are `foot_ik` keys you can move; arms keep their FK arcs and forearm twist |
| `BOTH` | the same keys as `LEGS_IK_ARMS_FK` | every limb 1 (FK) | FK by default, with IK ready on any limb |

```bash
cd $MR
$BPY retarget.py --blend $W/target_rig.blend --npz $W/mocap/take1_joints3d.npz --out $W/take1_legs_ik.blend \
    --set control_mode=LEGS_IK_ARMS_FK action_name=take1_legs_ik
$BPY validate.py --blend $W/take1_legs_ik.blend          # adds g) IK stretch and h) FK = IK
```

How the IK keys are made. They come from the same solved FK pose, analytically (`ik_bases()`), so both key sets describe one motion:
- `hand_ik` / `foot_ik` are attached rigidly to `hand_fk` / `foot_fk`: the hands and feet are exactly where the FK puts them, so the foot lock carries over unchanged. A planted foot is a run of (nearly) constant `foot_ik` keys, creeping or pivoting only where the solver's lock does.
- `toe_ik` gets the toe's rotation relative to the foot (it bends only during a heel lift).
- The poles (`upper_arm_ik_target`, `thigh_ik_target`, with **Pole** on) are carried rigidly by the upper arm / thigh at their rest offset. They always stay in the elbow / knee bend plane on the bending side, so they cannot flip.
- `upper_arm_ik`, `thigh_ik`, `foot_heel_ik` and `foot_spin_ik` stay at identity (not keyed); foot roll is free for you to key.
- Rig settings written by the bake: `IK_Stretch` = 0, `pole_vector` = on, `IK_parent` = `pole_parent` = Root (the IK controls live in world space), and the bone-level IK stretch of `upper_arm_ik`, `MCH-forearm_ik`, `thigh_ik` and `MCH-shin_ik` set to 0 (`ik_zero_bone_stretch`).
- On take1 (both characters) the IK evaluation matches the FK one within 0.1 mm at the wrists, ankles and toe tips, and the leg directions within 0.03° (p95). Flipping any `IK_FK` slider therefore doesn't pop (validate h).

**The stretch gotcha.** Rigify's **IK Stretch** slider (`IK_Stretch`, default 1 on a generated rig) only clamps the distance of the IK target. Each IK chain bone also has a bone-level IK stretch of 0.1. With that, Blender's IK solver stretches the limbs by up to 4% in most frames even with the slider at 0 (validate a) fails at 4.2e-2). `ik_zero_bone_stretch` (on by default) sets it to 0. That is a change to the rig: to get Rigify's stretchy IK back, set *Bone ▸ Inverse Kinematics ▸ IK Stretch* of those four bones to 0.1 and raise the slider. A `foot_ik` dragged beyond the leg's reach still stretches `DEF-foot` (Rigify clamps the ankle, but the toe follows the control), so keep a moved foot within reach. Validate g) reports that as info.

**The arm-twist caveat (why the arms stay FK).** Rigify's IK chain bends only around the elbow / knee hinge (`MCH-forearm_ik` and `MCH-shin_ik` lock Y/Z), so it has no forearm or shin twist. In IK the hand keeps its exact orientation, but the forearm's pronation (up to 90° in take1) moves into the wrist: validate c) `wrist` goes from ≤ 6° in FK to 89–94° in IK, and Rigify passes about half of it to `DEF-forearm.001`. That is a candy-wrapper risk, and a game export without B-bones gets that twist distribution. The shin twist (≤ 20°) likewise moves to the ankle when the legs are in IK. Also, IK with a pole can't hyper-extend, so frames where the FK has a slightly hyper-extended elbow or knee (≤ `hyperextend`, 5°) come out straight in IK (≤ 5.2° direction difference on about 8% of the frames). Use IK arms only for short edits (switch with the slider, snap back with Rigify's FK→IK).

**Editing the IK feet.**
1. Select the rig, Pose Mode, select `foot_ik.L` (the foot box). The Rigify rig UI panel (N ▸ Item, needs script auto-run) has the IK-FK slider, **IK Stretch** and the snap buttons.
2. A contact is a run of keys with nearly the same location. To move a planted step, select that run of keys in the Dope Sheet / Graph Editor and move them together (G). To change the knee direction, move `thigh_ik_target`. To roll the foot, key `foot_heel_ik` (it is at identity).
3. To carry the edit back to FK (for an FK-only export), use Rigify's **FK->IK (foot.L)** snap with its **Action** button, which bakes over the keyed range.
4. A **Re-bake** replaces the whole action, both key sets included: keep hand edits in a separate action or an NLA layer.
5. Parent space: the IK controls are parented to Root. Rigify's **IK Parent** switcher (with its Action button) rebakes them into Torso or another space without moving them.

**Export.** Exporters sample the `DEF-` bones (step 9), so the mode only matters through the twist distribution: with the legs in IK the shin twist (≤ 20°) sits in the ankle. For an exact FK deform clip, export a `FK` or `BOTH` bake, or set the legs' `IK_FK` to 1 before exporting.

**Live mode** sets the IK controls too (the same `ik_pose()`), so scrubbing shows the chosen mode. It does no temporal processing, so a frame with a hyper-extended knee can differ between FK and IK by a few degrees there.

### 6) Validate

```bash
cd $MR
$BPY validate.py --blend $W/retargeted_take1.blend                    # JSON next to the .blend, exit code 0 = PASS
$BPY validate.py --blend $W/stocky_take1.blend --json $W/stocky_take1_validation.json
# several rigs in one file: --rig NAME
```

It evaluates the depsgraph, so this is what Blender actually shows, every frame. The metrics and thresholds (`validate.THRESH`):

| check | metric | pass |
|---|---|---|
| a) stretch | max relative length change of any DEF bone | < 1e-3 (it is about 1e-5: only rotations are keyed) |
| b) direction | angle between rig limb segments (upper arm, forearm, thigh, shin) and the processed mocap, only on frames where that segment was observed; feet, hands, trunk, shoulder/hip/ear lines are reported but not gated | limbs mean < 6°, p95 < 15° |
| c) twist | forearm vs upper arm, upper arm vs clavicle, thigh vs pelvis, shin vs thigh, chest vs pelvis, wrist (hand vs forearm) twist (on the ORG chain, i.e. what the IK_FK switches select) | no frame > 100° |
| d) hinge | elbow/knee bend outside the hinge plane; hyper-extension | off-plane < 2°, hyper-extension ≤ 10° |
| e) flips | per-frame rotation of every DEF bone vs the mocap's own angular speed | none > 45°/frame **and** > mocap + 20°/frame |
| f) feet | min height of heel/ball/toe tip vs the floor; slide speed of points held on the floor ≥ 4 frames (sustained contacts) | penetration < 15 mm, slide p95 < 10 mm/frame |

| g) IK stretch (IK keys only) | each limb in IK with its `hand_ik` / `foot_ik` pushed 15% beyond reach (action unassigned for the test): relative stretch of its upper / lower DEF bones; the `IK_Stretch` and bone IK stretch values are reported | < 1e-3 |
| h) FK = IK (IK keys only) | all limbs FK vs all limbs IK: DEF limb directions per arms / legs, and wrist / ankle / toe-tip positions | mean < 0.5°, p95 < 2°, positions < 1 mm |

a)–f) run on the file as saved, i.e. with its `IK_FK` switches. For `control_mode` `BOTH` they also run with every limb in IK. `OVERALL: PASS` needs all of them. The JSON also has the per-segment statistics and the worst flip frames, which are the frames to inspect.

### 7) Render previews and the side-by-side

```bash
cd $MR
B=$W/retargeted_take1.blend
$BPY render_preview.py --blend $B --out $W/preview_take1.mp4 --start 1 --end 600 \
    --ghost $W/mocap/take1_joints3d.npz                                  # torso-following 3/4 view + mocap dots
./side_by_side.sh $W/preview_take1.mp4 $W/kp/take1/camB.proxy.mp4 $W/sbs_take1.mp4 0 480 "retarget" "camB"
# review grid from a calibrated camera (rig.json is in the npz's world frame)
$BPY render_preview.py --blend $B --out $W/take1_camB_render.mp4 --calib $W/mocap/rig.json:camB
$BPY render_preview.py --blend $B --out $W/take1_camB_overlay.mov --calib $W/mocap/rig.json:camB --overlay --no-stamp
$BPY render_preview.py --blend $B --out $W/take1_track34.mp4 --view track34
./compose_grid.sh $W/take1_camB_render.mp4 $W/take1_camB_overlay.mov $W/take1_track34.mp4 \
    $W/kp/take1/camB.proxy.mp4 $W/take1_grid.mp4 camB 0 480 23
```

Options:
- **Views.** `--view front34|track34|side|front` follow the torso, `--view scene` uses the file's own camera, and `--calib FILE:CAM` uses a calibrated camera.
- **Sampling and output.** `--step N` renders every Nth frame (fast looks), `--res WxH` sets the size, and `--png-dir DIR` keeps the frames.
- **Speed and setup.** Workbench needs no GPU or X server; it renders at about 0.5–0.8 s per 960x540 frame on 4 cores, and the EGL warnings are harmless.
- **Side-by-side sync.** `side_by_side.sh RENDER VIDEO OUT [VIDEO_START_FRAME] [HEIGHT] [LABELS]` pairs render frame k with video frame START + k, resampled to the render's fps. If scene frame 1 is mocap frame 0, START is 0.

### 8) Several characters in one scene

`make_two_characters.py` builds the delivered `two_characters_take1.blend`:

```bash
cd $MR
$BPY make_two_characters.py --npz $W/mocap/take1_joints3d.npz --out $W/two_characters_take1.blend   # about 45 s
#   --control-mode LEGS_IK_ARMS_FK (default) | FK | BOTH
$BPY validate.py --blend $W/two_characters_take1.blend --rig rig_adult  --json $W/two_adult_validation.json
$BPY validate.py --blend $W/two_characters_take1.blend --rig rig_stocky --json $W/two_stocky_validation.json
$BPY render_preview.py --blend $W/two_characters_take1.blend --rig rig_adult --view scene --res 960x540 \
    --out $W/two_characters_take1.mp4                                    # full take, about 15 min (--step 2 halves it)
$BPY check_blend.py --blend $W/two_characters_take1.blend --rebake rig_stocky   # re-open + panel re-bake test
```

What it does, in the order you would do it by hand:
1. **Build two characters into one file.** It calls `build_rig.build_character()` twice: `rig_adult` (1.75 m, legs ×1.08, arms ×0.95, blue/orange) and `rig_stocky` (`--preset stocky`, purple/teal). Each one gets its own collection, metarig, Rigify rig and proxies. The rig objects stand at x = −1 m and +1 m (`--spacing 2.0`).
2. **Embed the mocap.** The npz goes into the file as the text `take1_joints3d.npz`, and both rigs read `text:take1_joints3d.npz`.
3. **Retarget each rig with its own settings**: `mocap_space = OBJECT`, because the performance must follow each rig object rather than stay at the world origin, and `action_name = take1_<rig>`. `control_mode` comes from `--control-mode`, default `LEGS_IK_ARMS_FK`: FK keys plus matching IK keys, legs switched to IK, arms to FK (5c). Keeping both key sets is what lets an animator flip any limb's `IK_FK` slider without a pop. That gives two actions (149 f-curves each) sharing the scene range 1–1177 @ 30 fps.
4. **Set up the scene.** It adds a camera whose target follows the smoothed midpoint of both characters, at the smallest distance that frames both on every frame, plus a sun light, the Workbench solid/object-colour look, a floor and `retarget.py` as the registered text `mocap_retarget.py`. Then it packs and saves (compressed).

The characters move the same way:
- Arms, spine and hands have the same world direction on every frame, within 0.1°.
- Legs differ by about 2° on average, because each rig's foot lock fits its own leg and foot proportions to the floor (§5).
- The root paths differ: each is scaled by its own leg ratio (1.16 vs 0.54), so the stocky character covers less ground.

To do the same by hand in the GUI:
1. Append or link the second character's collection (File ▸ Append ▸ `.blend` ▸ Collection).
2. Move its rig object sideways.
3. Select that rig, then in the Mocap Retarget panel set the npz, **Mocap space = Rig object** and an **Action** name.
4. Click **Retarget / Re-bake**.
5. Repeat for each rig.

### 9) Use the result: export for a character controller

- **Keys.** Bake every frame (`step` = 1, the default). The action holds quaternion keys on the FK controls plus `torso` location (plus the IK controls with `control_mode` ≠ `FK`), with linear interpolation. Rigify's DEF bones follow through constraints, so exporters must **sample** the pose, as below.
- **Root motion vs in place.** `root_motion=FULL` (default) moves `torso` through the world, so the action carries root motion. A controller that moves the character itself wants an in-place clip:

  ```bash
  $BPY retarget.py --blend $W/target_rig.blend --npz $W/mocap/take1_joints3d.npz --out $W/take1_inplace.blend \
      --set root_motion=IN_PLACE action_name=take1_inplace
  ```

  `IN_PLACE` keeps the height (hops, crouches) but removes XY. The feet then move backwards under the body during steps, like on a treadmill, and the controller's own motion has to match them. `validate.py` f) (foot slide) is therefore only meaningful for `FULL` root-motion clips; an in-place clip fails it by design (p95 about 2–3 cm/frame on take1). The other checks pass. For a clip with planted feet, export the `FULL` action and extract the root motion in the engine.
- **Export** the deform skeleton and the baked action (FBX or glTF; meshes bone-parented to the rig come along unless `--no-meshes`):

  ```bash
  $BPY export_anim.py --blend $W/retargeted_take1.blend --rig rig --out $W/take1.fbx
  $BPY export_anim.py --blend $W/two_characters_take1.blend --rig rig_stocky --out $W/stocky_take1.glb --at-origin
  ```

  This exports only the `DEF-` bones (73 for the human rig without face), every frame. Use `--all-bones` to include Rigify's controls. `--at-origin` removes a character's side offset. In the GUI the same is File ▸ Export ▸ FBX with *Only Deform Bones*, *Add Leaf Bones* off and *Bake Animation* on with step 1 and simplify 0. For glTF it is *Deformation Bones Only* and *Always Sample Animations*.
- **Retarget in the engine instead.** The FK action can also be retargeted again inside the engine (Unity Humanoid, Unreal IK Retargeter), because the deform hierarchy is a standard humanoid.

### Troubleshooting

| symptom | fix |
|---|---|
| `KeyError: control bone 'torso' missing` | The object is not a generated Rigify human rig, or it is the metarig. Pass `--rig` with the generated rig's name. |
| `ValueError: rest elbow is straight` | Bend the metarig's elbows slightly and re-generate. |
| No Mocap Retarget tab | Install the add-on, or allow script auto-run for an embedded file. Select the armature and press N. |
| `embedded mocap 'text:...' not found` | That `.blend` has no embedded npz: set a file path in the panel, or re-run with `--embed`. |
| Character far from where you placed it | `mocap_space` is World: switch to Rig object, or put the rig at the origin. |
| Feet slide or skate | Raise `foot_lock_height` / `foot_lock_speed`, and check `validate.py` f). With `step` > 1, keys are linearly interpolated, so bake every frame. |
| Knees bend too much in stance | Raise `heel_rise_max`, or set `foot_lock_hip_drop` off (feet may then slide). |
| Twisted forearm (candy wrapper) or knee flipping | Lower the twist limits, raise `continuity`, and look at `validate.py` c)/e) for the frames. |
| Jitter | Raise `smooth_sigma` / `rot_smooth`, or check the 3D data (step 3 `timeseries.png`). |
| `scripts disabled for ..., skipping 'rig_ui.py'` (headless) | Harmless. Rigify's UI script is not needed for baking or validation. |
| Limbs stretch a few % in IK (validate a) | The rig's IK chain bones still have bone-level IK stretch 0.1: re-bake with `ik_zero_bone_stretch` on (5c). |
| Wrist twisted ~90° after switching an arm to IK | Expected: Rigify's IK arm has no forearm twist (5c). Keep the arms in FK. |
| A moved `foot_ik` stretches the foot | It is beyond the leg's reach: move it back, or lower the hips. |
| Proxies in the wrong place after appending | Proxies are bone-parented to their rig: append the whole character collection. |

## 1. Research: approaches for joint-position mocap → Rigify

**(a) Copy positions / drive only IK targets.**
Put the hand/foot IK targets (and maybe poles) at the mocap joints.
- Trivial, and the end effectors are exact.
- The target has different proportions, so positions must be rescaled. Any mismatch either stretches the rig (Rigify IK stretch) or leaves limbs straight and shy of the target.
- Spine, neck, clavicles and every twist are undefined. Elbow/knee direction depends entirely on the pole, and wrist/ankle orientation is lost.
- Verdict: fine for effectors, wrong as the main driver.

**(b) FK rotations from direction vectors: swing from the segment direction, twist from a secondary reference, lengths from the target.**
Each segment gets an orthonormal frame built from the joint positions. The primary axis is the segment direction. The secondary axis is a stable anatomical reference:

| segment | twist reference |
|---|---|
| pelvis | hip line |
| chest | shoulder line |
| head | ear line plus face/neck |
| upper arm / thigh | elbow / knee bend plane |
| forearm | radial hand direction (index − pinky) |
| shin | foot heading |
| hand | index/pinky |
| foot | toe direction plus shin axis |

Only rotations are applied, so the target's bone lengths are untouched and there is zero stretch by construction.

The hard parts are all twist problems:
- ambiguous twist when a limb is straight
- the ± sign of the bend plane (a noisy, slightly hyper-extended limb looks exactly like a flexed limb with a 180°-twisted upper segment)
- quaternion hemisphere flips
- the rest-pose twist offset

These are solved by:
- building the rest frame with the *same* function from landmarks measured on the target's rest pose, so there is no rest-pose bias
- swing–twist decomposition per joint with limits
- hinge projection for the elbow and knee
- temporal unwrapping

**(c) Driving Rigify IK controls vs FK controls.**
IK means hand/foot targets scaled by the limb-length ratio, plus a pole from the elbow/knee plane.
- Feet stay grounded, the controls are friendly to animators, and the bend is guaranteed planar.
- But the end-effector position has to be invented when proportions differ. Limb length ratios differ per limb, so "scaled" targets still distort angles (arms reach too far or too short).
- Upper-arm/thigh twist comes from the pole, which is just as ambiguous as (b).
- Rigify IK stretch must be disabled or it stretches.

FK reproduces joint angles, which is what "the same movement" means across proportions. Ground contact is then a local fix: two-bone IK on the legs only where a foot is planted.

Measured on take1: IK targets at the leg-ratio-scaled mocap points would put the adult's hands out of reach on every frame (its arms are 0.76× the performer's, the root scale 1.16×), and bend the stocky character's elbows 60° more on average. So the IK keys are not made from mocap positions at all, but from the solved FK pose (5c).

**(d) Live constraints (Damped Track / Locked Track to mocap empties) vs baked keys.**
Constraints are quick to set up and live. However:
- Damped Track only gives swing.
- Locked Track gives twist but flips when the lock axis aligns with the target.
- Nothing enforces joint limits, hinge axes or continuity.
- The result depends on evaluation order and must still be baked for export or a character controller.

Baked quaternion keys are deterministic, exportable and filterable over time (despike, smoothing, foot locking need the whole take).

**Recommendation (implemented).**
Use (b) and bake onto Rigify **FK controls** (the IK_FK switch is set to FK). Specifically:
- quaternion keys with hemisphere continuity
- root (torso) translation from the hip centre scaled by the leg-length ratio
- foot handling on top: ground plant, contact detection, foot lock by analytic two-bone IK with fixed lengths, hip drop if out of reach, flat planted feet, swing clearance

A live frame-change mode gives (d)'s interactivity for tweaking, without baking.

IK output is an option on top (`control_mode`, 5c): the IK controls are computed analytically from the FK solution (16 ms/frame for the prototype; part of the 15–20 s bake). Rigify's own IK→FK snap operator (`pose.rigify_limb_ik2fk_<rig_id>`) also runs headless per frame, but at about 175 ms/frame (3.5 min per rig for take1). It needs the rig_ui script executed and the "Only Insert Available" keying preference off, and in a test it failed on 21 of 300 frames with straight limbs (it keyed scale up to ×3).

## 2. How the solver works (`retarget.py`)

1. **Pre-process the mocap.**
   - Repair frames whose left/right labels are swapped (`fix_lr_swaps`). The unpaired nose is the witness: the facing from the shoulders and hips must point to the nose side. Ambiguous neighbours of a confirmed swap are relabelled by continuity.
   - Reject outliers: bone-length consistency, a hip-relative temporal median test, and the frames next to tracking gaps (`gap_erode`).
   - Infer missing knees by two-bone IK from an observed hip and ankle (`infer_knees`), for example in an occluded crouch.
   - Fill gaps linearly. A joint that is never observed (no valid frame after the rejection) is an error naming it: the segments it defines cannot be solved, and the solver no longer silently puts it at the origin.
   - Median despike, then Gaussian smoothing.
   - Optional yaw.
2. **Per frame, world "delta" rotations `D = F_pose · F_restᵀ`** (rest → pose) for each segment.
   - The pelvis partly follows the trunk (`pelvis_follow`).
   - The spine bend is spread over torso / spine_fk.001 / .002 / .003. A sagittal correction makes the rig's hip→shoulder line match the mocap trunk.
   - Neck takes `neck_share` of the head rotation.
   - Clavicle takes `clavicle_share` of the arm elevation above horizontal.
   - **Limbs**: the upper-arm/thigh twist comes from the bend plane, blended toward a fallback when the limb is nearly straight (minimal twist for arms, foot heading for legs). *Both* plane signs are solved with all limits applied, and the one whose constrained lower segment matches the data best wins, with a bias to temporal continuity.
   - Elbow/knee are projected onto the hinge axis (`hinge_strict`) and clamped (`hyperextend`, `elbow_max`, `knee_max`).
   - Forearm twist comes from the hand (`twist_mix`, limits). The arm's total twist is limited (Rigify's mid twist segment flips near 180°).
   - Hand and foot: swing limits, no wrist twist.
   - Every twist is unwrapped against the previous frame and rate limited.
3. **Temporal post-processing on parent-local rotations.**
   - Gaps are filled per control in **rotation space**. An observed segment keeps its solved world rotation; an unobserved one gets its local rotation slerped between the frames where it was observed. Flickering 1–2 frame detections still inform positions but don't count as observed.
   - Glitch despike, angular-speed cap, Gaussian smoothing.
   - All joint limits are re-enforced.
4. **Root.**
   - The hip centre follows the mocap hip centre × (rig leg / mocap leg) × `root_scale`, anchored on the world origin, so the rig stays where the performer is. The height is scaled above the stance-ankle level.
   - With `CLAMP`, a leg that wasn't observed doesn't lift the observed hips; it is bent by IK instead.
   - `root_motion`: `FULL`, `IN_PLACE` (character controller), or `FROM_FEET`. `FROM_FEET` integrates XY from the planted foot and is meant for hip-centred input.
   - `ground_mode`: `AUTO` (the default) is `CLAMP` for `FULL` and `PLANT` otherwise. `PLANT` puts the lowest foot on the floor; `CLAMP` only prevents penetration.
   - `mocap_space`: `WORLD` (the default) maps the mocap into the rig's armature space through the inverse of the rig object's world matrix, so the performance stays where it was captured: moving or turning the rig object (about Z) does not move it. The root scale is applied in the mocap's world frame, around its origin, and only then mapped into the armature space. `OBJECT` attaches the mocap's world frame to the rig object, so moving a rig moves its whole performance by exactly the rig's transform (several characters side by side). The hip height is anchored on the rig's ankle level, so a rig placed higher (Z) lifts the performance in both modes.
   - Live mode and the bake share the root computation (`root_target()`, the same frame range and observed-joint mask).
   - `frame_start` / `frame_end` outside the mocap's frames are an error (`ValueError`).
5. **Feet.**
   - Contacts are detected by height and speed: from the scaled *mocap* feet for world-space input, otherwise from the rig.
   - The anchor (position + flat yaw heading, mostly from the leg) is held by two-bone IK.
   - Ratchets: the planted foot turns only when the leg turns more than `foot_pivot`, and it creeps (at most `foot_creep` per frame) only when the free ankle is further than `foot_lock_radius` away.
   - A planted foot that is out of reach first lifts its heel: it rolls about the ball, the toes stay flat and the `toe_fk` bends, up to `heel_rise_max`. Only then do the hips drop. Non-planted feet get `swing_clearance`.
   - The contact height and speed and the lock radius and creep are given for a 0.9 m leg and scaled by the rig's leg length (`foot_scale_ref_leg`), so a 1.2 m character's lower, shorter steps are detected the same way.
   - The knee pole of a planted leg follows the toe heading, which stays flat during a heel lift.
6. **Bake.** Local control quaternions are `q = R_restᵀ · (D_parentᵀ D) · R_rest`, plus the torso location. They are written with `foreach_set` into an Action (Blender 5 slotted actions / channelbag).
   - Rigify setup: `IK_FK = 1` (FK), `FK_limb_follow = 0`, `neck_follow = head_follow = 1`, so plain FK inheritance holds. The analytic FK was verified against the depsgraph to within 0.01°.
   - With `control_mode` ≠ `FK` (5c): IK controls attached rigidly to their FK counterparts. Each control X follows the FK control's motion A (x ↦ pos_c + D (x − H_c)): rotation `X_restᵀ · D · X_rest`, and location = the head offset in the bone's location axes. Rigify's IK controls have *Local Location* off, so that is the Root parent's rest axes. Quaternion hemisphere continuity applies here too. Then `IK_Stretch` = 0, Pole on, Root parents, bone IK stretch 0, and `IK_FK` per mode.

### Tests

`tests/` holds a pytest suite for Blender's Python module (bpy). It builds the rig once per session and uses `tests/take1_slice120.npz` (120 frames of take1), so it needs nothing from `/home/user/data`; about 25 s:

```bash
/opt/bpyenv/bin/python -m pip install pytest      # once
/opt/bpyenv/bin/python -m pytest blender/mocap_retarget/tests
```

It covers: the WORLD / OBJECT root placement, live pose == baked pose per root mode, live renders, Live off restoring the action, frame-range errors, loading stored parameters with unknown keys (panel and `validate.py`), add-on registration versus an equal or newer registered copy, the npz loader (no pickles), never-observed joints, `fps_base`, the bake's target scene, the `step` warning, and `validate.py --step`.

## 3. Commands (reference)

The tutorial above has the full sequence. These are the commands behind the delivered files in `/home/user/data/mocap/`:

```bash
cd /home/user/synclip/blender/mocap_retarget
# 0) stand-in input (numpy only). The real input is /home/user/data/mocap/take1_joints3d.npz
/home/user/venv/bin/python make_standin.py /home/user/data/kp/kp/take1/camA.npz /home/user/data/mocap/standin_joints3d.npz
# 1) target rig (1.75 m, longer legs / shorter arms than the performer on purpose)
/opt/bpyenv/bin/python build_rig.py --out /home/user/data/mocap/target_rig.blend --height 1.75 --leg-scale 1.08 --arm-scale 0.95
# 2) retarget + bake (hip-centred stand-in: add --set root_motion=FROM_FEET)
/opt/bpyenv/bin/python retarget.py --blend /home/user/data/mocap/target_rig.blend \
    --npz /home/user/data/mocap/take1_joints3d.npz --out /home/user/data/mocap/retargeted_take1.blend
# 3) validate, 4) previews and review grid: see tutorial steps 6 and 7
# two characters (adult + stocky) in one file, validated, rendered, re-open checked
D=/home/user/data/mocap/deliver
/opt/bpyenv/bin/python make_two_characters.py --npz /home/user/data/mocap/take1_joints3d.npz --out $D/two_characters_take1.blend
/opt/bpyenv/bin/python validate.py --blend $D/two_characters_take1.blend --rig rig_adult --json $D/two_characters_take1_rig_adult_validation.json
/opt/bpyenv/bin/python validate.py --blend $D/two_characters_take1.blend --rig rig_stocky --json $D/two_characters_take1_rig_stocky_validation.json
/opt/bpyenv/bin/python render_preview.py --blend $D/two_characters_take1.blend --rig rig_adult --view scene --res 960x540 --out $D/two_characters_take1.mp4
/opt/bpyenv/bin/python render_preview.py --blend $D/two_characters_take1.blend --rig rig_adult --view scene --res 960x540 --start 380 --end 800 --out $D/two_characters_take1_ik.mp4
/opt/bpyenv/bin/python check_blend.py --blend $D/two_characters_take1.blend --rebake rig_stocky
```

**Camera choice: camB.**
- Detection rate 0.86, the same as camA's 0.87; camC and camD are at 0.44 and 0.34.
- The performer is largest on screen there: median 247 px tall, vs 217 px for camA.
- It shows kick 2 most clearly: 156 px ankle separation, vs 113 px for camA.

**Ghost.** The optional ghost (`--ghost`) shows the processed mocap joints, mapped exactly like the rig's root. In `overlay` mode they are dots on the rig; in `side` mode they are dots plus sticks 1 m to the side. Unobserved joints are hidden. The old side ghost looked like a wide A-pose because it sat closer to the camera and was scaled by the leg ratio (1.16) about z=0, and the mocap trunk is long (MediaPipe hip points sit low). Reprojecting the data into the videos shows the data itself is right.

**Python API:**
```python
import retarget as mr
info = mr.retarget(bpy.data.objects["rig"], "take.npz", mr.Params(smooth_sigma=2, root_motion="IN_PLACE"))
mr.settings_from_params(bpy.data.objects["rig"].mocap_retarget, mr.Params(), "take.npz")  # panel settings
mr.embed_npz("take.npz"); mr.embed_text_module()       # make the open .blend self-contained
```

## 4. Interactive parameters (per rig, N-panel)

Each rig object stores its own settings (`Object.mocap_retarget`, the same fields as `Params`). The scene-level `Scene.mocap_retarget` of version 1.0 files is still registered: its rig pick selects which rig the panel edits, and "Load settings from baked action" migrates it.

- **Frames:** start/end mocap frame, key every N (subsampling), scene start frame, mocap yaw, mocap space (World / Rig object).
- **Filtering:** `reject_outliers`, `despike`, `smooth_sigma` (joints), `rot_smooth`, `rot_despike`, `max_ang_vel`, gap fill in rotation space.
- **Trunk / head:** `pelvis_follow`, `trunk_lateral`, `neck_share`, `head_face_weight`, `head_limit`, `clavicle_share`.
- **Twist / hinge:**
  - `twist_mix` and the forearm, upper-arm and arm-total twist limits
  - thigh/shin twist limits and `leg_twist_mix`
  - `twist_rate_limit`, `continuity`
  - `hinge_strict`, `hyperextend`, `elbow_max`, `knee_max`, `bend_lo` / `bend_hi`
  - `hand_mix`, `hand_limit`, `foot_limit`
- **Root / ground:** auto scale (leg ratio), `root_scale`, `root_height_offset`, `root_motion`, `ground_mode`, `ground_smooth`.
- **Foot lock:**
  - on/off, height and speed thresholds, blend
  - flatten, pole, heading from leg, pivot, radius, creep
  - hip drop, swing clearance
  - `heel_rise_max`
  - `foot_scale_ref_leg`: the length thresholds scale with the rig's leg length
- **Per-limb rotation offsets** (bone-local Euler): spine, head, arm, forearm, hand, leg, shin and foot, for L and R.
- **Output:** `control_mode` (**Controls**: FK / Legs IK, arms FK / FK + IK keys; the tooltip has the arm-twist caveat), set IK/FK switches (`set_fk`), `ik_zero_bone_stretch`, action name.

## 5. Results (1177 frames @ 30 fps)

The current code produced these numbers:
- The two take1 columns are the two characters of `deliver/two_characters_take1.blend`.
- The adult is the same rig as `target_rig.blend`: 1.75 m, legs ×1.08, arms ×0.95.
- The stand-in was retargeted onto `target_rig.blend` with `FROM_FEET` (see the note under the tables).

Direction errors are scored per segment, only on frames where that segment's joints were observed. Unobserved segments are filled in rotation space.

| check | take1, adult 1.75 m | take1, stocky 1.2 m | stand-in, camA MediaPipe world |
|---|---|---|---|
| frames with the whole core observed | 412 | 412 | 689 |
| leg ratio rig/performer (root scale) | 1.16 | 0.54 | 1.22 |
| a) max DEF stretch | 9e-6 | 9e-6 | 6e-6 |
| b) limb direction, mean / p95 | 2.8° / 9.2° | 3.1° / 9.6° | 2.8° / 11.5° |
| arms (mean) | 0.5–2.3° | 0.5–2.3° | 0.3–1.9° |
| legs (mean) | 3.9–4.5° ¹ | 4.1–4.9° ¹ | 2.0–7.7° ¹ |
| trunk / shoulder / hip / ear line (mean) | 2.9 / 2.5 / 1.1 / 6.3° | 2.9 / 2.5 / 1.1 / 6.3° | 6.6 / 4.5 / 1.1 / 1.3° ² |
| c) max twist: forearm / upper arm / thigh / shin / spine | 90 / 95 / 60 / 20 / 74° | 90 / 95 / 60 / 20 / 74° | 90 / 95 / 94 / 20 / 33° |
| d) hinge: off-plane / hyper-extension | 0.00° / ≤ 5° | 0.00° / ≤ 5° | 0.00° / ≤ 5° |
| e) flips beyond the mocap's own rate | none (worst excess 36°/frame) | none (worst 36°/frame) | none (worst 33°/frame) |
| f) penetration | 0 | 0 | 0 |
| f) sustained-contact slide, p95 L / R | 7.4 / 6.4 mm/frame | 3.7 / 3.6 mm/frame | 5.3 / 1.1 mm/frame |
| **overall** | **PASS** | **PASS** | **PASS** |

**Control modes** (`make_two_characters.py --control-mode ...`, same take; the `FK` keys of every mode are bit-identical to the version-1.0 bake). a)–f) are the same as in the table above unless listed:

| | `FK` | `LEGS_IK_ARMS_FK` (delivered) | `BOTH`, as saved (all FK) | `BOTH`, all IK |
|---|---|---|---|---|
| a) max DEF stretch, adult / stocky | 9e-6 / 9e-6 | 5e-4 / 3e-4 | 9e-6 / 9e-6 | 5e-4 / 3e-4 |
| c) wrist twist max L / R | 5 / 6° | 5 / 6° | 5 / 6° | 94 / 89° |
| c) shin twist max | 20° | 0° (in the ankle) | 20° | 0° |
| d) hyper-extension max | 5° | 5° (arms) | 5° | 0° |
| g) limb stretch, end control 15% out of reach | – | ≤ 1.5e-6 | ≤ 1.5e-6 | |
| h) FK vs IK, arms mean / p95 / max | – | 0.13 / 0.03 / 5.2° | same | |
| h) FK vs IK, legs mean / p95 / max (adult; stocky) | – | 0.007 / 0.03 / 1.9° (0.006 / 0.03 / 0.04°) | same | |
| h) wrist / ankle / toe position max | – | 0.10 mm | same | |
| **overall, both characters** | **PASS** | **PASS** | **PASS** | **PASS** |

**Events on take1** (rig vs the data mapped by the same root scale):

| event | adult rig | adult data | stocky rig | stocky data |
|---|---|---|---|---|
| hop (~13.4 s): ankles up | 0.53 m | 0.54 m | 0.28 m | 0.28 m |
| crouch (~15.9 s): hips down to | 0.50 m | 0.50 m | 0.26 m | 0.26 m |
| crouch: knee flexion L / R | 112 / 124° | | 111 / 122° | |
| kick 2 (~25.8 s): ankle up | 0.72 m | 0.73 m | 0.38 m | 0.37 m |

**Adult vs stocky, same frame** (world directions, every 2nd frame of the take):
- Arms, spine and hands differ by under 0.1°.
- The head differs by a constant 1.2°, because its rest shape is different.
- Thighs and shins differ by 2.1–2.5° on average (p95 7–8°), because each rig's foot lock fits its own leg and foot proportions to the floor.
- Feet differ by 3.4–4.3° on average, because planted feet are flattened and heel-lifted per rig.

Kick 3 has no leg data, and kick 4 is barely lifted in the data. The foot lock only engages for feet that are low and slow, so it doesn't fight the kicks.

¹ The rigs' thigh/shin ratio (1.18) differs from the performer's (about 1.0), so a planted foot changes the knee angle; with `foot_lock=0` the legs track at about 1°. Before `heel_rise_max` (version 1.0), an out-of-reach planted foot lowered the hips at once. That bent both knees about 15° more than the performer's (legs 6–8°, p95 14°), and for the stocky rig 12–14° (limb p95 24°, FAIL). Its feet are 1.8× longer relative to its legs and its contact thresholds were not scaled to its size.

² The mocap's trunk is not perpendicular to its own shoulder line, so `trunk_lateral` splits the difference.

The reports are `deliver/two_characters_take1_rig_adult_validation.json` and `deliver/two_characters_take1_rig_stocky_validation.json`. The videos are `deliver/two_characters_take1.mp4` (both characters, version-1.0 FK bake), `deliver/two_characters_take1_ik.mp4` (frames 380–800 with the legs in IK: hop, crouch, kick 2) and `deliver/take1_side_by_side.mp4` (review grid of the version-1.0 bake).

The stand-in column comes from re-running §3 step 2 with the current code. `retargeted_take1.blend`, `retargeted_standin.blend` and their `_validation.json` in `/home/user/data/mocap/` are still the version-1.0 bakes; re-run §3 steps 2–3 to refresh them.

Feet and hands are reported but not gated:
- Planted feet are flattened on purpose (in the stand-in, monocular depth makes them toe-down).
- MediaPipe's pose hand points are noisy.
- Subsampling (`step` > 1) keys every N frames with linear interpolation, so locked feet drift between keys. Bake every frame for a controller.

## 6. Known limitations

- **The stand-in data is poor.** These are single-camera MediaPipe world landmarks: depth is compressed, frames seen from behind are garbage (`make_standin.py` flags 157 frames, the outlier rejection more), and the right foot hovers about 5 cm in the data. It is useful only as a stress test; the triangulated take is much cleaner (see §5).
- **No finger, toe or foot-roll data.** Toes stay flat; they bend only when a planted heel lifts. Hands are only roughly oriented from the index/pinky points. There is no foot inversion/eversion.
- **The twist of a fully straight limb** is inferred: minimal twist for arms, foot heading for legs.
- **Forced tradeoffs.** Joint limits, the hinge, foot lock, heel lift and the trunk correction trade direction fidelity for plausibility. All of them are parameters.
- **Proportions are not motion-adapted.**
  - Joint angles are copied, so contacts that depend on proportions are not kept: hands on knees or hips, hands on the floor, arms against a wide torso. The stocky character's long arms can pass through its big torso or its short legs in a crouch.
  - Per-limb offsets fix constant differences only.
  - The root path is scaled by the leg ratio, so two characters side by side drift apart or together along their path.
- **Rigify-specific assumptions.** The controls are the standard human metarig's, with the solver forcing `neck_follow = head_follow = 1` and `FK_limb_follow = 0`. The IK modes need the standard IK limb controls (`hand_ik`, `foot_ik`, `toe_ik`, `upper_arm_ik_target`, `thigh_ik_target`, `*_parent` property bones); without them the bake warns and keys FK only. A Root bone you animate yourself would move the IK controls (they are keyed in Root space).
- **IK limbs lose the forearm / shin twist and the hyper-extension** (5c): an inherent limit of Rigify's IK chain, which is why arms stay FK.
- **Live mode** does no temporal processing.
- **Embedded add-on.** The panel in a delivered `.blend` runs only if Python auto-run is allowed for that file. The embedded copy is a snapshot of `retarget.py` at save time; re-run `make_two_characters.py` or `retarget.py --embed`, or call `embed_text_module()`, to update it.
- **Sparse take1 frames.** The current `take1_joints3d.npz` has all core joints valid in only about 412–510 of 1177 frames (depending on filtering). The crouch (frames 465–495) and kick 3 (835–865) have almost no leg samples. Knees are inferred where the hip and ankle exist; kick 3 cannot be recovered from this npz, and kick 4 is barely lifted in the data.
