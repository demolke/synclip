# capmo — multi-phone capture & 3D pose

A sibling of `synclip`: receive video from an arbitrary number of phones,
record it, calibrate the cameras, estimate 2D pose per view, fuse into a 3D
pose, and drive a character. Built **separate** from synclip for now; the
pieces that turn out generic are written so they can later be lifted into a
shared layer and synclip can do both.

Same stack as synclip (Python 3.11 + PySide6 + OpenCV/MediaPipe) so a future
merge is natural.

## Phone protocol (already shipped on the phones — `streamcam` Android app)

The phone is the **TCP client**; capmo is the **server**. See `protocol.py`.

- **server → phone**: newline text commands —
  `PING <t0>`, `START <w> <h>`, `STOP`, `FRAMERATE <fps>`, `IFRAME <s>`,
  `FOCUS <x> <y>`, `ZOOM <r>`.
- **phone → server**: framed binary `[type:u8][len:i32 BE][payload]` —
  - `0x01 VIDEO` raw H.264 (Annex-B, Baseline) access unit
  - `0x02 META`  `i64` phone wall-clock ms, sent right before each keyframe
  - `0x03 SYNC`  `t0,t1,t2` reply to a PING → NTP clock offset

Known protocol gaps (need a tiny phone-side change later, kept generic):
- The phone does **not** transmit its device id yet (commented out in the app),
  so capmo keys cameras by peer IP. Sending the name would make identity robust.
- META (timestamp) is sent **only on keyframes**. capmo records the server
  arrival time of *every* frame and the phone capture time of *keyframes*; with
  the clock offset that anchors the whole stream. If we want exact per-frame
  capture times, have the phone send META per frame.

## Architecture: core / tools / ui

```
 phones --TCP--> server / recorder / decode ---> session dir (.h264 + timing + manifest)
                 (live capture + preview)               |
                        ^                               |
                        |                               v
 ui/ (Qt) ---------- RECORD: grid, Record        capmo.core  (no Qt; the shared library)
   main_window       Board check -------------->   board, board_coverage (live tracker)
                     Process... / View ---------->  pipeline: session -> keypoints
                     (QThread worker)                         -> rig -> joints3d
                                                    pose, session_io, timeline, h264,
 tools/* (CLI) ---- argparse + console ------->     joints3d, sync/, presets
                                                        |
                                                        v  uses
                                         solvers: multiview, selfcal, motion,
                                         calibration, tools/{calibrate_board,
                                         reconstruct_take} internals
                                         character: retarget, animate, skinning
```

* **The UI records and checks; it never solves.** It shows the live cameras,
  writes session directories and, with **Board check** on, runs the shared
  ChArUco detector on the preview to tell the operator whether the board segment
  will be good enough for `calibrate_board` (per-camera image coverage,
  distinct views, tilt, sharpness, simultaneous views with the other cameras;
  amber tile → green, plus hints such as "cover the top-left" or "show to camC
  together with camA"). Recording with it on stores the board config and that
  verdict (`board`, `board_check`) in the session manifest.
* **All processing is one library, `capmo.core.pipeline`**, used by the CLI
  tools and by the UI's **Process…** / **View**: session → MediaPipe keypoints
  (`extract_keypoints`) → a rig (`calibrate_board` on the board segment when the
  manifest names a board, else body self-calibration) → `reconstruct_take` →
  the 33-joint `joints3d` track (`core/joints3d.py`), which `animate` /
  `retarget` consume directly (Z-up file → Y-up character).
* **`capmo/tools/*`** = argument parsing + console output around those library
  calls (each tool's `run()` / `main()` is the same code the pipeline calls).
* One implementation per concern: board model + detection + view metrics
  (`core/board.py`, also used by `extract_keypoints --calib` and
  `calibrate_board`), MediaPipe wrapper + model files (`core/pose.py`), sync
  helpers grouped by purpose under `core/sync/` (live NTP clock, nearest-frame
  matching, session-clock frame offsets, body-motion and board-motion offsets),
  one triangulation (`multiview.py`).

## Layout

| module | role | deps | tested |
|---|---|---|---|
| `protocol.py` | wire framing, commands, H.264 NAL/keyframe helpers (phone protocol, unchanged) | stdlib | ✅ |
| `server.py`   | threaded TCP server, clock sync, frame dispatch | stdlib | ✅ |
| `recorder.py` | server↔session glue, arm/disarm, frame fan-out, clock sampling, `update_info` | stdlib | ✅ |
| `session.py`  | record raw `.h264` + `.index.jsonl` + `.clock.jsonl` + `manifest.json` | stdlib | ✅ |
| `decode.py`   | live H.264 → frames (PyAV), per-phone decode threads, latest frame + arrival time | av, numpy | ✅ |
| `core/pipeline.py` | **the offline library**: `extract_keypoints`, `calibrate_board`, `reconstruct_take`, `process_session` (whole chain, progress/cancel), `triangulate_poses` (in-memory, calibrated) | (below) | ✅ |
| `core/board.py` | ChArUco `BoardConfig`, `detect`, `BoardDetector` (layout autodetect), coverage / tilt / degeneracy / sharpness metrics + thresholds | opencv | ✅ |
| `core/board_coverage.py` | `LiveBoardTracker`: per-camera coverage, views, tilt, blur, simultaneous views, idle/scanning/good + hints | numpy, opencv | ✅ |
| `core/pose.py` | the MediaPipe PoseLandmarker wrapper, model download/cache, full-res mapping (`capmo.pose` is an alias) | mediapipe | ✅* |
| `core/session_io.py`, `core/timeline.py`, `core/h264.py` | session reader (schema 1/2), per-frame host capture times, SPS parsing | numpy | ✅ |
| `core/joints3d.py` | the `<take>_joints3d.npz` contract, Z-up → Y-up for the character | numpy | ✅ |
| `core/sync/` | `clock` (NTP offsets; `capmo.timesync` alias), `frames` (nearest-frame matching), `timeline` (session frame offsets), `body` (epipolar offsets, from selfcal), `board_motion` (board-speed offsets) | numpy | ✅ |
| `core/presets.py` | capture presets shared by UI and CLI | — | — |
| `calibration.py` | intrinsics accumulator, PnP, `RigCalibration` (`rig.json`), `describe_rig()` | opencv, numpy | ✅ |
| `geometry.py` | DLT triangulation + reprojection error (board tests) | numpy | ✅ |
| `multiview.py` | lens-aware camera, batched DLT, L/R-swap repair, ghost-detection consensus, leave-worst-view-out triangulation — the one triangulation | numpy, opencv | ✅ |
| `selfcal.py` | rig self-calibration from body keypoints (sync scan, E/PnP init, sparse BA, floor/Z-up/metric world frame, rig.json I/O) | numpy, scipy, opencv | ✅ |
| `motion.py` | 3D track cleanup: chirality fix, outlier rejection, gap fill, zero-phase low-pass, constant bone lengths | numpy, scipy | ✅ |
| `synth.py`    | pinhole flat-colour rasterizer; ChArUco board as grey quads (test fixtures) | opencv, numpy | ✅ |
| `retarget.py` | joints3d track → humanoid character pose (root + per-bone rotations); FK + clip export | numpy | ✅ |
| `character_rig.py` | bind the retarget armature to a vendored character rig (glTF bind pose + mesh load); proportion-aware | numpy | ✅ |
| `skinning.py` | linear blend skinning of the rig mesh (glTF skin) into a pose; ground-truth posed joints | numpy | ✅ |
| `animate.py` | drive the rig from a joints3d track (per-bone node rotations) + software character render | numpy | ✅ |
| `export_glb.py` | library: bake a retarget track + calibrated camera nodes to glTF/GLB (rig-check); not called by the UI or any tool yet | numpy | ✅ |
| `tools/record.py` | CLI recorder (start/stop by keys, signal, HTTP) | — | ✅ |
| `tools/session_to_take.py` | session → host-synced MP4/MKV | av | ✅ |
| `tools/extract_keypoints.py` | CLI + video/session sources: keypoints / board detections → `kp/<take>/` | av, (mediapipe) | ✅ |
| `tools/calibrate_board.py` | CLI + board solver: detections → metric `rig.json` | opencv, scipy | ✅ |
| `tools/reconstruct_take.py` | CLI + reconstruction: keypoint take → `<take>_joints3d.npz` (+ raw, rig.json, report, validation images) | (above) | ✅ |
| `main.py`     | entry: `--headless` (= `tools.record`) or the Qt UI | — | — |
| `ui/main_window.py` | dynamic camera grid (live video) + record + Board check + Process…/View (worker thread) | PySide6 | smoke |
| `ui/character_view.py` | Qt playback of the animated character (software-rendered frames) | PySide6 | smoke |

Retired in Phase B: `process.py` (old per-instant offline pipeline), `sync.py`
(moved to `core/sync/frames.py`), `reconstruct.py` (superseded by
`multiview.py`) and the UI's live intrinsics/extrinsics solve (`rig.json` now
comes only from `calibrate_board` or self-calibration).

`✅*` pose: logic unit-tested via an injected backend. A full **end-to-end**
test (`tests/test_e2e_render_pose.py`) also closes the real loop — it renders the
vendored character with the software rasterizer (Lambert-shaded), runs the real
MediaPipe PoseLandmarker on the renders, and triangulates the skeleton back out;
it skips cleanly where MediaPipe/GL libs are absent (as in synclip's headless
tests). Both E2E tests fuse the views with `core.pipeline.triangulate_poses`. A second, **un-gated** full-stack test (`tests/test_e2e_posed_reconstruction.py`)
goes further: it *skins* the rig into a known **pose** (`skinning.py`), renders it
from three cameras, runs MediaPipe on its CPU delegate (no GL), reconstructs the
3D skeleton, and **measures the error vs the posed ground truth** — reprojection
residual, Procrustes-aligned joint error, per-limb direction error, and elbow/knee
flexion recovery. (Limb-direction/flexion are the offset-invariant metrics: the
larger upper-arm/elbow error is a fixed MediaPipe-acromion vs rig-bone offset, not
noise.)

Everything except `ui/` is headless and unit-tested (`capmo/tests/`).

Required dependency: `pip install -r capmo/requirements.txt` (PyAV) — it decodes
the phones' H.264, so the app refuses to start without it (the unit tests inject
a fake decoder, so they don't need it).

## Recording format (stored *as-is*, no re-encode)

```
<session>/<cam>.h264          raw H.264 access units, concatenated
<session>/<cam>.index.jsonl   per-frame: {n, offset, size, recv_ms, key, meta_ms, server_ms}
<session>/<cam>.clock.jsonl   clock offset + rtt samples (~1/s)
<session>/manifest.json       phones, clock offsets, counts, timings, board (+ board_check verdict)
```

`server_ms` is the frame's capture time on the **server timeline** (so frames
from different phones line up). The `.index` lets the offline tools seek any
frame without scanning.

## Run it now

```
# headless: record from real phones, print live clock-sync + fps (= capmo.tools.record)
python -m capmo --headless --record ./captures --start

# Qt UI: camera grid + Record, live Board check, Process… (offline pipeline) and View
python -m capmo --record ./captures --board-squares 7x5 --board-square-len 0.10 --board-marker-len 0.075
```

In the UI: switch **Board check** on, walk the board through the volume until
every tile is green (the readout says what is missing), keep it on and press
**● Record**, show the board for a few more seconds, put it away and perform,
stop. **Process…** runs the offline pipeline on that recording in the
background (`<session>/processed/`: `kp/<name>/`, `rig.json` from the board or
the body, `mocap/<name>_joints3d.npz`); **View** plays the result on the
character (processing first if needed). The same from a script:

```python
from capmo.core.pipeline import process_session
res = process_session("captures/session_20261028_101500")   # rig=..., model="heavy", progress=...
track = res.joints3d()                                       # (T,33,3) Z-up metres
```

The CLI tools (`capmo/tools/README.md`) expose every step with all its options.

Point each phone's StreamCam at the capmo host's IP and the chosen port.

Tests: `./run_tests.sh` runs everything (headless Qt). The slowest end-to-end
tests are marked `slow` (~2 min of the ~3 min total); for a fast loop use
`./run_tests.sh -m "not slow"` (about 1 minute).

## Roadmap

1. **Capture**: server, clock sync, dynamic grid, raw recording. ✅
2. **Live decode**: per-phone H.264 → frames for the grid, via PyAV. ✅
3. **Calibration**: ChArUco board segment at the start of a take →
   `calibrate_board` (intrinsics per camera, simultaneous views → camera graph,
   bundle adjustment, metric `rig.json`), or body self-calibration
   (`selfcal`). The UI only checks the board coverage live
   (`LiveBoardTracker`); the solve is offline. ✅
4. **Process**: MediaPipe PoseLandmarker per view (`extract_keypoints`), per-camera
   timelines resampled onto one grid with the clock-based sync as prior,
   robust multi-view triangulation + cleanup (`reconstruct_take`) into the
   33-joint `joints3d` track. `core.pipeline.process_session()` runs the whole
   chain on a session; validated end-to-end on fabricated sessions and the
   real take. ✅
5. **Retarget**: map the 3D skeleton track onto a humanoid armature — full 3-DOF
   root (pelvis position + facing from the hip/spine basis) and a minimal-swing
   rotation per bone, stored local to its parent. Unsolved joints hold the rest
   pose; the track helper fills gaps and optionally slerp-smooths. Forward
   kinematics inverts it exactly (the round-trip the tests assert), and a clip
   serialises to engine-ready JSON. ✅
6. **3D view**: animate a basic character driven by the reconstructed track. A CC0
   Quaternius rig is vendored in `data/characters/`, and `character_rig.py` maps
   our armature onto its skeleton on the rig's own proportions (bone rotations are
   proportion-free; the root translation is rescaled by torso ratio).
   `skinning.py` deforms the rig mesh into a pose (LBS over the glTF skin);
   `animate.py` bridges a reconstructed 3D skeleton track into per-rig-bone node
   rotations (parent-first swing, validated by a skinning round-trip) and renders
   the character with the software rasterizer; `ui/character_view.py` plays it
   back in the Qt window (the **View** button runs the pipeline → `animate_joints`
   → playback).
   The render is GL-free (software); a live OpenGL view is a drop-in future swap.
   (`export_glb.py` is a separate library module that can bake a retarget track
   plus the calibrated cameras to a GLB for a glTF viewer; nothing in the UI or
   the tools calls it.) ✅
7. **Modifiers** (maybe): smoothing/retarget on the 3D pose — the seam where
   capmo and synclip's pipelines could converge.

## Generic vs capmo-specific (for the eventual synclip merge)

- **Generic** (lift to a shared layer): `protocol` framing, `timesync`,
  the timestamped-`Stream` recording idea, the dynamic-grid widget,
  the "process recorded streams → output" pattern, `retarget`'s quaternion
  helpers (`quat_*`).
- **capmo-specific**: H.264 ingest, camera calibration, multi-view triangulation,
  the humanoid skeleton model and the position→bone-rotation retarget.
- synclip is the **server** broadcasting to viewers; capmo is the **server**
  receiving from phones — the framed-socket transport is the shared bit, the
  direction/role differs.
