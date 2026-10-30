# Recording: two paths into the offline tools

1. **capmo phones → session directory.** Phones running StreamCam stream to capmo
   (`capmo.tools.record` below, or the Qt UI `python -m capmo --record DIR`). A session
   stores each phone's H.264 verbatim plus per-frame timing and the clock offsets, so
   the cameras are synchronised to the PC clock. `extract_keypoints` reads a session
   directory directly; `session_to_take` turns it into ordinary MP4/MKV files whose
   timestamps are on that common clock. The UI's **Board check** shows live, per
   camera, whether the board segment will satisfy `calibrate_board` (see "Live board
   check" below); its **Process…** runs the chain below on the last recording.
2. **Any other cameras → video files**, one per camera; the offsets between them are
   estimated from the motion by `reconstruct_take`.

```
record ─► SESSION_DIR ─┬─► extract_keypoints SESSION_DIR ─► kp/take1 ─► calibrate_board / reconstruct_take ─► Blender
                       └─► session_to_take ─► videos/*.mp4 (synced) ─┘ (also accepted by extract_keypoints)
camera files ──────────────────────────────► extract_keypoints a.mp4 b.mp4 ...
```

Every tool is a thin command line around the shared library `capmo.core.pipeline`
(`extract_keypoints()`, `calibrate_board()`, `reconstruct_take()`, and
`process_session()` for the whole chain on one session); the Qt UI calls the same
functions, so a take processed from the UI and from the commands below is the same
computation.

# The whole chain from Python / the UI (`capmo.core.pipeline.process_session`)

```python
from capmo.core.pipeline import process_session
res = process_session("captures/session_20261028_101500",   # work_dir=<session>/processed
                      model="heavy",            # MediaPipe model (or estimator=...)
                      rig=None,                 # an existing rig.json skips the board step
                      progress=lambda stage, detail, frac: print(stage, detail, frac))
print(res.rig_source, res.joints3d_path, res.warnings)
```

1. Pose keypoints of every camera over the whole session → `processed/kp/<name>/`.
2. If no `rig` is given and the manifest names a board (the UI's Board check was on,
   or `record --board-*`): board detections (`--calib-every` ≈ 10 per second, `>= 8`
   corners) and `calibrate_board` → `processed/rig.json` + `rig_report.json`. If that
   fails (no board seen, a camera never linked ...), the warning is kept and the
   cameras are self-calibrated from the body instead.
3. `reconstruct_take` → `processed/mocap/<name>_joints3d.npz` (+ report, `rig.json`).
   No `--start` trimming: frames without a performer are simply not triangulated.

Session cameras missing from a given `rig` are left out (with a warning). `progress`
is called at each stage and camera; raising from it (`Cancelled`) stops the run,
which is how the UI's worker thread is cancelled. For finer control (board window,
`--start`, model per step, overlays) run the tools below.

# Live board check (Qt UI)

`capmo.core.board_coverage.LiveBoardTracker` runs on the preview frames (the same
ChArUco detector and metrics as `extract_keypoints --calib` / `calibrate_board`,
from `capmo.core.board`). A tile turns green when its camera has, as in the
`calibrate_board` report table below: coverage >= 0.5 of the 8x6 image grid, >= 15
distinct sharp views (>= 8 corners, the board moved since the last one), a max tilt
>= 20 deg, and — with two or more cameras — >= 3 simultaneous views (frames within
60 ms) linking it to the first camera through the camera graph. Otherwise the tile
stays amber and names what is missing: "cover the top-left", "tilt more", "more
board positions (7/15)", "hold the board still (blurry)", "show to camC together
with camA". Nothing is solved live; recording with the check on stores the board and
the verdict in the manifest (`board`, `board_check`).

# capmo.tools.record

Headless recorder (no Qt): capture server + session writer + start/stop control.

```bash
python -m capmo.tools.record --out captures --start --preset 1080p30
```

* Phones dial `<PC IP>:12345` (`--host`/`--port`); `--start` makes each phone stream at
  the preset (`480p30 720p30 720p60 1080p30 1080p60`) or `--width/--height/--fps`.
* Recording starts at once (`--wait`: idle until toggled). Toggle with **Enter** or
  `s`+Enter (when stdin is a terminal; `q` quits), `kill -USR1 <pid>`, or HTTP with
  `--control-port 8765` (bound to `--control-host`, default 127.0.0.1):
  `curl -X POST localhost:8765/start`, `/stop`, `/toggle`, `/quit`, `curl localhost:8765/status`
  (JSON: recording, session dir, per-phone fps / frames / clock offset / rtt).
* `--duration S` stops after S seconds of recording and exits; Ctrl+C / SIGTERM stop
  cleanly (the manifest is always written). `--convert` runs `session_to_take` into
  `<session>/videos/` after each recording. `--board-squares 7x5 --board-square-len 0.10
  --board-marker-len 0.075` records which ChArUco board was shown (used by
  `extract_keypoints --calib` on the session).
* A status line per phone every 2 s, as before. `python -m capmo --headless --record DIR`
  runs the same code.

Session directory (`session_YYYYMMDD_HHMMSS/`, a new one per recording):
`<phone>.h264` (the stream, verbatim), `<phone>.index.jsonl` (per frame: byte offset,
size, arrival time, keyframe flag, phone capture time on keyframes), `<phone>.clock.jsonl`
(clock offset + round-trip samples, ~1/s) and `manifest.json` (schema 2: per camera
`cam_id`, size from the SPS, nominal / measured fps, clock summary, codec config; the
session's preset, fps and board). Older sessions (schema 1) are still read.

Per-frame times: the phone's capture time is known only on keyframes; the frames in
between are spaced evenly, with frames the encoder dropped placed where the arrival
times show a gap, then mapped to the PC clock with a robust offset (low-rtt clock
samples). On a synthetic stream this is 0.75 ms RMS / 2.3 ms max off the true capture
time (arrival times alone: ~25 ms RMS); add the clock error, bounded by rtt/2 per phone.

# capmo.tools.session_to_take

```bash
python -m capmo.tools.session_to_take captures/session_20261028_101500 --out videos/take1 \
    [--cam-ids camA,camB,camC] [--format mp4|mkv] [--no-verify]
```

Remuxes each camera **without re-encoding** into `<cam_id>.mp4` with PTS = capture time
on the PC clock minus a common session origin (the first decodable frame of any
camera), so equal timestamps in different files are the same instant. Frames before the
first keyframe are dropped (undecodable), missing SPS/PPS are restored from the
manifest, dropped frames stay gaps (variable frame rate). `take.json` lists per camera
the clock offset, timing method, dropped frames and the ffprobe PTS check, and a sync
uncertainty estimate for the worst camera pair. `--cam-ids a,b,c` renames in manifest
order; `slug=a,...` maps and selects.

---

# capmo.tools.extract_keypoints

Runs the heavy per-frame work (MediaPipe pose, ChArUco detection) on your own
machine for ordinary video files (MP4/MOV, phone footage). Output is a few MB of
`.npz` you upload for 3D reconstruction. Phone rotation metadata is applied, so
keypoints match the orientation you see in a player; pixels are always in the
video's full resolution.

## Setup (once; Python 3.10-3.13)

```bash
git clone <repo> synclip && cd synclip     # or unzip it; you only need the capmo/ folder
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install numpy av opencv-contrib-python mediapipe
```

That minimal set is enough (no Qt needed). The full stack
(`pip install -r capmo/requirements.txt -r synclip/requirements.txt`) also works
but is much larger. On Linux, MediaPipe needs `libGLESv2` (`sudo apt install libgles2`).

The first run downloads the chosen model (`lite`/`full`/`heavy`, ~6/9/30 MB) from
`storage.googleapis.com/mediapipe-models/...` and prints the exact URL.

## Run (from the repo root, one command per take, all 4 angles together)

```bash
python -m capmo.tools.extract_keypoints --out kp --take take1 --model heavy \
    --cam-ids camA,camB,camC,camD --keyframes 6 --proxy \
    take1_A.mp4 take1_B.mp4 take1_C.mp4 take1_D.mp4
python -m capmo.tools.extract_keypoints --out kp --take take2 --model heavy \
    --cam-ids camA,camB,camC,camD --keyframes 6 --proxy \
    take2_A.mp4 take2_B.mp4 take2_C.mp4 take2_D.mp4
```

Keep the same `--cam-ids` order (same physical camera) across takes. Options:
`--model lite|full|heavy` (default full), `--max-height 1080` (downscale for the model
only), `--every N` (skip frames; default 1 = all), `--keyframes K` (K JPEG thumbnails
per video), `--proxy` (480p H.264 MP4 per video), `--quiet`.

## A capmo session as input

```bash
python -m capmo.tools.extract_keypoints --out kp --take take1 --model heavy --proxy \
    --cam-ids camA,camB,camC,camD captures/session_20261028_101500
python -m capmo.tools.extract_keypoints --calib --start 0 --end 20 --calib-every 1 \
    --out kp --take take1 --cam-ids camA,camB,camC,camD captures/session_20261028_101500
```

Give one session directory instead of the video files. The raw H.264 is decoded
directly; every frame's `t` is its capture time on the PC clock relative to the session
origin (identical to the `session_to_take` PTS), and `--start/--end` are in that time,
the same instant for every camera. Camera ids come from the manifest (`--cam-ids`
renames in manifest order). In `--calib` mode the board stored in the manifest fills in
board options that are not given. `meta.json` gets `"timebase": "session"` (`"video"`
for video files) and the
session's clock/sync info, which `reconstruct_take` uses.

## Board window and pose window (`--start` / `--end`)

`--start S --end E` (seconds of video time) restrict either mode to a window; the
demuxer seeks to `S`, so skipping a long lead-in is cheap. `t` and `frame_idx` in
the output always count from the start of the *video*, so a calibration window and a
pose extraction of the same files stay comparable (for variable-frame-rate phone
video, `frame_idx` after a seek is derived from the timestamp: `round(t * fps)`).
*Video time* is the frame's presentation timestamp (the clock of the npz `t`): it starts
at 0 for ordinary MP4/MOV/AVI files, at the container's start time for MKV/TS files or
MP4s with an edit list (`meta.json` keeps it as `start_time`), and is the session clock
for `session_to_take` videos (so one window is the same instant for every camera).
`calibrate_board` and `reconstruct_take` interpret `--start/--end` on the same clock.

Each run **replaces** the camera list of its kind in `meta.json` (`videos` for pose,
`calib_videos` for `--calib`) and always writes `timebase` (`"video"` or `"session"`):
cameras of an earlier run that this run did not produce are dropped and their npz
deleted, and if the time base changed (plain videos into a take that held a session
extraction, or the reverse) the other kind's cameras and the `session` block go too.

```bash
# board segment (the first ~20 s): every frame of the window, keep views with >= 6 corners
python -m capmo.tools.extract_keypoints --calib --start 0 --end 20 --calib-every 1 \
    --min-corners 6 --out kp --take take1 --cam-ids camA,camB,camC,camD \
    take1_A.mp4 take1_B.mp4 take1_C.mp4 take1_D.mp4
```

Calibration-mode options: `--calib-every N` (every Nth frame of the window; 1 = all,
recommended for the board segment), otherwise `--calib-samples 200` frames spread
evenly over the window (the old default, whole video); `--min-corners 4`. Board
defaults: 8x5 squares, 40 mm squares, 20 mm markers, `DICT_6X6_250`; override with
`--squares-x --squares-y --square-len --marker-len --dict` (lengths in metres, as
*measured* on the print). The legacy/new ChArUco layout is autodetected. Output:
`kp/<take>/<cam>.calib.npz` (default take name `calib`), next to the pose `<cam>.npz`
of the same take if you use the same `--take`. It is turned into a `rig.json` by
`capmo.tools.calibrate_board` (below).

## What to upload

Zip the whole output folder:

```bash
zip -r kp.zip kp          # Windows: Compress-Archive kp kp.zip
```

Contents: `kp/<take>/<cam>.npz` (kp, vis, world, t, frame_idx, width, height, fps,
source), `meta.json`, optional `*.calib.npz`, `keyframes/*.jpg`, `*.proxy.mp4`.

Expected size: each pose `.npz` is ~0.8 KB per frame (about 0.7 MB for a 30 s,
30 fps clip, ~3 MB per take of 4 cameras). Calibration files are ~50-300 KB.
Proxies add ~1-3 MB per video, thumbnails ~30 KB each. Whole upload: roughly 10-40 MB
with proxies, under 5 MB without.

## Speed

`full` runs roughly 20-50 frames/s on a laptop CPU for 1080p input; `heavy` is
several times slower but more accurate. Use `--every 2` to halve the time.

---

# capmo.tools.reconstruct_take

Turns an uploaded keypoint take (`kp/<take>/`) into a 3D landmark track. With no
calibration it **self-calibrates the cameras from the performer's body**; pass
`--rig rig.json` (from `calibrate_board`, or the `rig.json` of an earlier
self-calibrated take with the same camera placement) to skip that.

```bash
pip install scipy matplotlib        # on top of the extractor's set
python -m capmo.tools.reconstruct_take kp/take1 --out mocap            # self-calibrate
python -m capmo.tools.reconstruct_take kp/take2 --out mocap2 --rig mocap/rig.json
# board rig + drop the board segment (video time < 20 s) + sanity check vs self-calibration
python -m capmo.tools.reconstruct_take kp/take1 --out mocap --rig rig.json --start 20 \
    --compare-selfcal
```

`--start S` / `--end E` (seconds of video time, the clock of the npz `t`) trim the take
**before** anything runs: the sync scan, the self-calibration / BA stride and all the
outputs only see the kept frames. In the output npz `t` always starts at 0 (the first
output row, frame 1 in Blender; trimmed or not), `t_source` keeps the take's time (video
time, or session time for a session take) and `frame_idx` the first camera's video frame
index of every row (`t_source` and `frame_idx` are always written). Trimming is
optional with `--rig` (frames without a performer are simply not triangulated), but it
keeps the board segment out of the animation and out of the floor/heading fit.

`--timeline auto|first|resample`: per-camera timestamps. A take extracted from a capmo
session (or any take whose cameras' `t` differ by more than half a frame) is resampled
onto one grid at the nominal fps, anchored on the first camera: each camera's 2D
keypoints are interpolated linearly between detected neighbours (≤ 2.5 frames apart),
else taken from the nearest frame within half a frame. The clock-based alignment is
then the prior of the epipolar offset scan (centre of the scan, and the fallback for a
camera with no reliable pair), refined as usual. `first` is the legacy behaviour (rows =
frame numbers, the first camera's `t`), which `auto` keeps for ordinary videos.

`--compare-selfcal` (with `--rig`): also self-calibrates from the body and reports, per
camera, the centre distance (m), rotation difference (deg) and focal difference (%)
against the rig after the best similarity (`report.json` → `compare_selfcal`). A few
cm / < 1-2 deg means the board rig and the take agree; large differences point at a
moved camera, a wrong `--cam-ids` order or a bad board calibration.

Pipeline: time-offset scan (epipolar, then a calibrated reprojection check) →
initial cameras (focal from `--hfov`, centred principal point, no distortion;
essential matrix for the best pair, PnP for the rest) → sparse bundle adjustment
(shared focal by default) → world frame → robust triangulation (ghost-detection
consensus, per-part L/R swap repair, leave-worst-view-out, ≥2 views) → cleanup
(chirality fix, outlier rejection, ≤6-frame gap fill, zero-phase Butterworth
4/6/7 Hz for torso/elbows+knees/extremities, constant symmetric bone lengths).

World frame (also the frame of the written `rig.json`): metres, right-handed,
**+Z up, floor at z = 0** (plane fitted to the lowest foot landmark per frame),
origin under the mean pelvis, the performer's **average facing along −Y**
(Blender front view; their left is +X). Scale: the median upper-arm + forearm +
thigh + shin lengths are matched to an adult reference (1.37 m total, ≈1.72 m
stature); `--scale shoulder --shoulder-width 0.38` uses the shoulder width instead.
With `--rig` the rig's own metric scale is kept, and the rig is re-expressed in this
world frame (the board frame of `calibrate_board` is only an intermediate).

Outputs in `--out`: `<take>_joints3d.npz` (`joints` (T,33,3) float32,
`valid` (T,33) bool, `fps`, `t` (T,), `names` (33,) MediaPipe names; `joints` is
finite everywhere, samples bridged over long gaps have `valid=False`),
`<take>_joints3d_raw.npz` (triangulation only, nan where unsolved), `rig.json`,
`report.json` (sync curves, BA/reprojection RMS, bone CV, view coverage, jitter,
floor contact, left/right checks) and `validation/` images (reprojection
overlays on the proxies, 3D skeleton frames, camera layout, time series).

---

# capmo.tools.calibrate_board

Turns the ChArUco detections of a take (`kp/<take>/<cam>.calib.npz`) into a metric
`rig.json` for `reconstruct_take --rig`.

```bash
python -m capmo.tools.calibrate_board kp/take1 --out rig.json --report rig_report.json \
    --overlay-dir rig_check
```

What it does:

1. **Intrinsics per camera.** Board views with `>= --min-corners` (8) corners that are
   not a single row; a well-spread subset of at most `--max-views` (40) views is picked
   (image position, board size, tilt) and solved with `cv2.calibrateCamera`; views with
   an RMS far above the rest are dropped and the solve repeated. `--dist` picks the
   distortion model: `none`, `k1`, `k1k2`, `k1k2p` (k1, k2, p1, p2; default), `k1k2pk3`.
   `--fix-principal-point` (centre) and `--fix-aspect` (fx = fy) help when the board
   covered little of the image; so does `--ba-intrinsics` (step 4), which refines the
   intrinsics with the multi-camera constraints. `--intrinsics-only camA=K.json` takes a camera's
   intrinsics from a file (`{"K", "dist"}` or a `rig.json`; `--save-intrinsics DIR`
   writes them) and only solves its pose.
2. **Simultaneous views.** Views of different cameras with the same `frame_idx` (within
   `--sync-tol` 0.5 frames) are the same instant. Unsynchronised cameras: give the
   offsets with `--offsets camB=+2,camC=-1` (camera X's frame `k + offset` shows the
   reference camera's frame `k`), or `--estimate-offsets` (cross-correlates the 3D speed
   of the board centre seen by each camera; needs `--calib-every 1`-dense detections and
   a moving board). A take extracted from a **capmo session** needs neither: each
   camera's `frame_idx` counts from its own first frame, so the offsets are taken from
   the common PC clock (`t`) automatically (`offsets_source: "session clock"` in the
   report). Sub-frame offsets remain, so *hold the board still* for a moment in each
   position.
3. **Camera graph.** Per simultaneous view, PnP in each camera gives each pair's
   relative pose; the samples are averaged robustly. Pairs with `>= --min-shared` (3)
   agreeing views are edges; cameras are chained from the reference (`--ref-cam`,
   default the first) along a maximum spanning tree, so a camera that never sees the
   board together with the reference is placed through the others.
4. **Bundle adjustment** of all camera poses and one board pose per simultaneous view
   (board-corner reprojection, Huber loss, sparse Jacobian; `--ba-intrinsics` also
   refines the board-solved intrinsics). The board fixes the metric scale.
5. **World frame** = the board at one reference instant (`--world-frame FRAME`; default:
   the instant seen by the most cameras): origin at the board's outer corner next to
   ChArUco corner 0, +X along `squares_x`, +Y along `squares_y` (down the printed
   pattern), +Z into the board (away from the cameras), metres. `reconstruct_take`
   re-derives the floor / up / heading from the performer and keeps the scale.

Output: `rig.json` (the cameras plus a `board_calibration` block), a `describe_rig`
printout (camera positions, look directions, baselines), the `--report` JSON and, with
`--overlay-dir`, JPEGs of sample frames with the detected corners (green crosses) and
the BA-reprojected board (red circles). The videos are found through `meta.json`; pass
`--video camA=path.mp4` if they moved.

**Reading the report.** `summary` has one line per camera; details are under `cameras`,
`graph`, `ba` and `warnings`:

| value | good | what a bad value means |
|---|---|---|
| `intrinsic_rms_px` | < 0.5 px (< 1 px acceptable) | blur, rolling shutter, a bent board, a wrong `--square-len` |
| `n_views_intrinsics` | >= 15 | show the board longer / closer to this camera |
| `coverage` | > 0.5 of the image | move the board into the image corners, or rerun with `--fix-principal-point` and/or `--ba-intrinsics` (a small board far away mostly misplaces the principal point, which tilts the cameras by ~1-2 deg) |
| `tilt_deg.max` | 20-45 deg | the focal length is weakly determined without tilt |
| `ba_rms_px` (per camera) | < 1 px (< 2 px acceptable) | unsynchronised cameras or a moving board (offsets), wrong intrinsics |
| `graph.edges.*.spread_deg` | < 0.5 deg | the pair's PnP samples disagree: sync / motion blur |
| `graph.not_connected` | empty | a camera never shares the board with a linked camera |

A camera that never sees the board (or never together with a linked camera) is an
error that names it. Either record the board again, run `reconstruct_take` without
`--rig` (body self-calibration), or pass `--fallback-selfcal`: all cameras are
self-calibrated from the performer (`<cam>.npz` pose files of the take, or
`--pose-take DIR`, `--pose-start S` to skip the board segment), and that solution is
mapped onto the board-calibrated cameras with a similarity (needs >= 2 of them); the
missing cameras keep their body-calibrated pose and intrinsics (less accurate: expect
~1 deg / several cm), `fallback_selfcal.anchor_residuals` shows how well both agreed.

---

# Workflow: board at the start of the videos

Every take's videos *start* with the ChArUco board shown to the cameras for 10-30 s;
the performance follows in the same files. No separate calibration recording is needed
and every take carries its own calibration.

**Recording.**
- Board: print it flat and rigid (foam board / glued to a panel); size it so a square is
  at least ~20 px in the image at the working distance. For a room-scale rig (cameras
  3-5 m away) use e.g. 7x5 squares of 10 cm (A1/A2 print), not the A4 default:
  ```bash
  python -c "from capmo.calibration import BoardConfig; import cv2; \
  cv2.imwrite('board.png', BoardConfig(7, 5, 0.10, 0.075).board().generateImage((2300, 1700), marginSize=100))"
  ```
  Measure a printed square with a ruler and pass that (and the marker size) to the
  extractor (`--squares-x 7 --squares-y 5 --square-len 0.10 --marker-len 0.075`).
- Start all cameras, then walk the board through the capture volume: keep it visible
  to **at least two cameras at once**, hold it **still for ~1 s** in each position
  (the cameras are not frame-synchronised), **tilt** it 20-45 deg in different
  directions, and bring it close to every camera so it also covers the **image corners
  and edges**. Make sure every pair of neighbouring cameras sees it together several
  times. Then put it down (out of view), and perform. Note the time the board is gone.
- Same frame rate everywhere; lock focus/exposure/zoom (intrinsics must not change).

**Commands** (board gone at ~20 s):

```bash
# 1) board window -> ChArUco detections, pose (performance) window -> keypoints
python -m capmo.tools.extract_keypoints --calib --start 0 --end 20 --calib-every 1 \
    --squares-x 7 --squares-y 5 --square-len 0.10 --marker-len 0.075 \
    --out kp --take take1 --cam-ids camA,camB,camC,camD  take1_A.mp4 take1_B.mp4 take1_C.mp4 take1_D.mp4
python -m capmo.tools.extract_keypoints --start 20 --model heavy --proxy --keyframes 6 \
    --out kp --take take1 --cam-ids camA,camB,camC,camD  take1_A.mp4 take1_B.mp4 take1_C.mp4 take1_D.mp4
# 2) board -> rig.json (check the report: RMS, coverage, graph, warnings)
python -m capmo.tools.calibrate_board kp/take1 --out kp/take1/rig.json \
    --report kp/take1/rig_report.json --overlay-dir kp/take1/rig_check
# 3) 3D reconstruction with the board rig (--start 20 is a no-op here because the pose
#    extraction already began at 20 s; it matters if you extracted the whole video)
python -m capmo.tools.reconstruct_take kp/take1 --out mocap/take1 --rig kp/take1/rig.json \
    --start 20 --compare-selfcal
# 4) Blender: mocap/take1/take1_joints3d.npz -> blender/mocap_retarget (unchanged)
```

If the cameras did not start together, add `--estimate-offsets` (or `--offsets ...`) to
step 2; `reconstruct_take` still estimates and applies the body-based offsets of the
performance. If `calibrate_board` reports a camera it cannot place, see the fallback
above.

