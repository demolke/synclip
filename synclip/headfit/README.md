# headfit: fit the GNM head to a mesh

## Usage

Run from `synclip/synclip/`:

```
python fit_head.py MESH [-o OUT_DIR] [--metric] [--no-renders] [--landmark-weight W] [--prior L]
                        [--no-fit-expression] [--expr-prior L] [--no-detail] [--detail-smooth A]
                        [--no-lips] [--lip-weight W] [--guard-all-targets]
                        [--feature-detail {off,moderate}] [--no-ears] [-v]
```

- `MESH`: any trimesh-readable mesh (.glb/.obj/.ply/...). Only the geometry is used; morph targets are ignored. Orientation, up axis and units can be anything.
- `-o OUT_DIR`: defaults to `fit_out/<mesh stem>/`. It receives:
  - `<stem>_gnmfit.glb`: the fitted head, ready to import into Blender. It is written by the GNM viewer's exporter (`gnm_model.build_gltf_with_quads_and_facs`), so its structure matches a `gnm_head_viewer.py` "export full" (as one binary .glb instead of .gltf + .bin): the `GNM_Head` mesh with vertices split at UV seams, POSITION/NORMAL/TEXCOORD_0, the quad side channel in the mesh/node extras (run `gnm_blender_quads.py` in Blender to rebuild quads) and the 106 morph targets of `gnm_model.ALL_FACS` (ARKit 52 + FACS AUs + semantic) in that order. The base is the neutral fitted bind pose (template + detail offset D + identity, expression = 0, pose = 0). Every morph target is generated from that fitted base: GNM is evaluated with D added to the template and the template joints moved by `joint_regressor @ D` (`gnm_model.gen_verts(..., template_offset=D)`), so jaw/eye rotations skin and get pose correctives about the fitted joints.
  - `<stem>_gnmfit.json`: the identity (`id`, 253, eyes/teeth = 0), the fitted expression vector and its stats (`expression`, for reference only: it was solved together with `id_with_expression` on the plain GNM template, so adding it to the neutral head double-counts shape), the detail offset stats, the bind-pose `joint_positions` of the fitted head, the 170 head PCs, alignment transform/scale/confidence/fallback, the input-to-GNM 4x4, and two metric sets: `metrics` (neutral export vs target) and `metrics_with_expression` (joint identity+expression fit vs target).
  - `<stem>_viewer.json` (+ `<stem>_offset.npy`): the fit as a `gnm_head_viewer.py` parameter file, to tweak by hand there and export from the viewer. See "Editing the fit in the viewer" below.
  - Renders: `<stem>_gnmfit_compare.png` (target / neutral export / error heatmap, front and 3/4) and `<stem>_gnmfit_expr_compare.png` (the same for the with-expression fit).
- `--metric`: the input is already in metres. Only the rigid part of the alignment is applied, so the target keeps its true size and GNM's identity PCs absorb head size. Without it, the target is scaled to the size of the GNM mean head.
- `--landmark-weight W` (default 1, 0 disables) sets the initial weight of the landmark term. `--prior L` (default 1) multiplies the identity prior weight.
- `--no-fit-expression` turns off the joint identity + expression fit (and the shape recovery that follows it). `--expr-prior L` (default 0.1) is the weight on the mean squared face displacement (mm^2) caused by the expression; lower lets the expression explain more.
- `--no-detail` turns off the per-vertex template offset. `--detail-smooth A` (default 1) is its Laplacian smoothness weight; larger is smoother and smaller follows the target more closely.
- `--no-lips` fits the lips like the rest of the face (no lip terms in the offset). `--lip-weight W` (default 1) scales all lip terms (0 is the same as `--no-lips`).
- `--feature-detail {off,moderate}` (default moderate): feature-aware offset (curvature-adaptive smoothness and a face-wide two-sided term, see Method). `off` is the plain offset.
- `--no-ears`: do not fit the ears. They then only follow the head through the identity and the smoothness term.
- `--guard-all-targets`: the flip guard checks all 106 exported morph targets instead of jawOpen / smile / jawOpen+smile. Slower (~+30 s) and the lips come out slightly smoother.

The CLI exits non-zero if GNM is unavailable. It warns when alignment falls back to ICP or when the landmark confidence is low (< 0.5).

## Editing the fit in the viewer

`<stem>_viewer.json` uses the viewer's Save/Load format unchanged: `{"format": "gnm_params", "identity": [253], "expression": [383], "rotations": [[3] x 4], "translation": [3]}`, here with the fitted identity, expression 0 and a neutral pose. Open it with the viewer's Load [Ctrl+O] button.

The detail + lip offset D cannot be expressed in GNM parameters, so it is written to `<stem>_offset.npy` (float32 (V,3), ~210 KB; as JSON text it would be ~1 MB and slow to parse) and referenced by an extra key, `"template_offset": "<stem>_offset.npy"` (path relative to the JSON). When a loaded JSON has this key, the viewer passes D as `template_offset` to `gen_verts` and to the glTF exporter, shows "Template offset ACTIVE" under the Save/Load row, and writes the (absolute) offset path back on Save. "Clear offset" drops it (undoable). A JSON without the key loads exactly as before and clears any active offset. Loading the fit's JSON and pressing "EXPORT glTF - ALL FACS" to a `.glb` path reproduces `<stem>_gnmfit.glb` byte for byte (fit_head exports with the same float32 D).

D stays fixed while you move sliders. It was solved for the fitted identity, so it stays exact for expression and pose edits, and for small identity changes. Large identity changes put the fitted detail onto a different base shape, which can look wrong.

## Setup

```
git submodule set-url synclip/gnm https://github.com/google/GNM.git   # HTTPS instead of SSH
git submodule update --init synclip/gnm
pip install -r synclip/requirements.txt
sudo apt-get install libegl1      # needed by mediapipe (headless)
```

The MediaPipe models (`face_landmarker.task`, `blaze_face_short_range.tflite`) are downloaded on first use to `~/.cache/synclip/`. Set `SYNCLIP_CACHE` to change that location.

## Method

- **Canonicalise:** centre and RMS-normalise the mesh, estimate the bilateral symmetry plane (PCA candidates scored by mirror-ICP), and enumerate 8 candidate orientations around it.
- **Landmarks:** render each candidate, run MediaPipe FaceLandmarker, keep the most confident one, then re-render frontally at 0 and ±25 degrees of yaw and take the per-landmark median of the back-projected 3D points. The outer and inner lip contours are kept separately for the lip terms of the offset. The inner contour is not used for alignment. If no face is detected, fall back to similarity ICP against the GNM mean face.
- **Align:** compute an Umeyama similarity (with outlier rejection) from stable landmarks to the GNM mean landmarks, then run rigid point-to-plane ICP on the face region, regularised by the landmark pairs.
- **Fit identity:** solve for the 170 head identity PCs in closed form (Cholesky) using forward and reverse point-to-plane ICP on the GNM face mask (plus the ears, see below), a decaying landmark term and a Tikhonov prior. The PC count goes coarse to fine (10, 40, 170) and alternates with a rigid (no-scale) re-alignment of the target.
- **Expression (default on):** after the identity stages, solve the 170 head PCs jointly with the 350 lower-face and eye-region expression PCs (tongue and pupils stay at 0), still alternating with the rigid re-alignment. The expression prior is `L * mean_face ||B_expr e||^2` in mm^2, so it penalises the displacement the expression causes rather than the raw coefficients. This lets a target with a slight expression (closed lids, a frown, parted lips) be explained without bending identity. The log and JSON report the fitted expression's RMS and max face displacement and its largest components.
- **Shape recovery:** the export is neutral, so expression is then set to 0 and identity is refitted against the same target in the same rigid frame. Real shape that the expression absorbed goes back into identity (and then the offset) instead of being dropped. The neutral export is not expected to match the target where the target really has an expression.
- **Detail offset (default on):** a per-vertex template offset D (V,3) is solved as one sparse linear system per ICP iteration. It combines point-to-plane data on the face mask (plus the reverse term and a small point-to-point term), `A * |L D|^2` Laplacian smoothness over the whole mesh and a small damping term. Eyeballs translate rigidly with the mean offset of the surrounding lid/socket skin, and teeth, gums and tongue with the mouth sock. The joints move by `joint_regressor @ D`, so the eye joints follow the eyeballs.
- **Feature detail (default on, `fit.FeatureDetail`):** the identity PCs and a uniform Laplacian smooth out creases and ridges. The offset solve therefore scales the Laplacian weight of each face and lip vertex by `1 / (1 + (H / 40)^2)` (floor 0.1, softly eroded over 2 rings), where H is the target's mean curvature (1/m, 3 mm ball) under the vertex, and adds a face-wide two-sided term: 80k target samples each pull their nearest interior GNM face vertex onto their tangent plane (w 1, pairs over 8 mm or with disagreeing normals dropped). The lid margins (3 rings around the eye openings) keep the full weight because their sliver triangles would take the strain. Where the lip terms set a lower weight, that one is kept. On the two test heads this lowers the two-sided error on the nose, brows, nasolabial fold and chin by 15-35% at about the same edge strain. A stronger setting (H0 20, w 3) halves those errors again but raises the max edge strain to ~200%, so it is not offered.
- **Ears (default on, `fit.EarFit`):** the GNM `ears` group is outside the face mask, so without help the ears only follow the identity (p95 error 6-7 mm). Ear point-to-plane rows (GNM ear vertex -> closest target sample, and target sample -> its nearest GNM skin vertex when that is an ear vertex) are added to the identity solves (per-row weight 0.5 of a face row, pairs up to 12 mm) and to the offset solve (weight 1, up to 6 mm). A pair is only kept if the normals agree, the target point is closest to a GNM ear vertex rather than the scalp or cheek, and it is not in the worst 10%. A side with fewer than 40% of its outer ear vertices (helix, antihelix, lobe: > 8 mm from the rest of the skin) paired is treated as missing and gets no ear data. So ears hidden by hair or a cap, or cut off, fall back to the `--no-ears` behaviour instead of pulling the ear onto the scalp. The rigid re-alignment still uses the face only.
- **Lips:** the identity stages use point-to-plane data, so the GNM vermilion border can slide along the surface, and the landmark term has decayed by then. Without help the lips come out 25-35% too thin. The offset solve therefore (`fit.LipFit`): adds the lips, including the seam row next to the mouth sock, to the data mask; weights the data rows ×4 on the lips plus one ring; lowers the Laplacian weight there to ×0.1, blending back over 2 rings; adds point-to-point MediaPipe outer and inner lip contours (w 0.3) and a two-sided term from the target lip surface (inside the outer contour) to the nearest GNM lip vertex (w 1); and keeps the offsets of paired upper/lower seam-row vertices equal (w 1).
- **Flip guard:** a triangle counts as flipped if D flips it in the bind pose or in a posed mesh (jawOpen, smile, jawOpen+smile through `gnm_model`, the same path the exported morph targets take) compared with the same pose without D. These are mostly slivers at the commissures. Edges within 2 rings of a flipped triangle get an edge-strain penalty `w |D_i - D_j|^2 / l0^2`, and D is re-solved (up to 3 rounds, w ×10 each round). Any triangle still flipped after that has its patch moved rigidly. The log/JSON report D's max and face RMS, the edge strain p99/max, the remaining flip count (expected 0) and the posed flip counts.
- **Evaluate:** report GNM-to-target and target-to-GNM surface distances on the face mask, plus landmark RMSE, for both the neutral export and the with-expression fit. All of these are measured after the fit's rigid refinement, which is applied to both the mesh and the landmarks.
