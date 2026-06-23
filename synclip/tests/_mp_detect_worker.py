"""Subprocess worker: render expressions and report MediaPipe detection.

Run as a module (``python -m synclip.tests._mp_detect_worker`` from the
``tools`` directory) so MediaPipe's native libraries load in a throwaway
process.  Keeping them out of the main pytest interpreter matters: once loaded
they inflate the process image (slowing the ``fork()``-heavy IPC tests) and have
been seen to deadlock on interpreter shutdown.  This worker isolates all of that.

Renders a handful of expressions through SoftwareRenderer, runs MediaPipe on
each, and prints one JSON object ``{expression: detected_bool}`` on the last
stdout line.

Exit codes:
    0  ran; see JSON for per-expression results
    3  prerequisites missing (no head.glb, or MediaPipe runtime won't load)
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    glb = os.path.join(
        os.path.dirname(__file__), "..", "..", "godot", "head.glb"
    )
    if not os.path.isfile(glb):
        print("unavailable: head.glb not found")
        return 3

    from .. import head_mesh, retarget
    from ..arkit_names import BLENDSHAPE_NAMES
    from ..offscreen_renderer import SoftwareRenderer

    try:
        landmarker = retarget._build_landmarker()
    except RuntimeError as exc:
        print(f"unavailable: {exc}")
        return 3

    def expression(**names) -> list[float]:
        w = [0.0] * 52
        for name, value in names.items():
            w[BLENDSHAPE_NAMES.index(name)] = value
        return w

    cases = {
        "neutral": expression(),
        "jawOpen": expression(jawOpen=0.8),
        "smile": expression(mouthSmileLeft=0.7, mouthSmileRight=0.7),
        "blink": expression(eyeBlinkLeft=1.0, eyeBlinkRight=1.0),
        "pucker": expression(mouthPucker=0.9),
        # The smaller render a fast retarget might hand MediaPipe.
        "low_res": expression(),
    }
    sizes = {"low_res": 192}

    mesh = head_mesh.load_head_mesh(glb)
    results: dict[str, bool] = {}
    try:
        for name, weights in cases.items():
            px = sizes.get(name, 384)
            scores = retarget._run_mediapipe(
                landmarker, SoftwareRenderer(mesh, px, px).render(weights)
            )
            results[name] = bool(scores) and len(scores) == 52
    finally:
        try:
            landmarker.close()
        except Exception:
            pass

    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
