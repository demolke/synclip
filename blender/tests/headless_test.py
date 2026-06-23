"""
Headless Blender harness

    SYNCLIP_OUT=/path/out.json blender --background --python headless_test.py

Builds a mesh carrying the 52 ARKit shape keys, then exercises the synclip
client's pure mapping + frame-parsing logic against a Python-packed frame:

Writes a JSON result to $SYNCLIP_OUT and exits 0 on success, non-zero on error.
The pytest wrapper compares the result; this file is standalone so it can also
be run by hand.
"""

import json
import os
import struct
import sys

import bpy

# Make the tools dir importable so we can import the client module.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from blender import client  # noqa: E402
from blender.client import ARKIT_NAMES  # noqa: E402

_STRUCT = struct.Struct("<I d 52f 3f 3f")
MODE_LIVE = 0xAF0002


def _build_mesh_with_shape_keys():
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    # Basis key first, then one key per ARKit channel.
    obj.shape_key_add(name="Basis")
    for name in ARKIT_NAMES:
        obj.shape_key_add(name=name)
    return obj


def main() -> int:
    out_path = os.environ.get("SYNCLIP_OUT", "")
    result = {"ok": False, "errors": []}

    obj = _build_mesh_with_shape_keys()

    # build the key map from the negotiated name list.
    key_map = client._build_key_map(obj, list(ARKIT_NAMES))
    result["mapped_count"] = len(key_map)
    if len(key_map) != 52:
        result["errors"].append(f"expected 52 mapped keys, got {len(key_map)}")

    # decode a known frame and apply it to the shape keys.
    raw = [i / 52.0 for i in range(52)]
    rot = [1.0, 2.0, 3.0]
    pos = [4.0, 5.0, 6.0]
    data = _STRUCT.pack(MODE_LIVE, 123.0, *raw, *rot, *pos)
    parsed = client._parse_frame(data)
    if parsed is None:
        result["errors"].append("_parse_frame returned None")
    else:
        mode, audio_pos, blendshapes, prot, ppos = parsed
        if mode != MODE_LIVE:
            result["errors"].append(f"mode mismatch: {mode:#x}")
        if any(abs(a - b) > 1e-5 for a, b in zip(blendshapes, raw)):
            result["errors"].append("blendshape values did not round-trip")

        # Apply the parsed weights to the mapped shape keys.
        key_blocks = obj.data.shape_keys.key_blocks
        for idx, key_name in key_map.items():
            key_blocks[key_name].value = blendshapes[idx]
        # Verify a few keys landed the expected weight.
        for idx in (0, 13, 51):
            kb = key_blocks[key_map[idx]]
            if abs(kb.value - raw[idx]) > 1e-5:
                result["errors"].append(f"key {idx} weight {kb.value} != {raw[idx]}")

    result["ok"] = not result["errors"]
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f)
    print("[headless_test]", json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
