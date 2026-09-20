"""
GNM - Reassemble Quads in Blender (custom metadata only, no file)

viewer.py now stores quads directly in glTF node extras:
  node.extras.gnm_quads_shared = [[a,b,c,d], ...]
  node.extras.gnm_quad_uvs = [[[u0,v0],...x4], ...]
  node.extras.gnm_quad_tri_pairs

Blender's glTF importer imports node extras as object custom properties:
  obj["gnm_quads_shared"] etc.

This script reads ONLY from object custom props, no .json file needed.

Export:
  viewer.py -> .gltf + .bin (quads inside glTF)

Blender:
  Import glTF -> select GNM_Head -> run this script
  It dissolves diagonal edge a-c: tri [a,b,c] + [a,c,d] -> quad [a,b,c,d]
  Preserves UVs (viewer.py dedupes by (vert,uv)) and shape keys.
"""

import bpy
import bmesh

OBJECT_NAME = "GNM_Head"  # None = active object

def reassemble_quads():
    obj = bpy.data.objects.get(OBJECT_NAME) if OBJECT_NAME else None
    if not obj:
        obj = bpy.context.active_object
    if not obj or obj.type != 'MESH':
        raise RuntimeError("Select mesh object (GNM_Head)")

    # Read from object custom props (from glTF node extras)
    if "gnm_quads_shared" not in obj:
        # Fallback: mesh custom props (from glTF mesh extras)
        if "gnm_quads_shared" in obj.data:
            quads_shared = list(obj.data["gnm_quads_shared"])
            quad_uvs = list(obj.data.get("gnm_quad_uvs", []))
        else:
            raise RuntimeError(
                f"Object {obj.name} has no custom prop 'gnm_quads_shared'. "
                "Make sure you exported with updated viewer.py and imported glTF with 'Import Custom Properties' enabled."
            )
    else:
        quads_shared = list(obj["gnm_quads_shared"])
        quad_uvs = list(obj.get("gnm_quad_uvs", []))

    print(f"[Quads] Found {len(quads_shared)} quads on object {obj.name} (custom props, no file)")

    mesh = obj.data
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Build edge map (min,max) -> edge
    edge_map = {}
    for e in bm.edges:
        k = (min(e.verts[0].index, e.verts[1].index), max(e.verts[0].index, e.verts[1].index))
        edge_map[k] = e

    edges_to_dissolve = []
    for quad in quads_shared:
        if len(quad) != 4:
            continue
        a,b,c,d = quad
        k = (min(a,c), max(a,c))  # diagonal a-c is how viewer.py splits
        edge = edge_map.get(k)
        if edge and len(edge.link_faces) == 2:
            edges_to_dissolve.append(edge)

    print(f"[Quads] Dissolving {len(edges_to_dissolve)} diagonal edges (a-c)")

    if edges_to_dissolve:
        # This preserves shape keys (morph targets) and UVs
        bmesh.ops.dissolve_edges(bm, edges=edges_to_dissolve, use_verts=False, use_face_split=False)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
    else:
        print("[Quads] No edges found to dissolve - maybe already quads?")
        bm.free()

    quad_count = sum(1 for p in mesh.polygons if len(p.vertices) == 4)
    tri_count = sum(1 for p in mesh.polygons if len(p.vertices) == 3)
    print(f"[Quads] Result: {quad_count} quads, {tri_count} tris")

    # Cleanup custom properties if successful
    expected_quads = len(quads_shared)
    # Consider success if we got at least 90% of expected quads and no major tris left from quads
    success = quad_count >= int(expected_quads * 0.9) and tri_count < expected_quads * 0.2
    if success:
        print(f"[Quads] Success - removing temporary custom properties from object")
        props_to_remove = [
            "gnm_quads_shared",
            "gnm_quad_uvs",
            "gnm_quad_tri_pairs",
            "gnm_num_quads",
            "gnm_quads_original",
            "gnm_new_old_indices",
            "gnm_new_uvs",
            "gnm_triangle_quad_ids",
        ]
        for prop in props_to_remove:
            if prop in obj:
                try:
                    del obj[prop]
                    print(f"  Removed obj['{prop}']")
                except Exception as e:
                    print(f"  Failed to remove obj['{prop}']: {e}")
            if prop in obj.data:
                try:
                    del obj.data[prop]
                    print(f"  Removed mesh['{prop}']")
                except Exception as e:
                    print(f"  Failed to remove mesh['{prop}']: {e}")
        print(f"[Quads] Cleanup done - object now clean")
    else:
        print(f"[Quads] Not cleaning up - expected {expected_quads} quads but got {quad_count}. Keeping custom props for debugging.")

if __name__ == "__main__":
    reassemble_quads()
