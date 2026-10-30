# Character rig assets

Humanoid character meshes used as retarget targets for the capmo 3D pose track.
The capture pipeline produces a per-bone rotation track (`capmo/retarget.py`);
`capmo/character_rig.py` binds that track onto the skeleton of one of these
meshes so a real character can be driven by the reconstructed motion.

## Origin & license

Quaternius — **"Universal Base Characters"**, released **CC0 1.0** (public domain
dedication, no attribution required). See `cc0_license.txt` in this directory.

- https://quaternius.com/packs/universalbasecharacters.html
- https://quaternius.itch.io/universal-base-characters

The pack advertises six characters (Superhero / Regular / Teen × male / female);
only the two **Superhero** full bodies ship in the free tier, so those are what
we vendor here. The Regular/Teen variants live in the paid "Source" pack and are
**not** included. All six share the same skeleton, so these two fully exercise
the retarget path.

## Contents

```
superhero_male/    superhero_male.gltf  + .bin + 7 PBR textures   (8483 verts)
superhero_female/  superhero_female.gltf + .bin + 7 PBR textures  (8844 verts)
```

Stored as the vendor shipped them — glTF 2.0 (`.gltf` + external `.bin` +
PNG textures), not single-file `.glb`. The textures are kept for a future
textured OpenGL view; the current flat-shaded renderer ignores them.

## Skeleton

Both meshes are skinned to one shared **65-joint, UE-mannequin-style** armature
(Y-up, metres; the `root` node carries the −90°X Blender Z-up→Y-up bake):

```
root → pelvis → spine_01 → spine_02 → spine_03 → neck_01 → Head
                                     ├ clavicle_l → upperarm_l → lowerarm_l → hand_l (+fingers)
                                     └ clavicle_r → upperarm_r → lowerarm_r → hand_r (+fingers)
       pelvis → thigh_{l,r} → calf_{l,r} → foot_{l,r} → ball_{l,r}
```

`capmo/character_rig.py` (`QUATERNIUS_JOINTS`) maps our 14-bone capture armature
onto these bones (wrist→`hand_*`, forearm→`lowerarm_*`, shin→`calf_*`, toe→
`ball_*`, chest→`spine_03`). Fingers and the 3-segment spine's middle joints have
no capture counterpart and stay at their bind pose.
