"""
SynClip Receiver - Blender addon.

Receives ARKit blendshape data from the synclip capture tool over TCP and
records it as shape-key animation, with a manual keyframe-cleanup step.

Install: Edit -> Preferences -> Add-ons -> Install..., choose this folder zipped
(or the bundled addon zip), then enable the "Animation: SynClip Receiver"
checkbox. Disable the checkbox to unregister cleanly.

The implementation lives in ``client.py``; this module only carries ``bl_info``
and delegates registration so Blender's addon system can enable/disable it.
"""

bl_info = {
    "name": "SynClip Receiver",
    "author": "demolke",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N) > SynClip",
    "description": (
        "Receive ARKit blendshape data from the SynClip capture tool and "
        "record it as shape-key animation, with manual keyframe cleanup."
    ),
    "category": "Animation",
}

# Reload-safe import: when Blender re-runs this module (e.g. after editing the
# source and toggling the addon), reload the implementation rather than using a
# stale cached copy.
if "client" in locals():
    import importlib
    client = importlib.reload(client)  # noqa: F821  (defined by the import below on first load)
else:
    from . import client


def register() -> None:
    client.register()


def unregister() -> None:
    client.unregister()


# Allow `blender --python tools/blender/__init__.py` and direct testing.
if __name__ == "__main__":
    register()
