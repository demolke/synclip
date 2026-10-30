"""pytest fixtures for the mocap_retarget tests (run with Blender's Python module, bpy):

    /opt/bpyenv/bin/python -m pytest blender/mocap_retarget/tests

The rig is generated once per session (Rigify, ~1 s) and saved to a temporary .blend;
every test re-opens that file, so tests do not see each other's changes. The mocap is
``take1_slice120.npz``: 120 frames (a walk) of the real take1 reconstruction.
"""
import os
import sys

import pytest

bpy = pytest.importorskip("bpy")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import build_rig  # noqa: E402
import retarget as mr  # noqa: E402

NPZ = os.path.join(HERE, "take1_slice120.npz")


@pytest.fixture(scope="session")
def rig_blend(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("rig") / "rig.blend")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build_rig.build_character("rig")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    mr.register()
    return path


@pytest.fixture
def rig(rig_blend):
    """The rig object of a freshly opened copy of the session's rig file."""
    for hs in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.frame_change_post):
        for h in list(hs):
            if getattr(h, "__name__", "") == "_live_handler":
                hs.remove(h)
    bpy.ops.wm.open_mainfile(filepath=rig_blend)
    mr._LIVE.clear()
    mr.register()
    return bpy.data.objects["rig"]


def quiet_params(**kw):
    """Params whose bake has no temporal post-processing (live == bake per frame)."""
    base = dict(foot_lock=False, rot_smooth=0.0, rot_despike=0.0, max_ang_vel=0.0,
                ground_smooth=0.0, gap_fill_rotations=False)
    base.update(kw)
    return mr.Params(**base)
