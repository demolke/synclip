"""Deprecated: moved to :mod:`capmo.core.sync.frames`."""
import warnings

warnings.warn("capmo.sync is deprecated; use capmo.core.sync.frames", DeprecationWarning,
              stacklevel=2)
from .core.sync.frames import FrameSet, _nearest, nearest, synchronized_frames  # noqa: E402,F401
