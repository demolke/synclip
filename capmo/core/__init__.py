"""
capmo.core -- the shared library of capmo (no Qt, no sockets required), used
by the CLI tools (``capmo/tools``) and the Qt UI alike:

* :mod:`capmo.core.pipeline`       -- the offline processing chain (session ->
  keypoints -> rig -> joints3d) and its per-step entry points
* :mod:`capmo.core.board`          -- ChArUco board model, detection (layout
  autodetect) and view-quality metrics + thresholds
* :mod:`capmo.core.board_coverage` -- :class:`LiveBoardTracker`, the live
  board-coverage / sharpness indicator behind the UI's Board check
* :mod:`capmo.core.pose`           -- the MediaPipe pose wrapper + model files
* :mod:`capmo.core.joints3d`       -- the ``<take>_joints3d.npz`` contract
* :mod:`capmo.core.session_io`     -- recording-session schema + :func:`load_session`
* :mod:`capmo.core.timeline`       -- per-frame host capture times, clock-offset models
* :mod:`capmo.core.h264`           -- Annex-B NAL scanning, SPS parsing (size / fps)
* :mod:`capmo.core.sync`           -- clock / frame / body / board time alignment
  (see its docstring for which is used where)
* :mod:`capmo.core.presets`        -- capture quality presets (resolution @ fps)
* :mod:`capmo.core.protocol`, :mod:`capmo.core.timesync`, :mod:`capmo.core.session`
  -- re-exports of the wire protocol, clock sync and session writer

Kept import-light on purpose (nothing is imported eagerly) so that the old
modules can depend on ``capmo.core.*`` without import cycles. Old import paths
that moved here (``capmo.pose``, ``capmo.timesync``) remain as aliases.
"""
