"""
capmo.core.sync -- every way capmo lines cameras up in time, by purpose.

They stay distinct because they answer different questions with different
inputs:

* :mod:`.clock` -- **live, network**: NTP-style phone clock offset / rtt from
  the PING/SYNC exchange (``ClockEstimator``; used by the capture server, the
  numbers go into every session's ``<slug>.clock.jsonl``). Turns phone capture
  times into host times; the per-frame model on top of it is
  :mod:`capmo.core.timeline`.
* :mod:`.frames` -- **timestamps already on one clock**: nearest-frame matching
  within a tolerance (live board tracker: which cameras saw the board at the
  same moment; ``synchronized_frames`` groups whole streams).
* :mod:`.timeline` -- frame-index offsets of a session take from its common
  clock (feeds ``calibrate_board --offsets`` for sessions).
* :mod:`.body` -- **offline, from the performer**: epipolar / reprojection
  offset scan of the 2D keypoints (``selfcal.estimate_offsets``,
  ``refine_offsets``); refines (and, for ordinary videos, replaces) the clock.
* :mod:`.board_motion` -- **offline, from the board**: cross-correlation of the
  board-centre speed between cameras (``calibrate_board --estimate-offsets``).

Import the submodules explicitly; this package imports nothing eagerly.
"""
