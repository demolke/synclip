"""Moved to :mod:`capmo.core.sync.clock` (NTP-style phone clock offsets); this
old import path stays as an alias."""
from .core.sync.clock import ClockEstimator, _Sample, compute_offset_rtt  # noqa: F401
