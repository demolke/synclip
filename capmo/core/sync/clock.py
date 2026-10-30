"""
Clock synchronisation between the capmo server and each phone.

Each phone runs its own wall clock; to line up frames from different phones we
need every phone's clock expressed on the server's timeline. We use the standard
NTP four-timestamp exchange driven by the server's ``PING``/``SYNC`` round trip:

    t0  server sends PING            (server clock)
    t1  phone receives PING          (phone clock)
    t2  phone sends SYNC reply       (phone clock)
    t3  server receives SYNC reply   (server clock)

With those four points:

    offset = ((t1 - t0) + (t2 - t3)) / 2     # phone_clock - server_clock
    rtt    = (t3 - t0) - (t2 - t1)           # round-trip time, minus phone think time

``offset`` is how far the phone clock leads the server clock, so a phone
timestamp maps to server time as ``server = phone - offset``.

Pure module (no sockets), unit-testable, and generic enough to lift into a
shared layer later.
"""

from __future__ import annotations

from dataclasses import dataclass


def compute_offset_rtt(t0: float, t1: float, t2: float, t3: float) -> tuple[float, float]:
    """Return ``(offset_ms, rtt_ms)`` for one PING/SYNC exchange."""
    offset = ((t1 - t0) + (t2 - t3)) / 2.0
    rtt = (t3 - t0) - (t2 - t1)
    return offset, rtt


@dataclass
class _Sample:
    offset: float
    rtt: float
    t_ms: float | None = None    # server time the exchange completed (t3)


class ClockEstimator:
    """Tracks one phone's clock offset from a stream of PING/SYNC samples.

    The sample with the **smallest round-trip time** carries the least queuing
    jitter, so its offset is the most trustworthy -- we keep the best (lowest
    rtt) seen within a sliding set of recent samples. This is the same heuristic
    NTP uses ("clock filter").
    """

    def __init__(self, window: int = 8) -> None:
        self._window = max(1, window)
        self._recent: list[_Sample] = []
        self._best: _Sample | None = None

    def add_exchange(self, t0: float, t1: float, t2: float, t3: float) -> _Sample:
        offset, rtt = compute_offset_rtt(t0, t1, t2, t3)
        s = _Sample(offset, rtt, t3)
        self._recent.append(s)
        if len(self._recent) > self._window:
            self._recent.pop(0)
        # Best offset = the lowest-rtt sample currently in the window.
        self._best = min(self._recent, key=lambda x: x.rtt)
        return s

    @property
    def ready(self) -> bool:
        return self._best is not None

    @property
    def offset_ms(self) -> float:
        """phone_clock - server_clock (0.0 until the first exchange)."""
        return self._best.offset if self._best else 0.0

    @property
    def last_sample(self) -> _Sample | None:
        """The most recent raw exchange (unfiltered), or None."""
        return self._recent[-1] if self._recent else None

    @property
    def rtt_ms(self) -> float:
        return self._best.rtt if self._best else 0.0

    def to_server_ms(self, phone_ms: float) -> float:
        """Convert a phone wall-clock timestamp to the server timeline."""
        return phone_ms - self.offset_ms

    def to_phone_ms(self, server_ms: float) -> float:
        return server_ms + self.offset_ms
