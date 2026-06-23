"""
Detect rapid mouth closures in a blendshape animation and re-assert them post-mix.

Quick lip closures during speech -- the brief mouth-shut of bilabial / plosive
consonants (p, b, m) that lasts only one to three frames -- are routinely lost
by the time the animation reaches the output: MediaPipe under-drives jawOpen,
the audio-driven AI model smooths transients, and the temporal-EMA / AI-mix /
playback-interpolation passes all blunt a single-frame valley. These closures
are genuine and desirable, so we want to keep them.

  1. detect_closures() scans a source's jawOpen channel for a *valley* -- open
     then briefly (1-3 frames) closed then open again -- and returns timed
     ClosureEvents (in audio-position ms, so events from sources at different
     frame rates can be unioned on one timeline).

  2. enforce_closure() is called once per rendered frame, AFTER the mix, and
     drives the mouth shut on any detected event window using a raised-cosine
     envelope so the closure survives whatever the mix did to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ARKit-52 channel indices (see arkit_names.BLENDSHAPE_NAMES).
JAW_OPEN = 25
MOUTH_CLOSE = 27

# A rapid closure spans at most this many frames (the "one, two or three").
MAX_WIDTH = 3


@dataclass
class ClosureEvent:
    """One detected rapid mouth closure, on the shared audio-position timeline.

    start_ms / end_ms are the open "shoulders" either side of the valley (where
    the enforcement envelope is 0); center_ms is the deepest point of the dip
    (where the envelope peaks at 1).
    """

    start_ms: float
    center_ms: float
    end_ms: float
    strength: float  # depth of the detected dip (0..1), for merge/diagnostics


def detect_closures(
    frames: list[dict],
    positions: list[float] | None = None,
    *,
    drop: float = 0.10,
    open_min: float = 0.15,
    max_width: int = MAX_WIDTH,
    channel: int = JAW_OPEN,
) -> list[ClosureEvent]:
    """Return the rapid mouth closures found in *frames*.

    A frame window i..i+L-1 (L in 1..max_width) is a closure when both
    surrounding "shoulder" frames are open (jawOpen >= *open_min*) and each
    sits at least *drop* above the window's minimum -- i.e. open -> closed ->
    open. Requiring *both* shoulders open is what separates a quick
    close-and-reopen (kept) from a mouth that simply closes and stays shut at
    the end of a word (ignored).
    """
    n = len(frames)
    if n < 3:
        return []
    if positions is None:
        positions = [f["audio_position_ms"] for f in frames]
    # When using the default channel, combine jawOpen with mouthClose so
    # labial plosives (b/p/m) are caught even when the jaw barely moves.
    if channel == JAW_OPEN:
        jaw = []
        for f in frames:
            bs = f.get("blendshapes", [])
            j = float(bs[JAW_OPEN]) if JAW_OPEN < len(bs) else 0.0
            mc = float(bs[MOUTH_CLOSE]) if MOUTH_CLOSE < len(bs) else 0.0
            jaw.append(j * (1.0 - mc))
    else:
        jaw = [
            float(f["blendshapes"][channel]) if channel < len(f["blendshapes"]) else 0.0
            for f in frames
        ]

    events: list[ClosureEvent] = []
    i = 1
    while i < n - 1:
        best: tuple[float, int] | None = None  # (strength, width)
        for L in range(1, max_width + 1):
            if i + L > n - 1:
                break
            left, right = jaw[i - 1], jaw[i + L]
            if left < open_min or right < open_min:
                continue
            window = jaw[i:i + L]
            # Every frame in the window must sit at least *drop* below both
            # shoulders, so a window holds only the closure itself -- never the
            # open frames of the ramp into or out of it (which would otherwise
            # let a wide window mis-centre a sharp single-frame closure).
            if (left - max(window)) >= drop and (right - max(window)) >= drop:
                s = min(left, right) - min(window)
                # Prefer the shortest qualifying window (tightest fit).
                if best is None or L < best[1]:
                    best = (s, L)
        if best is not None:
            s, L = best
            center_idx = i + (L - 1) / 2.0
            lo = int(center_idx)
            hi = min(lo + 1, n - 1)
            frac = center_idx - lo
            center_ms = positions[lo] + frac * (positions[hi] - positions[lo])
            events.append(ClosureEvent(positions[i - 1], center_ms, positions[i + L], s))
            i += L + 1  # skip past this valley + its right shoulder
        else:
            i += 1
    return events


def merge_events(events: list[ClosureEvent]) -> list[ClosureEvent]:
    """Merge overlapping events (in ms) into single windows, keeping the deeper
    valley's center/strength. Returns a list sorted by start_ms."""
    if not events:
        return []
    ev = sorted(events, key=lambda e: e.start_ms)
    out = [ev[0]]
    for e in ev[1:]:
        last = out[-1]
        if e.start_ms <= last.end_ms:
            if e.strength >= last.strength:
                center, strength = e.center_ms, e.strength
            else:
                center, strength = last.center_ms, last.strength
            out[-1] = ClosureEvent(
                min(last.start_ms, e.start_ms),
                center,
                max(last.end_ms, e.end_ms),
                strength,
            )
        else:
            out.append(e)
    return out


def _envelope(pos_ms: float, e: ClosureEvent) -> float:
    """Raised-cosine weight in [0,1]: 0 at the shoulders, 1 at the center.

    Handles asymmetric windows (center not exactly mid-way) by scaling each
    side independently."""
    if pos_ms <= e.start_ms or pos_ms >= e.end_ms:
        return 0.0
    if pos_ms <= e.center_ms:
        denom = e.center_ms - e.start_ms
        t = 1.0 if denom <= 0 else (pos_ms - e.start_ms) / denom
    else:
        denom = e.end_ms - e.center_ms
        t = 1.0 if denom <= 0 else (e.end_ms - pos_ms) / denom
    return 0.5 - 0.5 * math.cos(math.pi * t)


def enforce_closure(
    values: list[float],
    pos_ms: float,
    events: list[ClosureEvent],
    amount: float,
    *,
    jaw_idx: int = JAW_OPEN,
    close_idx: int = MOUTH_CLOSE,
) -> list[float]:
    """Drive the mouth shut on *values* if *pos_ms* falls inside a closure.

    *amount* (0..1) is the peak close fraction at a valley center: jawOpen is
    scaled toward 0 and mouthClose pushed up, weighted by the raised-cosine
    envelope so the override blends smoothly back to the mix at the shoulders.
    Returns *values* unchanged when no event covers *pos_ms*.
    """
    if not events or amount <= 0.0:
        return values
    peak = 0.0
    for e in events:
        if e.start_ms >= pos_ms:
            break  # events are sorted by start_ms
        if pos_ms < e.end_ms:
            env = _envelope(pos_ms, e)
            if env > peak:
                peak = env
    if peak <= 0.0:
        return values
    out = list(values)
    close = peak * amount
    if jaw_idx < len(out):
        out[jaw_idx] = out[jaw_idx] * (1.0 - close)
    if close_idx < len(out):
        out[close_idx] = max(out[close_idx], close)
    return out
