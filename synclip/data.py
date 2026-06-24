"""
JSON save/load/append for .synclip.json files.

Schema version: 1.0
"""

from __future__ import annotations

import bisect
import json
import os
import pathlib
import datetime
from dataclasses import dataclass, field

from .arkit_names import BLENDSHAPE_NAMES

_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Stream: a keyed sequence of blendshape frames with pre-computed positions
# ---------------------------------------------------------------------------

@dataclass
class Stream:
    """A named sequence of blendshape frames with cached audio positions.

    Positions are extracted once from the frames list; all interpolation
    helpers operate on the cached list to avoid repeated comprehensions.
    """

    frames: list[dict] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)

    @classmethod
    def from_frames(cls, frames: list[dict]) -> "Stream":
        """Build a Stream, extracting audio positions from the frame dicts."""
        positions = [float(f["audio_position_ms"]) for f in frames]
        return cls(frames=list(frames), positions=positions)

    def values_at(self, pos_ms: float) -> list[float] | None:
        """Interpolated blendshapes at *pos_ms*, or None if empty."""
        return interp_blendshapes(self.frames, pos_ms, self.positions)

    def head_pose_at(self, pos_ms: float) -> dict | None:
        """Interpolated head pose at *pos_ms*, or None if unavailable."""
        return interp_head_pose(self.frames, pos_ms, self.positions)


# ---------------------------------------------------------------------------
# Interpolation helpers (pure functions; no Qt dependency)
# ---------------------------------------------------------------------------

def interp_blendshapes(
    frames: list[dict], audio_pos_ms: float, positions: list[float] | None = None
) -> list[float] | None:
    """Interpolate blendshapes from *frames* at *audio_pos_ms*."""
    if not frames:
        return None
    if positions is None:
        positions = [f["audio_position_ms"] for f in frames]
    if audio_pos_ms <= positions[0]:
        return list(frames[0]["blendshapes"])
    if audio_pos_ms >= positions[-1]:
        return list(frames[-1]["blendshapes"])
    idx = bisect.bisect_left(positions, audio_pos_ms)
    idx = max(1, min(idx, len(frames) - 1))
    f0, f1 = frames[idx - 1], frames[idx]
    t0, t1 = f0["audio_position_ms"], f1["audio_position_ms"]
    if t1 == t0:
        return list(f0["blendshapes"])
    alpha = (audio_pos_ms - t0) / (t1 - t0)
    b0, b1 = f0["blendshapes"], f1["blendshapes"]
    return [b0[i] + alpha * (b1[i] - b0[i]) for i in range(len(b0))]


def interp_head_pose(
    frames: list[dict], audio_pos_ms: float, positions: list[float] | None = None
) -> dict | None:
    """Interpolate head pose from *frames* at *audio_pos_ms*."""
    if not frames or "head_pose" not in frames[0]:
        return None
    if positions is None:
        positions = [f["audio_position_ms"] for f in frames]
    if audio_pos_ms <= positions[0]:
        return frames[0].get("head_pose")
    if audio_pos_ms >= positions[-1]:
        return frames[-1].get("head_pose")
    idx = bisect.bisect_left(positions, audio_pos_ms)
    idx = max(1, min(idx, len(frames) - 1))
    f0, f1 = frames[idx - 1], frames[idx]
    p0, p1 = f0.get("head_pose"), f1.get("head_pose")
    if not p0 or not p1:
        return p0 or p1
    t0, t1 = f0["audio_position_ms"], f1["audio_position_ms"]
    a = 0.0 if t1 == t0 else (audio_pos_ms - t0) / (t1 - t0)

    def lerp3(k):
        v0, v1 = p0.get(k, [0.0] * 3), p1.get(k, [0.0] * 3)
        return [v0[i] + a * (v1[i] - v0[i]) for i in range(3)]

    return {"rot": lerp3("rot"), "pos": lerp3("pos")}


# ---------------------------------------------------------------------------
# StreamStore: single owner of all named blendshape streams
# ---------------------------------------------------------------------------

class StreamStore:
    """Holds a dict[str, Stream] and provides a clean API for the window.

    Keys are canonical stream names: "mediapipe", "ai", "retarget", ...
    Positions are computed once when a stream is set, not on every sample.
    """

    def __init__(self) -> None:
        self._streams: dict[str, Stream] = {}

    def set(self, name: str, frames: list[dict]) -> None:
        """Store *frames* under *name*, caching audio positions."""
        if frames:
            positions = [float(f["audio_position_ms"]) for f in frames]
            self._streams[name] = Stream(frames=list(frames), positions=positions)
        else:
            self._streams.pop(name, None)

    def clear(self, name: str) -> None:
        """Remove *name* from the store (no-op if absent)."""
        self._streams.pop(name, None)

    def clear_all(self) -> None:
        self._streams.clear()

    def get(self, name: str) -> Stream | None:
        return self._streams.get(name)

    def frames(self, name: str) -> list[dict]:
        s = self._streams.get(name)
        return s.frames if s is not None else []

    def positions(self, name: str) -> list[float]:
        s = self._streams.get(name)
        return s.positions if s is not None else []

    def has(self, name: str) -> bool:
        return name in self._streams

    def sample(self, name: str, pos_ms: float) -> list[float] | None:
        """Return interpolated blendshapes for *name* at *pos_ms*, or None."""
        s = self._streams.get(name)
        if s is None:
            return None
        return s.values_at(pos_ms)

    def sample_all(self, pos_ms: float) -> dict[str, list[float] | None]:
        """Sample every stream at *pos_ms*."""
        return {name: s.values_at(pos_ms) for name, s in self._streams.items()}

    def prepare_all(self) -> dict[str, tuple[list[dict], list[float]]]:
        """Return the (frames, positions) pairs used by ViewPipeline.set_streams."""
        return {name: (s.frames, s.positions) for name, s in self._streams.items()}

    def names(self) -> list[str]:
        return list(self._streams.keys())


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def synclip_path(audio_path: str) -> str:
    """Return the sibling .synclip.json path for *audio_path*."""
    p = pathlib.Path(audio_path)
    return str(p.parent / (p.stem + ".synclip.json"))


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_synclip(audio_path: str) -> dict | None:
    """Load an existing .synclip.json file, or return None if absent.

    If the file exists but is corrupt (truncated / invalid JSON), it is moved
    aside to a ``.corrupt`` backup and None is returned, so a single bad write
    doesn't make the tool unusable for that audio file.
    """
    path = synclip_path(audio_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        backup = path + ".corrupt"
        try:
            os.replace(path, backup)
            print(f"[data] corrupt {path!r} ({exc}); moved to {backup!r}")
        except OSError:
            pass
        return None
    if not isinstance(data, dict):
        return None
    # Defensive: ensure a 'takes' list exists and holds only dicts.
    takes = data.get("takes")
    if not isinstance(takes, list):
        data["takes"] = []
    else:
        data["takes"] = [t for t in takes if isinstance(t, dict)]
    return data


def save_synclip(audio_path: str, data: dict) -> None:
    """Atomically write *data* to the .synclip.json file (write .tmp then rename)."""
    path = synclip_path(audio_path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _new_document(audio_path: str, audio_duration_ms: float, fps: int) -> dict:
    """Create a fresh synclip document for *audio_path*."""
    return {
        "version": _SCHEMA_VERSION,
        "audio_file": os.path.basename(audio_path),
        "audio_duration_ms": audio_duration_ms,
        "fps": fps,
        "blendshape_names": list(BLENDSHAPE_NAMES),
        "default_take": None,
        "takes": [],
    }


def _next_take_id(takes: list[dict]) -> str:
    """Generate take_NNN id that is one past the highest existing number."""
    max_n = 0
    for take in takes:
        tid = take.get("take_id", "")
        if tid.startswith("take_"):
            try:
                n = int(tid[5:])
                if n > max_n:
                    max_n = n
            except ValueError:
                pass
    return f"take_{max_n + 1:03d}"


# ---------------------------------------------------------------------------
# Public mutation API
# ---------------------------------------------------------------------------

def append_take(
    audio_path: str,
    frames: list[dict],
    audio_duration_ms: float,
    fps: int = 30,
    capture_settings: dict | None = None,
) -> dict:
    """Load (or create) the synclip document, append a new take, save, and return it.

    *frames* must be a list of dicts with keys:
        frame_index     : int
        audio_position_ms : float
        blendshapes     : list[float]  (52 values)

    Returns the newly appended take dict.
    """
    data = load_synclip(audio_path)
    if data is None:
        data = _new_document(audio_path, audio_duration_ms, fps)
    else:
        # Update duration in case it changed (e.g. resampled audio).
        data["audio_duration_ms"] = audio_duration_ms

    take_id = _next_take_id(data["takes"])
    is_first = len(data["takes"]) == 0

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    new_take: dict = {
        "take_id": take_id,
        "timestamp_utc": timestamp,
        "is_default": is_first,
        "notes": "",
        "blend_scales": {name: 1.0 for name in BLENDSHAPE_NAMES},
        "capture_settings": capture_settings or {},
        # All blendshape tracks live here, keyed by name. The recorded capture is
        # the "mediapipe" stream; ai/retarget/rhubarb/... are added to the same
        # take as they are generated. There is no doc-level stream storage.
        "streams": {"mediapipe": frames},
    }

    data["takes"].append(new_take)

    if is_first:
        data["default_take"] = take_id

    save_synclip(audio_path, data)
    return new_take


def set_default_take(audio_path: str, take_id: str) -> None:
    """Mark *take_id* as the default take, clearing the flag from all others."""
    data = load_synclip(audio_path)
    if data is None:
        raise FileNotFoundError(f"No synclip file for {audio_path!r}")

    found = False
    for take in data["takes"]:
        if take["take_id"] == take_id:
            take["is_default"] = True
            found = True
        else:
            take["is_default"] = False

    if not found:
        raise ValueError(f"take_id {take_id!r} not found in {synclip_path(audio_path)!r}")

    data["default_take"] = take_id
    save_synclip(audio_path, data)


def delete_take(audio_path: str, take_id: str) -> None:
    """Remove *take_id* from the document.

    If it was the default take, promote the next remaining take to default
    (or set default_take to None when no takes remain).
    """
    data = load_synclip(audio_path)
    if data is None:
        raise FileNotFoundError(f"No synclip file for {audio_path!r}")

    original_len = len(data["takes"])
    was_default = data["default_take"] == take_id

    data["takes"] = [t for t in data["takes"] if t["take_id"] != take_id]

    if len(data["takes"]) == original_len:
        raise ValueError(f"take_id {take_id!r} not found in {synclip_path(audio_path)!r}")

    if was_default:
        if data["takes"]:
            # Promote the first remaining take.
            data["takes"][0]["is_default"] = True
            data["default_take"] = data["takes"][0]["take_id"]
        else:
            data["default_take"] = None

    save_synclip(audio_path, data)


def rename_take(audio_path: str, take_id: str, name: str) -> None:
    """Set or clear the human-readable name for *take_id*."""
    data = load_synclip(audio_path)
    if data is None:
        raise FileNotFoundError(f"No synclip file for {audio_path!r}")
    for take in data["takes"]:
        if take["take_id"] == take_id:
            take["name"] = name.strip()
            save_synclip(audio_path, data)
            return
    raise ValueError(f"take_id {take_id!r} not found in {synclip_path(audio_path)!r}")
