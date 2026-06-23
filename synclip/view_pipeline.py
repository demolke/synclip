"""
Per-view processing pipeline for the 3D preview quads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import modifiers as mod_mod
from .modifiers import ModifierConfig, ModifierContext
from . import ai_blendshapes
from .data import Stream

_IDENTITY_CURVE: list[list[float]] = [[0.0, 0.0], [1.0, 1.0]]
_POSE_AXIS_KEYS = ("rot_x", "rot_y", "rot_z", "pos_x", "pos_y", "pos_z")


def _default_modifiers() -> list[ModifierConfig]:
    """A fresh view reads from MediaPipe and applies head pose."""
    return [
        mod_mod.InputModifier.default_config(),
        mod_mod.PoseFilterModifier.default_config(),
    ]


@dataclass
class ViewConfig:
    """Per-view settings (what the right-dock panel edits)."""

    label: str
    mesh_path: str | None = None
    is_output: bool = False
    source: str = "mediapipe"
    modifiers: list[ModifierConfig] = field(default_factory=_default_modifiers)
    camera: dict = field(default_factory=lambda: {"yaw": 0.0, "pitch": 0.0, "zoom": 0.0})

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "mesh_path": self.mesh_path,
            "is_output": self.is_output,
            "source": self.source,
            "modifiers": [m.to_dict() for m in self.modifiers],
            "camera": dict(self.camera),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ViewConfig":
        cfg = cls(label=d.get("label", "View"))
        cfg.mesh_path = d.get("mesh_path")
        cfg.is_output = bool(d.get("is_output", False))
        cfg.source = d.get("source", "mediapipe")
        if "modifiers" in d:
            cfg.modifiers = [ModifierConfig.from_dict(m) for m in d["modifiers"]]
        cam = d.get("camera", {})
        cfg.camera = {
            "yaw": float(cam.get("yaw", 0.0)),
            "pitch": float(cam.get("pitch", 0.0)),
            "zoom": float(cam.get("zoom", 0.0)),
        }
        return cfg


class ViewPipeline:
    """Runtime pipeline for one view: source stream -> modifier stack."""

    def __init__(self, config: ViewConfig) -> None:
        self.config = config
        self._modifiers: list[mod_mod.Modifier] = []
        self._stream_frames: dict[str, tuple[list[dict], list[float]]] = {}
        self.apply_config(config)

    def apply_config(self, config: ViewConfig) -> None:
        """Rebuild the runtime modifier stack from config."""
        self.config = config
        self._modifiers = [mod_mod.create(mc) for mc in config.modifiers]
        # Re-run whole-timeline preparation against the cached streams.
        if self._stream_frames:
            self.set_streams(self._stream_frames)

    def set_streams(self, stream_frames: dict[str, tuple[list[dict], list[float]]]) -> None:
        """Hand the full take/AI/retarget streams to each modifier (detection)."""
        self._stream_frames = stream_frames
        for m in self._modifiers:
            m.prepare(stream_frames)

    def reset(self) -> None:
        for m in self._modifiers:
            m.reset()

    # -- introspection used by the UI --------------------------------------

    def modifier(self, index: int) -> mod_mod.Modifier:
        return self._modifiers[index]

    # -- processing --------------------------------------------------------

    def filter_pose(self, pose: dict | None) -> dict | None:
        """Run only the pose modifiers (for broadcast / idle pose refresh)."""
        if pose is None:
            return None
        ctx = ModifierContext(streams={}, pos_ms=0.0)
        out = pose
        for mc, m in zip(self.config.modifiers, self._modifiers):
            if mc.enabled and m.signal == "pose":
                _, out = m.apply([], out, ctx)
        return out

    def process(self, streams: dict[str, list[float] | None],
                pose: dict | None = None, pos_ms: float = 0.0
                ) -> tuple[list[float], dict | None]:
        """Run the full stack: pick the source stream, apply every enabled
        modifier in order.  Returns (final blendshapes, final pose)."""
        values = [0.0] * 52
        out_pose = pose
        ctx = ModifierContext(streams=streams, pos_ms=pos_ms)
        for mc, m in zip(self.config.modifiers, self._modifiers):
            if not mc.enabled:
                continue
            values, out_pose = m.apply(values, out_pose, ctx)
        return values, out_pose

    def process_all(
        self,
        take_stream: Stream,
        named_streams: dict[str, Stream],
    ) -> list[dict]:
        """Process every frame in *take_stream* and return output frame dicts.

        *named_streams* maps stream names (e.g. "ai", "retarget") to their
        :class:`~data.Stream` objects.  The pipeline is reset once
        before the loop so temporal modifiers start from a clean state.

        Returns a list of ``{"audio_position_ms": ..., "blendshapes": [...]}``
        dicts suitable for GLB export or further processing.
        """
        stream_frames = {
            "mediapipe": (take_stream.frames, take_stream.positions),
            **{name: (s.frames, s.positions) for name, s in named_streams.items()},
        }
        self.set_streams(stream_frames)
        self.reset()

        out: list[dict] = []
        for frame in take_stream.frames:
            pos = float(frame.get("audio_position_ms", 0.0))
            point_streams: dict[str, list[float] | None] = {
                "mediapipe": list(frame["blendshapes"]),
                **{name: s.values_at(pos) for name, s in named_streams.items()},
            }
            weights, _ = self.process(point_streams, frame.get("head_pose"), pos)
            out.append({"audio_position_ms": pos, "blendshapes": weights})
        return out
