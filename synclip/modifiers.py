"""
Generic, self-describing blendshape/pose modifiers.

A *modifier* takes the running signal (52 blendshape weights + an optional head
pose) and applies one tweak, producing a new signal.  Modifiers are chained in
an arbitrary, per-view order (see :class:`view_pipeline.ViewPipeline`).

The whole point of this module is that adding a new modifier is a single
class and nothing else has to change:

  * it declares its tunable :class:`ParamSpec` list, so serialisation defaults
    and the right-dock editor build themselves;
  * it is discovered through the :data:`registry`, so the pipeline, the "+ Add"
    menu and the storage layer never name a concrete type;
  * ``influence`` (the wet/dry / intensity knob every modifier carries) is read
    from its :class:`ModifierConfig`, so the UI shows one consistent control.

Streams are just string-keyed channels of frames (``mediapipe``, ``ai``,
``retarget`` are conventions, not enums).  A modifier reaches any stream by name
through :class:`ModifierContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .blendshape_filter import _GAIN_CHANNELS, _WIDTH_CHANNELS, _sample_lut
from .curve_lut import build_lut
from . import ai_blendshapes
from . import mouth_closure as closure_mod

_IDENTITY_CURVE: list[list[float]] = [[0.0, 0.0], [1.0, 1.0]]


# ---------------------------------------------------------------------------
# Param schema (drives serialisation defaults + the auto-built editor)
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """Describes one tunable parameter of a modifier.

    *kind* selects how the value serialises and how the UI renders it:
        "float"     -> slider (min..max)
        "bool"      -> checkbox
        "enum"      -> combo (choices = list of (value, label))
        "curve"     -> CurveEditor (value is a list of [x, y] points)
        "vec3bool"  -> three checkboxes (value is [bool, bool, bool])
        "streamset" -> a checkbox per available stream (value is list[str])
    """
    name: str
    kind: str
    default: Any
    label: str = ""
    min: float = 0.0
    max: float = 1.0
    choices: list[tuple[Any, str]] | None = None

    def coerce(self, value: Any) -> Any:
        """Return *value* coerced/defaulted to this spec's type."""
        if value is None:
            return self.clone_default()
        if self.kind == "float":
            return float(value)
        if self.kind == "bool":
            return bool(value)
        if self.kind == "enum":
            return value
        if self.kind == "curve":
            return [list(p) for p in value]
        if self.kind == "vec3bool":
            return [bool(value[i]) if i < len(value) else True for i in range(3)]
        if self.kind == "streamset":
            return list(value)
        return value

    def clone_default(self) -> Any:
        d = self.default
        if isinstance(d, list):
            return [list(p) if isinstance(p, list) else p for p in d]
        return d


# ---------------------------------------------------------------------------
# Config + runtime context
# ---------------------------------------------------------------------------

@dataclass
class ModifierConfig:
    """Serialisable per-modifier settings."""
    type: str
    enabled: bool = True
    influence: float = 1.0
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "enabled": self.enabled,
            "influence": self.influence,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModifierConfig":
        return cls(
            type=d["type"],
            enabled=bool(d.get("enabled", True)),
            influence=float(d.get("influence", 1.0)),
            params=dict(d.get("params", {})),
        )


@dataclass
class ModifierContext:
    """Per-sample context handed to every ``apply()`` call."""
    streams: dict[str, list[float] | None]   # current interpolated sample/stream
    pos_ms: float = 0.0

    @staticmethod
    def blend(base: list[float], wet: list[float], influence: float) -> list[float]:
        """Wet/dry mix: ``base + influence * (wet - base)``."""
        if influence >= 1.0:
            return list(wet)
        if influence <= 0.0:
            return list(base)
        return [b + influence * (w - b) for b, w in zip(base, wet)]


# ---------------------------------------------------------------------------
# Base class + registry
# ---------------------------------------------------------------------------

class Modifier:
    """Base class for all modifiers.

    Subclasses set the class attributes and implement :meth:`transform` (for
    blendshape modifiers) or :meth:`transform_pose` (for pose modifiers).
    """

    type_name: str = ""
    display_name: str = ""
    signal: str = "blendshapes"     # "blendshapes" | "pose"
    has_influence: bool = True
    param_specs: list[ParamSpec] = []

    def __init__(self, config: ModifierConfig) -> None:
        self.config = config
        self.reset()

    # -- param access -------------------------------------------------------

    def p(self, name: str) -> Any:
        """Param value, defaulted/coerced via the spec."""
        for spec in self.param_specs:
            if spec.name == name:
                return spec.coerce(self.config.params.get(name))
        return self.config.params.get(name)

    @property
    def influence(self) -> float:
        return self.config.influence

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Drop any temporal state (called on reorder/source change)."""

    def prepare(self, stream_frames: dict[str, tuple[list[dict], list[float]]]) -> None:
        """Optional whole-timeline pass (e.g. event detection).

        *stream_frames* maps stream name -> (frames, positions) for the full
        take/AI/retarget streams.  Default: no-op.
        """

    def status_text(self) -> str:
        """Short status string shown in the modifier UI row. Empty = no status."""
        return ""

    # -- per-sample apply ---------------------------------------------------

    def apply(self, values: list[float], pose: dict | None,
              ctx: ModifierContext) -> tuple[list[float], dict | None]:
        if self.signal == "pose":
            return values, self.transform_pose(pose, ctx)
        wet = self.transform(values, ctx)
        return ctx.blend(values, wet, self.influence), pose

    def transform(self, values: list[float], ctx: ModifierContext) -> list[float]:
        """Return the fully-wet blendshapes; the base blends by influence."""
        return values

    def transform_pose(self, pose: dict | None, ctx: ModifierContext) -> dict | None:
        return pose

    # -- UI defaults --------------------------------------------------------

    @classmethod
    def default_config(cls) -> ModifierConfig:
        params = {s.name: s.clone_default() for s in cls.param_specs}
        return ModifierConfig(type=cls.type_name, enabled=True,
                              influence=1.0, params=params)


_REGISTRY: dict[str, type[Modifier]] = {}


def register(cls: type[Modifier]) -> type[Modifier]:
    _REGISTRY[cls.type_name] = cls
    return cls


def create(config: ModifierConfig) -> Modifier:
    cls = _REGISTRY.get(config.type)
    if cls is None:
        return _PassThrough(config)
    return cls(config)


def get_class(type_name: str) -> type[Modifier] | None:
    return _REGISTRY.get(type_name)


def available_types() -> list[tuple[str, str]]:
    """(type_name, display_name) for every registered modifier, in reg order."""
    return [(c.type_name, c.display_name) for c in _REGISTRY.values()]


class _PassThrough(Modifier):
    """Fallback for an unknown modifier type (forward-compatible loads)."""
    type_name = "_unknown"
    display_name = "Unknown"


# ---------------------------------------------------------------------------
# Concrete modifiers
# ---------------------------------------------------------------------------

@register
class CurvesModifier(Modifier):
    """Response curves for the mouth-articulation and mouth-width channels."""
    type_name = "curves"
    display_name = "Mouth curves"
    param_specs = [
        ParamSpec("gain_curve", "curve", _IDENTITY_CURVE, label="Gain curve"),
        ParamSpec("width_curve", "curve", _IDENTITY_CURVE, label="Width curve"),
    ]

    def _lut(self, curve) -> list[float] | None:
        pts = [list(p) for p in curve]
        if pts == _IDENTITY_CURVE:
            return None
        return build_lut([(p[0], p[1]) for p in pts])

    def transform(self, values: list[float], ctx: ModifierContext) -> list[float]:
        vals = list(values)
        glut = self._lut(self.p("gain_curve"))
        if glut:
            for idx in _GAIN_CHANNELS:
                vals[idx] = _sample_lut(glut, vals[idx])
        wlut = self._lut(self.p("width_curve"))
        if wlut:
            for idx in _WIDTH_CHANNELS:
                vals[idx] = _sample_lut(wlut, vals[idx])
        return vals


@register
class SmoothModifier(Modifier):
    """Temporal EMA smoothing.  ``influence`` is the smoothing strength."""
    type_name = "smooth"
    display_name = "Smooth"
    param_specs: list[ParamSpec] = []

    def reset(self) -> None:
        self._prev: list[float] | None = None

    def transform(self, values: list[float], ctx: ModifierContext) -> list[float]:
        # Fully-wet = the previous OUTPUT, so the base blend
        # ``v + influence*(prev - v)`` is exactly an EMA with weight=influence.
        prev = self._prev if self._prev is not None else values
        wet = list(prev)
        # Remember the post-blend result for the next frame.
        blended = ctx.blend(values, wet, self.influence)
        self._prev = blended
        return wet


@register
class ClosureModifier(Modifier):
    """Re-assert rapid mouth closures (p/b/m) the mix would smooth away.

    ``influence`` is the peak close strength at a valley centre.
    """
    type_name = "closure"
    display_name = "Mouth closure"
    param_specs = [
        ParamSpec("detect_streams", "streamset", ["mediapipe", "ai"],
                  label="Detect in"),
        ParamSpec("drop", "float", 0.05, label="Detection dip", min=0.01, max=0.5),
        ParamSpec("open_min", "float", 0.08, label="Min open", min=0.0, max=0.5),
    ]

    def __init__(self, config) -> None:
        super().__init__(config)
        self._events: list = []

    def prepare(self, stream_frames: dict[str, tuple[list[dict], list[float]]]) -> None:
        drop = self.p("drop")
        open_min = self.p("open_min")
        events: list = []
        for name in self.p("detect_streams"):
            entry = stream_frames.get(name)
            if not entry:
                continue
            frames, positions = entry
            if not frames:
                continue
            events += closure_mod.detect_closures(
                frames, positions, drop=drop, open_min=open_min
            )
        self._events = closure_mod.merge_events(events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def status_text(self) -> str:
        return f"Closures detected: {self.event_count}"

    def apply(self, values, pose, ctx):
        if not self._events:
            return values, pose
        wet = closure_mod.enforce_closure(values, ctx.pos_ms, self._events, 1.0)
        return ctx.blend(values, wet, self.influence), pose


@register
class InputModifier(Modifier):
    """Select and mix a named stream into the signal.

    ``influence`` blends between the current values and the chosen stream.
    ``scope`` limits mixing to mouth-only channels or all channels.
    """
    type_name = "input"
    display_name = "Input stream"
    has_influence = True
    param_specs = [
        ParamSpec("stream", "enum", "mediapipe", label="Stream",
                  choices=[("mediapipe", "MediaPipe"), ("ai", "AI"),
                           ("retarget", "Retarget")]),
        ParamSpec("scope", "enum", ai_blendshapes.SCOPE_ALL, label="Scope",
                  choices=[(ai_blendshapes.SCOPE_ALL, "All channels"),
                           (ai_blendshapes.SCOPE_MOUTH, "Mouth only")]),
    ]

    def transform(self, values: list[float], ctx: ModifierContext) -> list[float]:
        s = ctx.streams.get(self.p("stream"))
        if s is None:
            return values
        scope = self.p("scope")
        if scope == ai_blendshapes.SCOPE_MOUTH:
            return ai_blendshapes.mix_blendshapes(
                values, s, scope, ai_blendshapes.MODE_REPLACE, 1.0
            )
        return list(s)  # SCOPE_ALL: full copy of all 52 channels

    def apply(self, values, pose, ctx):
        wet = self.transform(values, ctx)
        return ctx.blend(values, wet, self.influence), pose


@register
class PoseFilterModifier(Modifier):
    """Mute head-pose axes and choose the rotation pivot (neck anchor)."""
    type_name = "pose_filter"
    display_name = "Head pose"
    signal = "pose"
    has_influence = False
    param_specs = [
        ParamSpec("rot", "vec3bool", [True, True, True], label="Rotation X/Y/Z"),
        ParamSpec("pos", "vec3bool", [True, True, True], label="Position X/Y/Z"),
        ParamSpec("neck_anchor", "float", 0.0, label="Neck anchor", min=0.0, max=1.0),
    ]

    def transform_pose(self, pose: dict | None, ctx: ModifierContext) -> dict | None:
        if pose is None:
            return None
        rot = list(pose.get("rot", [0.0, 0.0, 0.0]))
        pos = list(pose.get("pos", [0.0, 0.0, 0.0]))
        rmask = self.p("rot")
        pmask = self.p("pos")
        for i in range(3):
            if not rmask[i]:
                rot[i] = 0.0
            if not pmask[i]:
                pos[i] = 0.0
        return {"rot": rot, "pos": pos, "neck_anchor": self.p("neck_anchor")}
