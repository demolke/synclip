"""
Blender SynClip client - receives ARKit blendshape data from the capture tool
and records it as Blender shape-key animation.

Usage (as an addon)
-------------------
1. Zip the ``blender`` folder (or use the bundled addon zip).
2. Edit -> Preferences -> Add-ons -> Install..., pick the zip, tick the checkbox.
3. Use the "SynClip" panel in the 3-D Viewport sidebar (N-panel).

It registers three operators and a panel:

  SYNCLIP_OT_connect      - connect to the capture tool (creates Action, finds mesh)
  SYNCLIP_OT_disconnect   - stop receiving data
  SYNCLIP_OT_simplify     - manually simplify keyframes (Douglas-Peucker per channel)

Keyframe cleanup never happens automatically. It only runs when the user
presses the "Clean Up Keyframes" button in the panel's Track Cleanup section.

Protocol (matches ipc_server.py)
---------------------------------
  244 bytes / frame, little-endian
  struct.pack('<I d 52f 3f 3f', magic, audio_pos_ms, *blendshapes52, *rot_xyz, *pos_xyz)
  magic = 0xAF0002
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np

try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 9876
FRAME_SIZE = 244
MAGIC = 0xAF0002

# Handshake messages (length-prefixed JSON; see ipc_server.py).
HELLO_MAGIC = 0xAF0001   # server -> client: announces blendshape names
REPORT_MAGIC = 0xAF0005  # client -> server: reports its mapping
_MSG_HEADER = struct.Struct("<I I")  # magic + payload length

# Frame-mode tags carried in the magic's low byte (see ipc_server.py).
MODE_LIVE = 0xAF0002    # live / recording -> insert keyframes, advancing
MODE_REVIEW = 0xAF0003  # playback / scrub -> move playhead only, no insert
MODE_EDIT = 0xAF0004    # value edit -> overwrite the keyframe at this position
_VALID_MODES = (MODE_LIVE, MODE_REVIEW, MODE_EDIT)

# Blendshape names in the EXACT order MediaPipe FaceLandmarker emits them, which
# is the order the capture tool packs into each IPC frame (see
# synclip/arkit_names.py). Index 0 is "_neutral" and MediaPipe does NOT
# emit "tongueOut" - so this list must keep _neutral at the front and omit
# tongueOut, or every channel shifts by one against the transmitted data.
ARKIT_NAMES = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]

assert len(ARKIT_NAMES) == 52

# ---------------------------------------------------------------------------
# Global receiver state (lives outside Blender's undo history)
# ---------------------------------------------------------------------------

_receiver: Optional["SynClipReceiver"] = None


def _get_fps() -> float:
    if _IN_BLENDER:
        return bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    return 24.0


def _tag_redraw() -> None:
    """Force the 3-D viewport sidebar to repaint so live counters update."""
    if not _IN_BLENDER:
        return
    wm = getattr(bpy.context, "window_manager", None)
    if not wm:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# ---------------------------------------------------------------------------
# Blender version-safe helpers
# ---------------------------------------------------------------------------

def _action_fcurves(action: "bpy.types.Action"):
    """Yield all FCurves from an Action, compatible with both Blender <=4 and 5+.

    Blender 5 replaced action.fcurves with a layered strip/channelbag system.
    """
    try:
        yield from action.fcurves  # Blender <= 4.x
    except AttributeError:
        for layer in action.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    yield from cb.fcurves


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------

_STRUCT = struct.Struct("<I d 52f 3f 3f")  # 4+8+208+12+12 = 244


def _parse_frame(data: bytes) -> Optional[tuple[int, float, list[float], list[float], list[float]]]:
    """Return (mode, audio_pos_ms, blendshapes52, rot_xyz, pos_xyz) or None."""
    if len(data) < FRAME_SIZE:
        return None
    fields = _STRUCT.unpack_from(data, 0)
    mode = fields[0]
    if mode not in _VALID_MODES:
        return None
    audio_pos_ms = fields[1]
    blendshapes = list(fields[2:54])
    rot = list(fields[54:57])
    pos = list(fields[57:60])
    return mode, audio_pos_ms, blendshapes, rot, pos


# ---------------------------------------------------------------------------
# Mesh / shape-key discovery
# ---------------------------------------------------------------------------

def _has_shape_keys(obj) -> bool:
    return (
        obj is not None
        and obj.type == "MESH"
        and obj.data.shape_keys is not None
        and len(obj.data.shape_keys.key_blocks) >= 10
    )


def _find_shape_key_object() -> Optional["bpy.types.Object"]:
    """Return the mesh to drive. Prefers the active object (so the user can pick
    a specific character in a multi-mesh scene by selecting it), otherwise the
    first mesh with at least 10 shape keys."""
    if not _IN_BLENDER:
        return None
    active = getattr(bpy.context, "active_object", None)
    if _has_shape_keys(active):
        return active
    for obj in bpy.data.objects:
        if _has_shape_keys(obj):
            return obj
    return None


def _normalize(name: str) -> str:
    """Strip separators and spaces for fuzzy matching."""
    return name.lower().replace("_", "").replace("-", "").replace(".", "").replace(" ", "")


def _arkit_variants(arkit: str) -> list[str]:
    """Return normalized lookup keys for one ARKit blendshape name.

    Covers common Blender naming conventions:
      - exact camelCase          e.g. browDownLeft
      - snake_case / dot.case   stripped by _normalize
      - L/R suffix abbreviation e.g. browDownL instead of browDownLeft
    """
    base = _normalize(arkit)
    variants = [base]
    for full, abbr in (("left", "l"), ("right", "r")):
        if base.endswith(full):
            variants.append(base[: -len(full)] + abbr)
    return variants


def _build_key_map(obj: "bpy.types.Object",
                   names: Optional[list[str]] = None) -> dict[int, str]:
    """Map blendshape index -> Blender shape-key name (case/separator-insensitive).

    *names* is the authoritative blendshape-name list in transmit order; it
    defaults to ARKIT_NAMES but should be the list negotiated from the server so
    index alignment can never drift from what the capture tool actually sends.

    Returns a dict {index: blender_key_name} for every successfully matched
    channel.
    """
    if not _IN_BLENDER or not obj.data.shape_keys:
        return {}
    if names is None:
        names = ARKIT_NAMES
    # Normalize every shape key name so separators/case don't matter.
    sk_norm: dict[str, str] = {_normalize(k.name): k.name
                                for k in obj.data.shape_keys.key_blocks}
    result: dict[int, str] = {}
    for i, name in enumerate(names):
        for variant in _arkit_variants(name):
            if variant in sk_norm:
                result[i] = sk_norm[variant]
                break
    return result


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class SynClipReceiver:
    """Manages the TCP socket and inserts Blender keyframes."""

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._obj: Optional["bpy.types.Object"] = None
        self._key_map: dict[int, str] = {}
        self._action: Optional["bpy.types.Action"] = None
        self._action_name = ""

        # Frame queue: main thread consumes via Blender timer.
        self._queue: list[tuple[float, list[float], list[float], list[float]]] = []
        self._lock = threading.Lock()

        self.status = "Disconnected"
        self.mapped_count = 0
        self.total_count = 52
        self.frames_received = 0

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        # Discover the target mesh first (cheap, no side effects), but do NOT
        # create the Action yet - only after the socket actually connects, so a
        # failed connect doesn't leave an orphan action in the .blend.
        if _IN_BLENDER:
            self._obj = _find_shape_key_object()
            if self._obj is None:
                self.status = "Error: no mesh with shape keys found"
                return False
            self._key_map = _build_key_map(self._obj)
            self.mapped_count = len(self._key_map)
            self.total_count = len(ARKIT_NAMES)
            print(
                f"[synclip] found '{self._obj.name}' with "
                f"{self.mapped_count}/{self.total_count} mapped shape keys"
            )

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect((self._host, self._port))
            self._sock.settimeout(None)
        except OSError as exc:
            self.status = f"Connect failed: {exc}"
            self._sock = None
            return False

        if _IN_BLENDER:
            # Socket is up - now create the Action and bind it to the mesh.
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._action_name = f"SynClipTake_{ts}"
            self._action = bpy.data.actions.new(self._action_name)
            sk = self._obj.data.shape_keys
            if sk.animation_data is None:
                sk.animation_data_create()
            sk.animation_data.action = self._action
            print(f"[SynClip] created action '{self._action_name}'")

        self._running = True
        self.frames_received = 0
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        if _IN_BLENDER and not bpy.app.timers.is_registered(self._process_queue):
            bpy.app.timers.register(self._process_queue, first_interval=0.033)
        self.status = f"Connected -> {self._action_name}"
        return True

    def disconnect(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if _IN_BLENDER and bpy.app.timers.is_registered(self._process_queue):
            try:
                bpy.app.timers.unregister(self._process_queue)
            except Exception:
                pass
        self.status = "Disconnected"

    # ------------------------------------------------------------------
    # Receive loop (background thread)
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        buf = b""
        hello_done = False
        while self._running:
            try:
                chunk = self._sock.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk

            # The very first message is the HELLO handshake: a length-prefixed
            # JSON list of blendshape names in transmit order. Build the mapping
            # from those authoritative names and report it back, then stream.
            if not hello_done:
                consumed = self._try_consume_hello(buf)
                if consumed < 0:
                    break  # malformed handshake - bail
                if consumed == 0:
                    continue  # need more bytes for the full HELLO
                buf = buf[consumed:]
                hello_done = True

            while len(buf) >= FRAME_SIZE:
                parsed = _parse_frame(buf[:FRAME_SIZE])
                buf = buf[FRAME_SIZE:]
                if parsed:
                    with self._lock:
                        self._enqueue(parsed)
        self._running = False
        self.status = "Disconnected (connection closed)"

    def _try_consume_hello(self, buf: bytes) -> int:
        """Parse the HELLO handshake at the head of *buf*.

        Returns the number of bytes consumed, 0 if more data is needed, or -1 on
        a malformed/unexpected header. On success it (re)builds the key map from
        the negotiated names and sends a mapping REPORT back to the server.
        """
        if len(buf) < _MSG_HEADER.size:
            return 0
        magic, length = _MSG_HEADER.unpack_from(buf, 0)
        if magic != HELLO_MAGIC:
            print(f"[SynClip] unexpected handshake magic 0x{magic:08X}")
            return -1
        if len(buf) < _MSG_HEADER.size + length:
            return 0
        payload = buf[_MSG_HEADER.size:_MSG_HEADER.size + length]
        try:
            names = list(json.loads(payload.decode("utf-8")).get("blendshapes", []))
        except (ValueError, UnicodeDecodeError) as exc:
            print(f"[SynClip] bad HELLO payload: {exc}")
            return -1

        self._negotiate_mapping(names)
        return _MSG_HEADER.size + length

    def _negotiate_mapping(self, names: list[str]) -> None:
        """Rebuild the key map from the server's names and report it back."""
        if not _IN_BLENDER or self._obj is None or not names:
            return
        self._key_map = _build_key_map(self._obj, names)
        self.mapped_count = len(self._key_map)
        self.total_count = len(names)
        print(f"[SynClip] negotiated {self.mapped_count}/{self.total_count} "
              f"shape keys from server name list")

        mapping = {names[i]: key for i, key in self._key_map.items()}
        unmapped = [n for i, n in enumerate(names)
                    if i not in self._key_map and n != "_neutral"]
        report = {
            "client": "blender",
            "object": self._obj.name,
            "mapped_count": self.mapped_count,
            "total": self.total_count,
            "mapping": mapping,
            "unmapped": unmapped,
        }
        self._send_report(report)

    def _send_report(self, report: dict) -> None:
        """Send a length-prefixed JSON mapping report to the server."""
        if self._sock is None:
            return
        try:
            payload = json.dumps(report).encode("utf-8")
            self._sock.sendall(_MSG_HEADER.pack(REPORT_MAGIC, len(payload)) + payload)
        except OSError as exc:
            print(f"[SynClip] could not send mapping report: {exc}")

    def _enqueue(self, parsed: tuple) -> None:
        """Queue a parsed frame (caller holds the lock).

        LIVE/record frames are never dropped - losing one would gap the recorded
        animation. REVIEW/EDIT frames only need the latest, so they coalesce:
        replace a trailing run of the same non-record mode rather than piling up.
        A safety cap guards against unbounded growth if the main thread stalls.
        """
        mode = parsed[0]
        if mode != MODE_LIVE and self._queue and self._queue[-1][0] == mode:
            self._queue[-1] = parsed  # coalesce consecutive scrub/edit frames
            return
        self._queue.append(parsed)
        if len(self._queue) > 2000:  # ~minutes of record backlog; just in case
            self._queue = self._queue[-2000:]

    # ------------------------------------------------------------------
    # Main-thread queue consumer (Blender timer callback)
    # ------------------------------------------------------------------

    def _process_queue(self) -> Optional[float]:
        """Called by Blender's timer on the main thread."""
        if not self._running:
            return None  # unregister timer

        with self._lock:
            frames = list(self._queue)
            self._queue.clear()

        if not _IN_BLENDER or not frames:
            return 0.033  # 30 fps poll

        fps = _get_fps()
        sk = self._obj.data.shape_keys if self._obj else None
        scrub_to: Optional[float] = None

        for mode, audio_pos_ms, blendshapes, _rot, _pos in frames:
            frame_num = round(audio_pos_ms * fps / 1000.0)

            if mode == MODE_REVIEW:
                # Playback / scrub: just remember where to move the playhead.
                scrub_to = frame_num
                continue

            if not (sk and self._key_map):
                continue

            # LIVE (record) and EDIT both write keyframes at frame_num. We avoid
            # scene.frame_set entirely - keyframe_insert(frame=...) is explicit and
            # forcing a full depsgraph update per frame would be far slower.
            for idx, key_name in self._key_map.items():
                if idx < len(blendshapes):
                    kb = sk.key_blocks.get(key_name)
                    if kb is not None:
                        kb.value = blendshapes[idx]
                        kb.keyframe_insert("value", frame=frame_num)
            if mode == MODE_EDIT:
                scrub_to = frame_num  # keep playhead where the user is editing
            self.frames_received += 1

        if scrub_to is not None:
            bpy.context.scene.frame_set(int(scrub_to))

        _tag_redraw()
        return 0.033

    # Note: scrub (MODE_REVIEW) and edit (MODE_EDIT) sync from the capture tool
    # are handled directly in _process_queue based on the incoming frame mode -
    # the capture tool sends review/edit frames over the same socket.

    # ------------------------------------------------------------------
    # Keyframe simplification (Douglas-Peucker per channel)
    # ------------------------------------------------------------------

    def simplify_keyframes(self, epsilon: float = 0.003) -> int:
        """Remove redundant keyframes using Douglas-Peucker, per shape-key channel.

        *epsilon* is the max allowed blendshape-value deviation. Returns the
        total number of keyframes removed.
        """
        if not _IN_BLENDER or not self._action:
            return 0

        total_removed = 0
        fcs = list(_action_fcurves(self._action))
        for fcurve in fcs:
            total_removed += _simplify_fcurve(fcurve, epsilon)
        print(f"[SynClip] simplify removed {total_removed} keyframes across {len(fcs)} channels")
        return total_removed


# ---------------------------------------------------------------------------
# Douglas-Peucker simplification for a single FCurve
# ---------------------------------------------------------------------------

def _simplify_fcurve(fcurve: "bpy.types.FCurve", epsilon: float) -> int:
    """Simplify *fcurve* in-place. Returns the number of keyframes removed."""
    kfps = fcurve.keyframe_points
    n = len(kfps)
    if n < 3:
        return 0

    # Extract (frame, value) arrays.
    frames = np.array([kf.co[0] for kf in kfps], dtype=np.float64)
    values = np.array([kf.co[1] for kf in kfps], dtype=np.float64)

    keep_mask = _douglas_peucker_mask(frames, values, epsilon)
    keep_indices = set(np.where(keep_mask)[0])

    # Remove in reverse order to keep indices stable.
    removed = 0
    for i in range(n - 1, -1, -1):
        if i not in keep_indices:
            kfps.remove(kfps[i])
            removed += 1
    return removed


def _douglas_peucker_mask(x: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    """Return a boolean mask of which points to keep.

    Uses *vertical* (value-axis) deviation, not 2-D perpendicular distance: the
    x axis is frame number (large) and y is the blendshape value (0..1), so a
    geometric distance would be dominated by frame spacing and *epsilon* would
    not bound the value error. Vertical deviation makes epsilon mean exactly
    "max allowed blendshape-value error", which is the intent.

    Iterative (explicit stack) to avoid Python recursion limits on long takes.
    """
    n = len(x)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    mask[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        x0, x1 = x[lo], x[hi]
        y0, y1 = y[lo], y[hi]
        seg = x[lo + 1:hi]
        # Linearly interpolated value of the (lo->hi) segment at each interior x.
        denom = (x1 - x0)
        if denom == 0:
            interp = np.full(seg.shape, y0)
        else:
            interp = y0 + (y1 - y0) * (seg - x0) / denom
        dists = np.abs(y[lo + 1:hi] - interp)
        rel = int(np.argmax(dists))
        if dists[rel] > epsilon:
            mid = lo + 1 + rel
            mask[mid] = True
            stack.append((lo, mid))
            stack.append((mid, hi))
    return mask


# ---------------------------------------------------------------------------
# Blender operators and panel
# ---------------------------------------------------------------------------

if _IN_BLENDER:
    class SYNCLIP_OT_connect(bpy.types.Operator):
        bl_idname = "synclip.connect"
        bl_label = "Connect"
        bl_description = "Connect to the SynClip capture tool and start recording keyframes"

        def execute(self, context):
            global _receiver
            if _receiver and _receiver._running:
                _receiver.disconnect()
            scene = context.scene
            host = getattr(scene, "synclip_host", HOST) or HOST
            port = getattr(scene, "synclip_port", PORT)
            _receiver = SynClipReceiver(host=host, port=port)
            if _receiver.connect():
                self.report(
                    {"INFO"},
                    f"Connected. Mapped {_receiver.mapped_count}/{_receiver.total_count} shape keys."
                )
            else:
                self.report({"ERROR"}, _receiver.status)
            return {"FINISHED"}

    class SYNCLIP_OT_disconnect(bpy.types.Operator):
        bl_idname = "synclip.disconnect"
        bl_label = "Disconnect"
        bl_description = "Disconnect from the capture tool"

        def execute(self, context):
            global _receiver
            if _receiver:
                _receiver.disconnect()
            self.report({"INFO"}, "Disconnected.")
            return {"FINISHED"}

    class SYNCLIP_OT_simplify(bpy.types.Operator):
        bl_idname = "synclip.simplify"
        bl_label = "Simplify Keyframes"
        bl_description = (
            "Remove redundant keyframes from the current SynClip action "
            "using Douglas-Peucker per channel"
        )

        epsilon: bpy.props.FloatProperty(
            name="Epsilon",
            description="Maximum allowed value deviation (smaller = more accurate, more keyframes)",
            default=0.003,
            min=0.0001,
            max=0.1,
            precision=4,
        )

        def invoke(self, context, event):
            return context.window_manager.invoke_props_dialog(self)

        def execute(self, context):
            global _receiver
            if not _receiver or not _receiver._action:
                self.report({"ERROR"}, "No active SynClip action - connect first.")
                return {"CANCELLED"}
            removed = _receiver.simplify_keyframes(self.epsilon)
            self.report({"INFO"}, f"Removed {removed} redundant keyframes.")
            return {"FINISHED"}

    class SYNCLIP_PT_panel(bpy.types.Panel):
        bl_label = "SynClip"
        bl_idname = "SYNCLIP_PT_panel"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "SynClip"

        def draw(self, context):
            layout = self.layout
            global _receiver
            connected = _receiver is not None and _receiver._running

            # --- Connection / recording section ---
            if connected:
                layout.label(text=_receiver.status, icon="CHECKMARK")
                layout.label(
                    text=f"Shape keys: {_receiver.mapped_count}/{_receiver.total_count} mapped"
                )
                layout.label(text=f"Frames received: {_receiver.frames_received}")
                layout.operator("synclip.disconnect", icon="X")
            else:
                status = _receiver.status if _receiver else "Disconnected"
                layout.label(text=status, icon="INFO")
                col = layout.column(align=True)
                col.prop(context.scene, "synclip_host", text="Host")
                col.prop(context.scene, "synclip_port", text="Port")
                layout.operator("synclip.connect", icon="PLAY")

            # --- Separate, manual Track Cleanup section ---
            # Never runs automatically: the user must press this button.
            have_action = _receiver is not None and _receiver._action is not None
            box = layout.box()
            box.label(text="Track Cleanup (manual)", icon="IPO_EASE_IN_OUT")
            if have_action:
                box.label(text=f"Action: {_receiver._action_name}")
                row = box.row()
                row.scale_y = 1.4
                row.operator("synclip.simplify", text="Clean Up Keyframes")
            else:
                box.label(text="Record a take first.", icon="INFO")

    _CLASSES = [
        SYNCLIP_OT_connect,
        SYNCLIP_OT_disconnect,
        SYNCLIP_OT_simplify,
        SYNCLIP_PT_panel,
    ]

    def register():
        for cls in _CLASSES:
            bpy.utils.register_class(cls)
        # Server target, editable in the panel and remembered with the .blend.
        bpy.types.Scene.synclip_host = bpy.props.StringProperty(
            name="Host",
            description="Host/IP of the SynClip capture tool's IPC server",
            default=HOST,
        )
        bpy.types.Scene.synclip_port = bpy.props.IntProperty(
            name="Port",
            description="TCP port of the SynClip capture tool's IPC server",
            default=PORT,
            min=1,
            max=65535,
        )

    def unregister():
        # Tear down any live connection so disabling the addon is clean.
        global _receiver
        if _receiver is not None:
            try:
                _receiver.disconnect()
            except Exception:
                pass
            _receiver = None
        for prop in ("synclip_host", "synclip_port"):
            if hasattr(bpy.types.Scene, prop):
                delattr(bpy.types.Scene, prop)
        for cls in reversed(_CLASSES):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass

    # Allow running this file directly from the Scripting workspace as well as
    # loading it as an addon (where tools.blender.__init__ registers it).
    if __name__ == "__main__":
        register()
else:
    # Outside Blender (e.g. unit tests importing this module): provide no-op
    # register/unregister so __init__.register() doesn't raise AttributeError.
    def register():  # type: ignore[misc]
        pass

    def unregister():  # type: ignore[misc]
        pass
