## SynClip Viewer - main controller.
##
## Modes:
##   LIVE     - receive 244-byte binary frames from Python over TCP.
##   PLAYBACK - drive head from recorded .synclip.json take data + audio.
##
## TCP frame format (little-endian):
##   uint32  magic = 0xAF00xx  (low byte = frame mode; viewer just displays)
##   float64 audio_pos_ms
##   float32 x 52 blendshape weights
##   float32 x 3  head rotation (euler degrees x, y, z)
##   float32 x 3  head translation (x, y, z)
##   TOTAL   244 bytes

extends Node

const FRAME_SIZE := 244
# The capture tool tags frames with a mode in the magic's low byte
# (0xAF0002 live / 0xAF0003 review / 0xAF0004 edit). The viewer treats all of
# them the same - just display - so it matches on the high 3 bytes.
const MAGIC_HI  := 0xAF0000
const MAGIC_MASK := 0xFFFFFF00
# Handshake messages (length-prefixed JSON, see ipc_server.py).
const HELLO_MAGIC  := 0xAF0001  # server -> client: blendshape name list
const REPORT_MAGIC := 0xAF0005  # client -> server: mapping report
const IPC_HOST_DEFAULT := "127.0.0.1"
const IPC_PORT_DEFAULT := 9876
# Settable in UI and overridable via --host/--port args
var _ipc_host := IPC_HOST_DEFAULT
var _ipc_port := IPC_PORT_DEFAULT

enum Mode { LIVE, PLAYBACK }

# -- State ---------------------------------------------------------------------
var _mode: Mode = Mode.LIVE
var _current_dir: String = ""
var _audio_files: Array[String] = []
var _current_audio_idx: int = -1
var _current_audio_path: String = ""
var _data: Dictionary = {}
var _current_take_id: String = ""
var _blend_scales: Array = []      # 52 floats
var _scales_modified: bool = false

# -- TCP (LIVE mode) -----------------------------------------------------------
var _tcp: StreamPeerTCP = StreamPeerTCP.new()
var _tcp_reconnect_timer: float = 0.0
const TCP_RECONNECT_INTERVAL := 2.0
var _rx_buf: PackedByteArray = PackedByteArray()  # persistent receive buffer
var _live_values: Array = []                      # reused 52-float scratch array
var _server_names: Array = []                     # blendshape names from HELLO
var _handshake_done: bool = false                 # HELLO consumed this connection

# -- 3D scene refs -------------------------------------------------------------
var _viewport_container: SubViewportContainer
var _viewport: SubViewport
var _camera: Camera3D
var _head_root: Node3D
var _head_mesh: MeshInstance3D
var _blend_shape_map: Dictionary  # shape name -> index in mesh
# Precomputed [source_value_index, mesh_blend_index] pairs for the per-frame
# hot path, so _apply_blendshapes avoids name/dict lookups every frame.
var _blend_apply_pairs: Array = []

# -- Head movement -------------------------------------------------------------
var _head_rot_x_enabled := true   # per-axis rotation toggles
var _head_rot_y_enabled := true
var _head_rot_z_enabled := true
var _head_move_enabled := false                      # translation toggle
var _head_rot_scale := Vector3(1.0, 1.0, 1.0)        # per-axis rotation multiplier
var _head_move_scale := Vector3(1.0, 1.0, 1.0)       # per-axis translation multiplier
const _HEAD_MOVE_BASE := 0.01                        # translation units -> metres
var _head_pos_ref: Variant = null                    # baseline translation (first frame)
var _rot_x_chk: CheckBox
var _rot_y_chk: CheckBox
var _rot_z_chk: CheckBox
var _move_chk: CheckBox

# -- UI node refs --------------------------------------------------------------
var _dir_label: Label
var _mode_button: Button
var _play_btn: Button
var _loop_btn: Button
var _status_label: Label
var _host_edit: LineEdit
var _port_spin: SpinBox
var _file_list: ItemList
var _take_list: ItemList
var _blend_scale_sliders: Array = []
var _blend_scale_val_labels: Array = []

# -- Audio ---------------------------------------------------------------------
var _audio_player: AudioStreamPlayer


func _ready() -> void:
	_blend_scales.resize(52)
	_blend_scales.fill(1.0)
	_live_values.resize(52)
	_live_values.fill(0.0)

	# Grab the 3D nodes that live in the .tscn before building the UI,
	# so _build_viewport_panel() can reparent the container into the split.
	_viewport_container = $ViewportContainer
	_viewport = $ViewportContainer/SubViewport
	_camera = $ViewportContainer/SubViewport/Camera3D
	_head_root = $ViewportContainer/SubViewport/Head

	# Parse --host/--port (and a positional directory) before building the UI
	var start_dir := _parse_cmdline_args()

	_build_ui()
	_find_head_mesh()

	_tcp.connect_to_host(_ipc_host, _ipc_port)

	if start_dir == "" or not DirAccess.dir_exists_absolute(start_dir):
		var home := OS.get_environment("HOME")
		start_dir = home if home != "" else "."
	_set_directory(start_dir)


## Parse user command-line args (those after `--`). Sets _ipc_host/_ipc_port from
## --host/--port (both `--flag value` and `--flag=value` forms) and returns the
## first non-flag argument as the start directory (or "" if none).
func _parse_cmdline_args() -> String:
	var args := OS.get_cmdline_user_args()
	var start_dir := ""
	var i := 0
	while i < args.size():
		var arg: String = args[i]
		if arg == "--host" and i + 1 < args.size():
			_ipc_host = args[i + 1]; i += 2; continue
		elif arg.begins_with("--host="):
			_ipc_host = arg.substr(7); i += 1; continue
		elif arg == "--port" and i + 1 < args.size():
			_ipc_port = int(args[i + 1]); i += 2; continue
		elif arg.begins_with("--port="):
			_ipc_port = int(arg.substr(7)); i += 1; continue
		elif not arg.begins_with("-") and start_dir == "":
			start_dir = arg
		i += 1
	if _ipc_host == "":
		_ipc_host = IPC_HOST_DEFAULT
	return start_dir


func _process(delta: float) -> void:
	match _mode:
		Mode.LIVE:     _process_live(delta)
		Mode.PLAYBACK: _process_playback()


# --- TCP / LIVE ---------------------------------------------------------------

func _process_live(delta: float) -> void:
	_tcp.poll()
	var status := _tcp.get_status()

	match status:
		StreamPeerTCP.STATUS_NONE, StreamPeerTCP.STATUS_ERROR:
			_reconnect_after(delta)
			_status_label.text = "LIVE - waiting for Python server on %s:%d" % [_ipc_host, _ipc_port]
			return
		StreamPeerTCP.STATUS_CONNECTING:
			# Guard against a socket stuck in CONNECTING (server never accepts):
			# force a fresh socket after the same interval used for retries.
			_tcp_reconnect_timer += delta
			if _tcp_reconnect_timer >= TCP_RECONNECT_INTERVAL * 2.0:
				_reconnect_after(0.0, true)
			_status_label.text = "LIVE - connecting..."
			return

	_tcp_reconnect_timer = 0.0
	_status_label.text = "LIVE - connected"

	# Drain all available bytes into a persistent buffer, then process whole
	# frames out of it. Buffering (rather than reading fixed 244-byte chunks
	# straight off the socket) is what lets us resync if the stream ever
	# desyncs: on a bad magic we drop a single byte and re-scan.
	var avail := _tcp.get_available_bytes()
	if avail > 0:
		var got := _tcp.get_data(avail)
		if got[0] == OK:
			_rx_buf.append_array(got[1])

	var budget := 10
	while budget > 0 and _rx_buf.size() >= 4:
		var magic := _rx_buf.decode_u32(0)
		# Handshake HELLO: length-prefixed JSON, must be consumed before frames.
		if magic == HELLO_MAGIC:
			if _rx_buf.size() < 8:
				break  # need the length field
			var length := _rx_buf.decode_u32(4)
			if _rx_buf.size() < 8 + length:
				break  # wait for the full payload
			var payload := _rx_buf.slice(8, 8 + length)
			_rx_buf = _rx_buf.slice(8 + length)
			_handle_hello(payload)
			continue
		if _rx_buf.size() < FRAME_SIZE:
			break
		if (magic & MAGIC_MASK) != MAGIC_HI:
			# Not on a frame boundary - discard one byte and try to re-align.
			_rx_buf = _rx_buf.slice(1)
			continue
		_parse_tcp_frame(_rx_buf)
		_rx_buf = _rx_buf.slice(FRAME_SIZE)
		budget -= 1

	# Don't let the buffer grow without bound if we ever fall behind.
	if _rx_buf.size() > FRAME_SIZE * 64:
		_rx_buf = _rx_buf.slice(_rx_buf.size() - FRAME_SIZE * 64)


## Read the host/port fields and force an immediate reconnect to the new target.
func _apply_server_target() -> void:
	var h := _host_edit.text.strip_edges()
	if h == "":
		h = IPC_HOST_DEFAULT
		_host_edit.text = h
	_ipc_host = h
	_ipc_port = int(_port_spin.value)
	_status_label.text = "Reconnecting to %s:%d ..." % [_ipc_host, _ipc_port]
	_reconnect_after(0.0, true)


func _reconnect_after(delta: float, force: bool = false) -> void:
	_tcp_reconnect_timer += delta
	if force or _tcp_reconnect_timer >= TCP_RECONNECT_INTERVAL:
		_tcp_reconnect_timer = 0.0
		_tcp.disconnect_from_host()  # release the old peer before replacing it
		_tcp = StreamPeerTCP.new()
		_tcp.connect_to_host(_ipc_host, _ipc_port)
		_rx_buf.clear()
		_handshake_done = false  # expect a fresh HELLO on the new connection


func _parse_tcp_frame(frame: PackedByteArray) -> void:
	# frame[0..3] = magic (already validated by caller). Decode by fixed offset.
	@warning_ignore("unused_variable")
	var audio_pos: float = frame.decode_double(4)
	var off := 12
	for i in 52:
		_live_values[i] = frame.decode_float(off)
		off += 4
	_apply_blendshapes(_live_values)
	var rot := Vector3(frame.decode_float(off), frame.decode_float(off + 4), frame.decode_float(off + 8))
	off += 12
	var pos := Vector3(frame.decode_float(off), frame.decode_float(off + 4), frame.decode_float(off + 8))
	_apply_head_pose(rot, pos)


func _handle_hello(payload: PackedByteArray) -> void:
	# Parse the server's authoritative blendshape-name list, rebuild the mapping
	# from it (so index alignment can never drift from what's transmitted), and
	# report the result back so the server can log it.
	var json := JSON.new()
	if json.parse(payload.get_string_from_utf8()) != OK:
		push_warning("[godot] could not parse HELLO handshake")
		return
	var data: Dictionary = json.data
	_server_names = data.get("blendshapes", [])
	_handshake_done = true
	_build_blend_shape_map()
	_send_mapping_report()


func _send_mapping_report() -> void:
	if _tcp.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return
	var names: Array = _server_names if not _server_names.is_empty() else ArkitNames.NAMES
	var mapping: Dictionary = {}
	var unmapped: Array = []
	for src in names.size():
		var nm: String = names[src]
		if nm == "_neutral":
			continue
		if nm in _blend_shape_map:
			var idx: int = _blend_shape_map[nm]
			mapping[nm] = _head_mesh.mesh.get_blend_shape_name(idx) if is_instance_valid(_head_mesh) else nm
		else:
			unmapped.append(nm)
	var report := {
		"client": "godot",
		"object": _head_mesh.name if is_instance_valid(_head_mesh) else "?",
		"mapped_count": mapping.size(),
		"total": names.size(),
		"mapping": mapping,
		"unmapped": unmapped,
	}
	var body := JSON.stringify(report).to_utf8_buffer()
	var msg := PackedByteArray()
	msg.resize(8)
	msg.encode_u32(0, REPORT_MAGIC)
	msg.encode_u32(4, body.size())
	msg.append_array(body)
	_tcp.put_data(msg)


# --- Playback -----------------------------------------------------------------

func _process_playback() -> void:
	if not _audio_player.playing:
		return
	var pos_ms: float = _audio_player.get_playback_position() * 1000.0
	var take := SynClipData.get_take(_data, _current_take_id)
	if take.is_empty():
		return
	var frames: Array = SynClipData.get_take_frames(take)
	if frames.is_empty():
		return
	var values := _interpolate_frames(frames, pos_ms)
	_apply_blendshapes(values)
	var pose := _interpolate_head_pose(frames, pos_ms)
	if not pose.is_empty():
		_apply_head_pose(pose["rot"], pose["pos"])
	_status_label.text = "PLAYBACK  %s  %.0f ms" % [_current_take_id, pos_ms]


func _interpolate_frames(frames: Array, pos_ms: float) -> Array:
	if frames.size() == 1:
		return frames[0]["blendshapes"]

	# Binary search: first index with audio_position_ms >= pos_ms
	var lo := 0
	var hi := frames.size() - 1
	while lo < hi:
		var mid := (lo + hi) / 2
		if float(frames[mid]["audio_position_ms"]) < pos_ms:
			lo = mid + 1
		else:
			hi = mid

	if lo == 0:
		return frames[0]["blendshapes"]
	if lo >= frames.size():
		return frames[-1]["blendshapes"]

	var f0: Dictionary = frames[lo - 1]
	var f1: Dictionary = frames[lo]
	var t0 := float(f0["audio_position_ms"])
	var t1 := float(f1["audio_position_ms"])
	var dt := t1 - t0
	var t := clampf((pos_ms - t0) / dt, 0.0, 1.0) if dt > 0.0 else 0.0

	var bs0: Array = f0["blendshapes"]
	var bs1: Array = f1["blendshapes"]
	var result: Array = []
	result.resize(min(bs0.size(), bs1.size()))
	for i in result.size():
		result[i] = lerpf(float(bs0[i]), float(bs1[i]), t)
	return result


func _interpolate_head_pose(frames: Array, pos_ms: float) -> Dictionary:
	# Returns {"rot": Vector3, "pos": Vector3} or {} if no pose data present.
	if frames.is_empty() or not (frames[0] as Dictionary).has("head_pose"):
		return {}

	var lo := 0
	var hi := frames.size() - 1
	while lo < hi:
		var mid := (lo + hi) / 2
		if float(frames[mid]["audio_position_ms"]) < pos_ms:
			lo = mid + 1
		else:
			hi = mid

	if lo == 0:
		return _pose_to_vectors(frames[0]["head_pose"])
	if lo >= frames.size():
		return _pose_to_vectors(frames[-1]["head_pose"])

	var f0: Dictionary = frames[lo - 1]
	var f1: Dictionary = frames[lo]
	var t0 := float(f0["audio_position_ms"])
	var t1 := float(f1["audio_position_ms"])
	var dt := t1 - t0
	var t := clampf((pos_ms - t0) / dt, 0.0, 1.0) if dt > 0.0 else 0.0
	var p0 := _pose_to_vectors(f0["head_pose"])
	var p1 := _pose_to_vectors(f1["head_pose"])
	return {
		"rot": (p0["rot"] as Vector3).lerp(p1["rot"], t),
		"pos": (p0["pos"] as Vector3).lerp(p1["pos"], t),
	}


func _pose_to_vectors(pose: Dictionary) -> Dictionary:
	var rot: Array = pose.get("rot", [0.0, 0.0, 0.0])
	var pos: Array = pose.get("pos", [0.0, 0.0, 0.0])
	return {
		"rot": Vector3(float(rot[0]), float(rot[1]), float(rot[2])),
		"pos": Vector3(float(pos[0]), float(pos[1]), float(pos[2])),
	}


# --- Blendshape application ---------------------------------------------------

func _apply_blendshapes(values: Array) -> void:
	if not is_instance_valid(_head_mesh):
		return
	var n := values.size()
	for pair in _blend_apply_pairs:
		var src: int = pair[0]
		var mesh_idx: int = pair[1]
		var raw := float(values[src]) if src < n else 0.0
		var scaled := clampf(raw * float(_blend_scales[src]), 0.0, 1.0)
		_head_mesh.set_blend_shape_value(mesh_idx, scaled)


func _apply_head_pose(rot_deg: Vector3, pos: Vector3) -> void:
	# Drives the head root node from the tracked head pose, honouring the
	# per-axis rotation toggles and the movement toggle. Disabled axes stay 0.
	if not is_instance_valid(_head_root):
		return

	var r := Vector3(
		rot_deg.x * _head_rot_scale.x if _head_rot_x_enabled else 0.0,
		rot_deg.y * _head_rot_scale.y if _head_rot_y_enabled else 0.0,
		rot_deg.z * _head_rot_scale.z if _head_rot_z_enabled else 0.0,
	)
	_head_root.rotation_degrees = r

	if _head_move_enabled:
		# Translate relative to the first observed position so the head stays
		# centred and only the *delta* movement is shown.
		if _head_pos_ref == null:
			_head_pos_ref = pos
		var delta: Vector3 = (pos - (_head_pos_ref as Vector3)) * _HEAD_MOVE_BASE
		_head_root.position = Vector3(
			delta.x * _head_move_scale.x,
			delta.y * _head_move_scale.y,
			delta.z * _head_move_scale.z,
		)
	else:
		_head_root.position = Vector3.ZERO
		_head_pos_ref = null


func _find_head_mesh() -> void:
	# Search the entire viewport subtree for a MeshInstance3D that has blend shapes.
	var queue: Array = [_viewport]
	while queue.size() > 0:
		var node = queue.pop_front()
		if node is MeshInstance3D:
			var mi := node as MeshInstance3D
			if is_instance_valid(mi.mesh) and mi.mesh.get_blend_shape_count() > 0:
				_head_mesh = mi
				_build_blend_shape_map()
				var total: int = mi.mesh.get_blend_shape_count()
				var mapped: int = _blend_shape_map.size()
				print("[godot] Found head mesh: %s  (%d blend shapes, %d/%d ARKit mapped)" % [
					mi.get_path(), total, mapped, 52
				])
				_status_label.text = "Head mesh: %s - %d blend shapes, %d/52 ARKit mapped" % [
					mi.name, total, mapped
				]
				return
		for child in node.get_children():
			queue.append(child)
	push_error("[godot] No MeshInstance3D with blend shapes found in the scene. Add an ARKit head mesh to the SubViewport.")
	_status_label.text = "ERROR: No head mesh with blend shapes found in scene!"


# Common prefixes that rigs attach before the bare ARKit name.
const _NAME_PREFIXES := [
	"blendShape1.", "blendShape.", "ARKit.", "Blendshape.", "blendshape.",
	"head_", "Head_", "face_", "Face_",
]


static func _normalize_blend_name(raw: String) -> String:
	"""Strip known rig prefixes and return a lowercase ARKit candidate."""
	for prefix in _NAME_PREFIXES:
		if raw.begins_with(prefix):
			return raw.substr(prefix.length())
	return raw


func _build_blend_shape_map() -> void:
	_blend_shape_map.clear()
	if not is_instance_valid(_head_mesh) or not is_instance_valid(_head_mesh.mesh):
		return

	# Prefer the server-negotiated name list (authoritative transmit order); fall
	# back to the local ArkitNames when no handshake has happened yet.
	var names: Array = _server_names if not _server_names.is_empty() else ArkitNames.NAMES

	# Build a lower-case lookup from the canonical names.
	var arkit_lower: Dictionary = {}
	for name in names:
		arkit_lower[String(name).to_lower()] = name

	var mesh_res := _head_mesh.mesh
	for i in mesh_res.get_blend_shape_count():
		var raw: String = mesh_res.get_blend_shape_name(i)
		var candidate := _normalize_blend_name(raw)
		# Normalise _L / _R suffix to Left / Right so that mesh conventions
		# like "browDown_L" match the ARKit canonical name "browDownLeft".
		if candidate.ends_with("_L"):
			candidate = candidate.left(candidate.length() - 2) + "Left"
		elif candidate.ends_with("_R"):
			candidate = candidate.left(candidate.length() - 2) + "Right"
		if candidate.to_lower() in arkit_lower:
			var arkit_name: String = arkit_lower[candidate.to_lower()]
			_blend_shape_map[arkit_name] = i

	# Precompute apply pairs (source value index -> mesh blend index) so the
	# per-frame path skips name/dict lookups. Skip _neutral (index 0).
	_blend_apply_pairs.clear()
	for src in names.size():
		var arkit: String = names[src]
		if arkit == "_neutral" or arkit not in _blend_shape_map:
			continue
		_blend_apply_pairs.append([src, int(_blend_shape_map[arkit])])
	if _blend_apply_pairs.is_empty():
		push_warning("[godot] No ARKit blendshapes mapped - the face will not move.")


# --- Directory / file browser -------------------------------------------------

func _set_directory(path: String) -> void:
	_current_dir = path
	_dir_label.text = path
	_refresh_file_list()


func _refresh_file_list() -> void:
	_audio_files.clear()
	_file_list.clear()
	if not DirAccess.dir_exists_absolute(_current_dir):
		return
	var dir := DirAccess.open(_current_dir)
	if not dir:
		return
	dir.list_dir_begin()
	var fname := dir.get_next()
	while fname != "":
		if not dir.current_is_dir():
			var ext := fname.get_extension().to_lower()
			if ext in ["wav", "ogg", "mp3"]:
				_audio_files.append(fname)
		fname = dir.get_next()
	dir.list_dir_end()
	_audio_files.sort()
	for f in _audio_files:
		_file_list.add_item(f)
	if _audio_files.size() > 0:
		_file_list.select(0)
		_on_file_selected(0)


func _on_file_selected(idx: int) -> void:
	if idx < 0 or idx >= _audio_files.size():
		return
	_audio_player.stop()
	_current_audio_idx = idx
	_current_audio_path = _current_dir.path_join(_audio_files[idx])
	_data = SynClipData.load_synclip(_current_audio_path)
	_refresh_takes_list()
	var default_id: String = _data.get("default_take", "")
	if default_id != "":
		_select_take(default_id)
	_load_audio(_current_audio_path)


func _navigate_files(delta: int) -> void:
	if _audio_files.is_empty():
		return
	var new_idx := clampi(_current_audio_idx + delta, 0, _audio_files.size() - 1)
	if new_idx != _current_audio_idx:
		_file_list.select(new_idx)
		_on_file_selected(new_idx)


# --- Takes panel --------------------------------------------------------------

func _refresh_takes_list() -> void:
	_take_list.clear()
	if _data.is_empty():
		return
	var default_id: String = _data.get("default_take", "")
	for take in _data.get("takes", []):
		var tid: String = take.get("take_id", "")
		var star := " *" if tid == default_id else "  "
		var ts: String = take.get("timestamp_utc", "")
		var date_part := ts.substr(0, 16).replace("T", " ") if ts.length() >= 16 else ""
		var frame_count: int = SynClipData.get_take_frames(take).size()
		_take_list.add_item("%s%s  %s  [%d frames]" % [tid, star, date_part, frame_count])
		_take_list.set_item_metadata(_take_list.item_count - 1, tid)


func _select_take(take_id: String) -> void:
	_current_take_id = take_id
	for i in _take_list.item_count:
		if _take_list.get_item_metadata(i) == take_id:
			_take_list.select(i)
			break
	_load_scales_for_take(take_id)
	_scales_modified = false


func _on_take_selected(idx: int) -> void:
	if idx < 0 or idx >= _take_list.item_count:
		return
	_select_take(_take_list.get_item_metadata(idx))


func _navigate_takes(delta: int) -> void:
	if _take_list.item_count == 0:
		return
	var sel := _take_list.get_selected_items()
	var cur := sel[0] if sel.size() > 0 else 0
	var new_idx := clampi(cur + delta, 0, _take_list.item_count - 1)
	_take_list.select(new_idx)
	_on_take_selected(new_idx)


func _set_default_take() -> void:
	if _current_take_id.is_empty() or _current_audio_path.is_empty():
		return
	_data["default_take"] = _current_take_id
	for take in _data.get("takes", []):
		take["is_default"] = take.get("take_id", "") == _current_take_id
	SynClipData.save_synclip(_current_audio_path, _data)
	_refresh_takes_list()


func _delete_take() -> void:
	if _current_take_id.is_empty() or _current_audio_path.is_empty():
		return
	# Confirm - deleting is destructive and Del is easy to hit by accident.
	var dialog := ConfirmationDialog.new()
	dialog.dialog_text = "Delete %s? This cannot be undone." % _current_take_id
	dialog.confirmed.connect(func() -> void:
		_do_delete_take()
		dialog.queue_free())
	dialog.canceled.connect(dialog.queue_free)
	add_child(dialog)
	dialog.popup_centered()


func _do_delete_take() -> void:
	if _current_take_id.is_empty() or _current_audio_path.is_empty():
		return
	var takes: Array = _data.get("takes", [])
	takes = takes.filter(func(t: Dictionary) -> bool:
		return t.get("take_id", "") != _current_take_id)
	_data["takes"] = takes
	if _data.get("default_take", "") == _current_take_id:
		_data["default_take"] = \
			(takes[0] as Dictionary).get("take_id", "") if takes.size() > 0 else ""
	SynClipData.save_synclip(_current_audio_path, _data)
	_current_take_id = ""
	_refresh_takes_list()


# --- Blend scales -------------------------------------------------------------

func _load_scales_for_take(take_id: String) -> void:
	var take := SynClipData.get_take(_data, take_id)
	var saved: Dictionary = take.get("blend_scales", {})
	for i in 52:
		var name: String = ArkitNames.NAMES[i]
		var val := float(saved.get(name, 1.0))
		_blend_scales[i] = val
		if i < _blend_scale_sliders.size():
			(_blend_scale_sliders[i] as HSlider).set_value_no_signal(val)
			(_blend_scale_val_labels[i] as Label).text = "%.2f" % val


func _on_scale_changed(value: float, channel_idx: int) -> void:
	_blend_scales[channel_idx] = value
	if channel_idx < _blend_scale_val_labels.size():
		(_blend_scale_val_labels[channel_idx] as Label).text = "%.2f" % value
	_scales_modified = true


func _save_scales() -> void:
	if _current_take_id.is_empty() or not _scales_modified or _current_audio_path.is_empty():
		return
	for take in _data.get("takes", []):
		if take.get("take_id", "") == _current_take_id:
			var scales: Dictionary = {}
			for i in 52:
				scales[ArkitNames.NAMES[i]] = _blend_scales[i]
			take["blend_scales"] = scales
			break
	SynClipData.save_synclip(_current_audio_path, _data)
	_scales_modified = false
	_status_label.text = "Scales saved for %s" % _current_take_id


func _reset_scales() -> void:
	# Reset in memory only - do NOT force-write to disk. The user can preview the
	# reset and press Save (or switch takes to discard) like any other edit.
	for i in 52:
		_blend_scales[i] = 1.0
		if i < _blend_scale_sliders.size():
			(_blend_scale_sliders[i] as HSlider).set_value_no_signal(1.0)
			(_blend_scale_val_labels[i] as Label).text = "1.00"
	_scales_modified = true
	_status_label.text = "Scales reset (not saved - press Save to keep)"


# --- Audio --------------------------------------------------------------------

func _load_audio(path: String) -> void:
	var stream := _load_audio_stream(path)
	if stream:
		_audio_player.stream = stream


func _load_audio_stream(path: String) -> AudioStream:
	var ext := path.get_extension().to_lower()
	match ext:
		"ogg":
			var data := FileAccess.get_file_as_bytes(path)
			if data.is_empty():
				push_error("Cannot read OGG: " + path)
				return null
			return AudioStreamOggVorbis.load_from_buffer(data)
		"mp3":
			var data := FileAccess.get_file_as_bytes(path)
			if data.is_empty():
				push_error("Cannot read MP3: " + path)
				return null
			var stream := AudioStreamMP3.new()
			stream.data = data
			return stream
		"wav":
			return _load_wav(path)
	push_warning("Unsupported audio format: " + ext)
	return null


func _load_wav(path: String) -> AudioStreamWAV:
	var file := FileAccess.open(path, FileAccess.READ)
	if not file:
		push_error("Cannot open WAV: " + path)
		return null

	var riff_id := file.get_buffer(4).get_string_from_ascii()
	if riff_id != "RIFF":
		push_error("Not a RIFF file: " + path)
		file.close()
		return null
	file.get_32()  # file size - 8
	var wave_id := file.get_buffer(4).get_string_from_ascii()
	if wave_id != "WAVE":
		push_error("Not a WAVE file: " + path)
		file.close()
		return null

	var channels := 1
	var sample_rate := 44100
	var bits_per_sample := 16
	var data_bytes := PackedByteArray()

	while not file.eof_reached():
		var chunk_id := file.get_buffer(4).get_string_from_ascii()
		var chunk_size: int = file.get_32()
		if chunk_size <= 0:
			break
		if chunk_id == "fmt ":
			file.get_16()             # audio_format (1 = PCM)
			channels = file.get_16()
			sample_rate = file.get_32()
			file.get_32()             # byte rate
			file.get_16()             # block align
			bits_per_sample = file.get_16()
			if chunk_size > 16:
				file.get_buffer(chunk_size - 16)
		elif chunk_id == "data":
			data_bytes = file.get_buffer(chunk_size)
		else:
			file.get_buffer(chunk_size)
	file.close()

	if data_bytes.is_empty():
		push_error("No PCM data in WAV: " + path)
		return null

	var fmt: int
	match bits_per_sample:
		8:  fmt = AudioStreamWAV.FORMAT_8_BITS
		16: fmt = AudioStreamWAV.FORMAT_16_BITS
		_:
			push_warning("WAV: unsupported bit depth %d in %s - treating as 16-bit" % [bits_per_sample, path])
			fmt = AudioStreamWAV.FORMAT_16_BITS
	var stream := AudioStreamWAV.new()
	stream.data = data_bytes
	stream.format = fmt
	stream.stereo = channels == 2
	stream.mix_rate = sample_rate
	return stream


func _play_loop() -> void:
	if not _audio_player.stream:
		return
	var s := _audio_player.stream
	if s is AudioStreamWAV:
		(s as AudioStreamWAV).loop_mode = AudioStreamWAV.LOOP_FORWARD
	elif s is AudioStreamOggVorbis:
		(s as AudioStreamOggVorbis).loop = true
	elif s is AudioStreamMP3:
		(s as AudioStreamMP3).loop = true
	_audio_player.play()


func _play_once() -> void:
	if not _audio_player.stream:
		return
	var s := _audio_player.stream
	if s is AudioStreamWAV:
		(s as AudioStreamWAV).loop_mode = AudioStreamWAV.LOOP_DISABLED
	elif s is AudioStreamOggVorbis:
		(s as AudioStreamOggVorbis).loop = false
	elif s is AudioStreamMP3:
		(s as AudioStreamMP3).loop = false
	_audio_player.play()


func _on_audio_finished() -> void:
	_status_label.text = "Playback finished - %s" % _current_take_id


# --- Mode ---------------------------------------------------------------------

func _set_mode(m: Mode) -> void:
	_mode = m
	if m == Mode.LIVE:
		_mode_button.text = "Mode: LIVE"
		_play_btn.disabled = true
		_loop_btn.disabled = true
		_audio_player.stop()
	else:
		_mode_button.text = "Mode: PLAYBACK"
		_play_btn.disabled = false
		_loop_btn.disabled = false


func _toggle_mode() -> void:
	_set_mode(Mode.PLAYBACK if _mode == Mode.LIVE else Mode.LIVE)


# --- Input --------------------------------------------------------------------

func _unhandled_input(event: InputEvent) -> void:
	# _unhandled_input runs only after focused controls (SpinBox, FileDialog,
	# ItemList...) have had their chance, so typing in a field no longer triggers
	# play/navigate. Extra guard: ignore keys while a control holds focus.
	if not event is InputEventKey:
		return
	var key := event as InputEventKey
	if not key.pressed or key.echo:
		return
	var focus_owner := get_viewport().gui_get_focus_owner()
	if focus_owner != null and not (focus_owner is ItemList):
		return
	match key.keycode:
		KEY_LEFT:
			_navigate_files(-1)
		KEY_RIGHT:
			_navigate_files(1)
		KEY_UP:
			_navigate_takes(-1)
		KEY_DOWN:
			_navigate_takes(1)
		KEY_SPACE:
			if _mode == Mode.PLAYBACK:
				if _audio_player.playing:
					_audio_player.stop()
				else:
					_play_once()
		KEY_L:
			if _mode == Mode.PLAYBACK:
				_play_loop()
		KEY_ENTER, KEY_KP_ENTER:
			_set_default_take()
		KEY_DELETE:
			_delete_take()
		KEY_ESCAPE:
			_audio_player.stop()
		KEY_S:
			if key.ctrl_pressed:
				_save_scales()


# --- UI construction ----------------------------------------------------------

func _build_ui() -> void:
	var canvas := CanvasLayer.new()
	add_child(canvas)

	var root := Control.new()
	root.name = "UI"
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	canvas.add_child(root)

	# Top bar
	var top := _make_top_bar()
	top.set_anchor_and_offset(SIDE_LEFT,   0,  0)
	top.set_anchor_and_offset(SIDE_RIGHT,  1,  0)
	top.set_anchor_and_offset(SIDE_TOP,    0,  0)
	top.set_anchor_and_offset(SIDE_BOTTOM, 0, 36)
	root.add_child(top)

	# Main splitter
	var split := HSplitContainer.new()
	split.name = "MainSplit"
	split.set_anchor_and_offset(SIDE_LEFT,   0,   0)
	split.set_anchor_and_offset(SIDE_RIGHT,  1,   0)
	split.set_anchor_and_offset(SIDE_TOP,    0,  36)
	split.set_anchor_and_offset(SIDE_BOTTOM, 1, -24)
	root.add_child(split)

	split.add_child(_build_file_browser())
	split.add_child(_build_viewport_panel())
	split.add_child(_build_right_panel())

	# Status bar
	_status_label = Label.new()
	_status_label.name = "Status"
	_status_label.set_anchor_and_offset(SIDE_LEFT,   0,  0)
	_status_label.set_anchor_and_offset(SIDE_RIGHT,  1,  0)
	_status_label.set_anchor_and_offset(SIDE_TOP,    1, -24)
	_status_label.set_anchor_and_offset(SIDE_BOTTOM, 1,  0)
	_status_label.text = "Ready  |  Shortcuts: <-/-> file  Up/Down take  Space play  L loop  Enter set default  Del delete  Ctrl+S save scales"
	root.add_child(_status_label)

	# Audio player
	_audio_player = AudioStreamPlayer.new()
	_audio_player.finished.connect(_on_audio_finished)
	add_child(_audio_player)


func _make_top_bar() -> HBoxContainer:
	var bar := HBoxContainer.new()
	bar.name = "TopBar"

	var open_btn := Button.new()
	open_btn.text = "Open Dir..."
	open_btn.pressed.connect(_on_open_dir)
	bar.add_child(open_btn)

	_dir_label = Label.new()
	_dir_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_dir_label.clip_text = true
	bar.add_child(_dir_label)

	_mode_button = Button.new()
	_mode_button.text = "Mode: LIVE"
	_mode_button.toggle_mode = false
	_mode_button.pressed.connect(_toggle_mode)
	bar.add_child(_mode_button)

	_play_btn = Button.new()
	_play_btn.text = "Play once"
	_play_btn.disabled = true
	_play_btn.pressed.connect(_play_once)
	bar.add_child(_play_btn)

	_loop_btn = Button.new()
	_loop_btn.text = "Loop"
	_loop_btn.disabled = true
	_loop_btn.pressed.connect(_play_loop)
	bar.add_child(_loop_btn)

	var stop_btn := Button.new()
	stop_btn.text = "Stop"
	stop_btn.pressed.connect(func() -> void: _audio_player.stop())
	bar.add_child(stop_btn)

	# Server target (host:port) - editable, with a Connect button to (re)connect.
	bar.add_child(VSeparator.new())
	var srv_lbl := Label.new()
	srv_lbl.text = "Server:"
	bar.add_child(srv_lbl)
	_host_edit = LineEdit.new()
	_host_edit.text = _ipc_host
	_host_edit.custom_minimum_size = Vector2(110, 0)
	_host_edit.tooltip_text = "Host/IP of the SynClip capture tool's IPC server"
	_host_edit.text_submitted.connect(func(_t): _apply_server_target())
	bar.add_child(_host_edit)
	_port_spin = SpinBox.new()
	_port_spin.min_value = 1
	_port_spin.max_value = 65535
	_port_spin.step = 1
	_port_spin.value = _ipc_port
	_port_spin.custom_minimum_size = Vector2(80, 0)
	_port_spin.tooltip_text = "TCP port of the SynClip capture tool's IPC server"
	bar.add_child(_port_spin)
	var connect_btn := Button.new()
	connect_btn.text = "Connect"
	connect_btn.tooltip_text = "Apply the host/port and (re)connect to the capture tool"
	connect_btn.pressed.connect(_apply_server_target)
	bar.add_child(connect_btn)

	bar.add_child(VSeparator.new())
	var head_lbl := Label.new()
	head_lbl.text = "Head rot:"
	bar.add_child(head_lbl)

	_rot_x_chk = _make_head_checkbox("X", true, func(on): _head_rot_x_enabled = on)
	_rot_x_chk.tooltip_text = "Enable head pitch (rotation about X)"
	bar.add_child(_rot_x_chk)
	bar.add_child(_make_scale_spin(func(v): _head_rot_scale.x = v, "X rotation scale (multiplier)"))
	_rot_y_chk = _make_head_checkbox("Y", true, func(on): _head_rot_y_enabled = on)
	_rot_y_chk.tooltip_text = "Enable head yaw (rotation about Y)"
	bar.add_child(_rot_y_chk)
	bar.add_child(_make_scale_spin(func(v): _head_rot_scale.y = v, "Y rotation scale (multiplier)"))
	_rot_z_chk = _make_head_checkbox("Z", true, func(on): _head_rot_z_enabled = on)
	_rot_z_chk.tooltip_text = "Enable head roll (rotation about Z)"
	bar.add_child(_rot_z_chk)
	bar.add_child(_make_scale_spin(func(v): _head_rot_scale.z = v, "Z rotation scale (multiplier)"))

	bar.add_child(VSeparator.new())
	_move_chk = _make_head_checkbox("Move", false, func(on):
		_head_move_enabled = on
		_head_pos_ref = null)
	_move_chk.tooltip_text = "Enable head translation (delta from the first frame)"
	bar.add_child(_move_chk)
	var move_lbl := Label.new()
	move_lbl.text = "xyz:"
	bar.add_child(move_lbl)
	bar.add_child(_make_scale_spin(func(v): _head_move_scale.x = v, "X movement scale (multiplier)"))
	bar.add_child(_make_scale_spin(func(v): _head_move_scale.y = v, "Y movement scale (multiplier)"))
	bar.add_child(_make_scale_spin(func(v): _head_move_scale.z = v, "Z movement scale (multiplier)"))

	return bar


func _make_head_checkbox(text: String, on: bool, cb: Callable) -> CheckBox:
	var chk := CheckBox.new()
	chk.text = text
	chk.button_pressed = on
	chk.toggled.connect(cb)
	return chk


func _make_scale_spin(cb: Callable, tooltip: String = "") -> SpinBox:
	# Per-axis scale multiplier for a head rotation / movement channel.
	var spin := SpinBox.new()
	spin.min_value = 0.0
	spin.max_value = 5.0
	spin.step = 0.1
	spin.value = 1.0
	spin.custom_minimum_size = Vector2(60, 0)
	spin.tooltip_text = tooltip
	spin.value_changed.connect(cb)
	return spin


func _build_file_browser() -> VBoxContainer:
	var vbox := VBoxContainer.new()
	vbox.name = "FileBrowser"
	vbox.custom_minimum_size = Vector2(200, 0)

	var lbl := Label.new()
	lbl.text = "Audio Files"
	lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(lbl)

	_file_list = ItemList.new()
	_file_list.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_file_list.item_selected.connect(_on_file_selected)
	vbox.add_child(_file_list)

	return vbox


func _build_viewport_panel() -> VBoxContainer:
	var vbox := VBoxContainer.new()
	vbox.name = "ViewportPanel"
	vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vbox.size_flags_vertical   = Control.SIZE_EXPAND_FILL

	# The SubViewportContainer is defined in main.tscn; reparent it here.
	# stretch = true makes the SubViewport resize to the container, so the 3D
	# scene scales with the window instead of staying a fixed resolution.
	_viewport_container.stretch = true
	_viewport_container.size_flags_vertical   = Control.SIZE_EXPAND_FILL
	_viewport_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_viewport_container.reparent(vbox)

	return vbox


func _build_right_panel() -> VSplitContainer:
	var vsplit := VSplitContainer.new()
	vsplit.name = "RightPanel"
	vsplit.custom_minimum_size = Vector2(300, 0)

	# -- Takes section ---------------------------------------------------------
	var takes_vbox := VBoxContainer.new()
	takes_vbox.name = "TakesSection"
	takes_vbox.custom_minimum_size = Vector2(0, 180)

	var takes_lbl := Label.new()
	takes_lbl.text = "Takes"
	takes_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	takes_vbox.add_child(takes_lbl)

	_take_list = ItemList.new()
	_take_list.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_take_list.item_selected.connect(_on_take_selected)
	takes_vbox.add_child(_take_list)

	var take_btns := HBoxContainer.new()
	var def_btn := Button.new()
	def_btn.text = "* Set Default"
	def_btn.pressed.connect(_set_default_take)
	take_btns.add_child(def_btn)

	var del_btn := Button.new()
	del_btn.text = "Delete"
	del_btn.pressed.connect(_delete_take)
	take_btns.add_child(del_btn)
	takes_vbox.add_child(take_btns)

	vsplit.add_child(takes_vbox)

	# -- Blend scales section --------------------------------------------------
	var scales_vbox := VBoxContainer.new()
	scales_vbox.name = "ScalesSection"
	scales_vbox.size_flags_vertical = Control.SIZE_EXPAND_FILL

	var hdr := HBoxContainer.new()
	var scales_lbl := Label.new()
	scales_lbl.text = "Blend Shape Scales (per take)"
	scales_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	hdr.add_child(scales_lbl)

	var reset_btn := Button.new()
	reset_btn.text = "Reset"
	reset_btn.pressed.connect(_reset_scales)
	hdr.add_child(reset_btn)

	var save_btn := Button.new()
	save_btn.text = "Save"
	save_btn.pressed.connect(_save_scales)
	hdr.add_child(save_btn)
	scales_vbox.add_child(hdr)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scales_vbox.add_child(scroll)

	var sliders_vbox := VBoxContainer.new()
	sliders_vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(sliders_vbox)

	for i in 52:
		var name: String = ArkitNames.NAMES[i]
		var row := HBoxContainer.new()

		var name_lbl := Label.new()
		name_lbl.text = name
		name_lbl.custom_minimum_size = Vector2(155, 0)
		name_lbl.clip_text = true
		row.add_child(name_lbl)

		var slider := HSlider.new()
		slider.min_value = 0.0
		slider.max_value = 1.0
		slider.value = 1.0
		slider.step = 0.01
		slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		slider.custom_minimum_size = Vector2(60, 0)
		slider.value_changed.connect(_on_scale_changed.bind(i))
		row.add_child(slider)
		_blend_scale_sliders.append(slider)

		var val_lbl := Label.new()
		val_lbl.text = "1.00"
		val_lbl.custom_minimum_size = Vector2(38, 0)
		val_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		row.add_child(val_lbl)
		_blend_scale_val_labels.append(val_lbl)

		sliders_vbox.add_child(row)

	vsplit.add_child(scales_vbox)
	return vsplit


# --- Button / dialog handlers -------------------------------------------------

func _on_open_dir() -> void:
	var dialog := FileDialog.new()
	dialog.file_mode = FileDialog.FILE_MODE_OPEN_DIR
	dialog.access = FileDialog.ACCESS_FILESYSTEM
	dialog.dir_selected.connect(func(path: String) -> void:
		_set_directory(path)
		dialog.queue_free()
	)
	dialog.canceled.connect(dialog.queue_free)
	add_child(dialog)
	dialog.popup_centered_ratio(0.6)
