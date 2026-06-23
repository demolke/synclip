# Headless Godot harness: decode a Python-packed 244-byte synclip frame
# using the exact byte offsets main.gd uses, and write the decoded values back
# out as JSON. The Python test compares the result against the known input, so
# this verifies the GDScript wire-format offsets match the server's struct.
#
# Run:
#   SYNCLIP_FRAME_FILE=/path/in.bin SYNCLIP_OUT_FILE=/path/out.json \
#       godot --headless --script frame_parse_test.gd
#
# Exit code 0 on success, non-zero on any failure.
extends SceneTree

const HELLO_MAGIC := 0xAF0001

func _initialize() -> void:
	var in_path := OS.get_environment("SYNCLIP_FRAME_FILE")
	var out_path := OS.get_environment("SYNCLIP_OUT_FILE")
	if in_path == "" or out_path == "":
		push_error("SYNCLIP_FRAME_FILE / SYNCLIP_OUT_FILE not set")
		quit(2)
		return

	var f := FileAccess.open(in_path, FileAccess.READ)
	if f == null:
		push_error("cannot open frame file: %s" % in_path)
		quit(3)
		return
	var frame := f.get_buffer(f.get_length())
	f.close()

	if frame.size() != 244:
		push_error("expected 244 bytes, got %d" % frame.size())
		quit(4)
		return

	# Same decode as main.gd._parse_tcp_frame.
	var magic := frame.decode_u32(0)
	var audio_pos := frame.decode_double(4)
	var values := []
	var off := 12
	for i in 52:
		values.append(frame.decode_float(off))
		off += 4
	var rot := [frame.decode_float(off), frame.decode_float(off + 4), frame.decode_float(off + 8)]
	off += 12
	var pos := [frame.decode_float(off), frame.decode_float(off + 4), frame.decode_float(off + 8)]

	var result := {
		"magic": magic,
		"audio_pos": audio_pos,
		"blendshapes": values,
		"rot": rot,
		"pos": pos,
	}

	var out := FileAccess.open(out_path, FileAccess.WRITE)
	if out == null:
		push_error("cannot write out file: %s" % out_path)
		quit(5)
		return
	out.store_string(JSON.stringify(result))
	out.close()
	quit(0)
