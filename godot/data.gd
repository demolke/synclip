## Static helpers for reading/writing .synclip.json files.
class_name SynClipData


static func synclip_path(audio_path: String) -> String:
	return audio_path.get_basename() + ".synclip.json"


static func load_synclip(audio_path: String) -> Dictionary:
	var path := synclip_path(audio_path)
	if not FileAccess.file_exists(path):
		return {}
	var file := FileAccess.open(path, FileAccess.READ)
	if not file:
		return {}
	var content := file.get_as_text()
	file.close()
	var json := JSON.new()
	if json.parse(content) != OK:
		push_warning("data: failed to parse JSON: " + path)
		return {}
	var data = json.get_data()
	if not data is Dictionary:
		push_warning("data: root is not a Dictionary: " + path)
		return {}
	if not data.has("takes") or not data["takes"] is Array:
		push_warning("data: missing 'takes' array: " + path)
		data["takes"] = []
	return data


static func save_synclip(audio_path: String, data: Dictionary) -> void:
	var path := synclip_path(audio_path)
	var content := JSON.stringify(data, "\t")
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file:
		file.store_string(content)
		file.close()


static func get_default_scales() -> Dictionary:
	var scales: Dictionary = {}
	for name in ArkitNames.NAMES:
		scales[name] = 1.0
	return scales


static func get_take(data: Dictionary, take_id: String) -> Dictionary:
	for take in data.get("takes", []):
		if take.get("take_id", "") == take_id:
			return take
	return {}
