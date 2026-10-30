"""capmo.tools.record: start/stop controls, HTTP endpoint, signals, --convert."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

from ..core import h264
from ..tools.record import Recorder
from .session_fixtures import encode_h264
from .test_server import MockPhone, _wait

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _http(port, method, path):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def _stream(phone, aus, t_ms):
    for k, au in enumerate(aus):
        if h264.is_idr(au):
            phone.send_keyframe(int(t_ms + k * 33.3), body=au)
        else:
            phone.send_delta(body=au)


@pytest.fixture
def recorder(tmp_path):
    from ..server import CaptureServer
    lines = []
    srv = CaptureServer(host="127.0.0.1", port=0, ping_interval=0.1)
    rec = Recorder(str(tmp_path), host="127.0.0.1", server=srv, fps=30, width=160,
                   height=96, start_phones=False, log=lines.append)
    rec.open()
    rec.lines = lines
    yield rec
    rec.close()


def test_controls_and_http_endpoint(recorder, tmp_path):
    aus = encode_h264(20, gop=10)
    phone = MockPhone("127.0.0.1", recorder.server.port, phone_lead_ms=250)
    try:
        assert _wait(lambda: len(recorder.server.connections()) == 1)
        assert phone.answer_one_ping()
        assert _wait(lambda: recorder.server.connections()[0].clock.ready)
        port = recorder.serve_control(0)
        code, st = _http(port, "GET", "/status")
        assert code == 200 and st["recording"] is False and len(st["phones"]) == 1
        code, st = _http(port, "POST", "/start")
        assert code == 200 and st["recording"] and st["session_dir"]
        d = st["session_dir"]
        recorder.rec.sample_clocks()
        _stream(phone, aus, time.time() * 1000 + 250)
        conn = recorder.server.connections()[0]
        assert _wait(lambda: conn.frames >= 20)
        code, st = _http(port, "POST", "/stop")
        assert st["stopped"] == d and not st["recording"] and st["sessions"] == [d]
        assert _http(port, "POST", "/stop")[1]["stopped"] is None          # idempotent
        # toggle twice via the Python API -> a second session
        d2 = recorder.toggle()
        assert recorder.recording and d2 != d
        assert recorder.toggle() == d2 and not recorder.recording
        assert recorder.status_line() and "frames=20" in recorder.status_line()
        with pytest.raises(urllib.error.HTTPError):
            _http(port, "GET", "/nope")
        _http(port, "POST", "/quit")
        assert recorder.quit_event.is_set()
    finally:
        phone.close()
    man = json.load(open(os.path.join(d, "manifest.json")))
    p = man["phones"][0]
    assert man["schema_version"] == 2 and man["nominal_fps"] == 30
    assert man["requested_size"] == [160, 96] and man["recorder"]["tool"] == "capmo.tools.record"
    assert p["frames"] == 20 and (p["width"], p["height"]) == (160, 96)
    assert p["clock_samples"] >= 1 and abs(p["clock_summary"]["offset_last_ms"] - 250) < 20
    from ..core.session_io import load_session
    c = load_session(d).cameras[0]
    assert c.times.method == "meta_anchors" and c.offset.method == "clock_history"


def test_duration_and_convert(recorder, tmp_path):
    aus = encode_h264(20, gop=10)
    recorder.convert = True
    phone = MockPhone("127.0.0.1", recorder.server.port, phone_lead_ms=0)
    try:
        assert _wait(lambda: len(recorder.server.connections()) == 1)
        assert phone.answer_one_ping()
        d = recorder.start_recording(name="dur")
        _stream(phone, aus, time.time() * 1000)
        assert _wait(lambda: recorder.server.connections()[0].frames >= 20)
        t = time.time()
        recorder.run(duration=0.3, keys=False)
        assert time.time() - t < 30      # only guards against a hang; the frame count below is the check
        assert recorder.stop_recording() == d
    finally:
        phone.close()
    take = json.load(open(os.path.join(d, "videos", "take.json")))
    assert take["videos"][0]["frames"] == 20
    assert os.path.isfile(os.path.join(d, "videos", take["videos"][0]["file"]))


def _spawn(args, tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.Popen([sys.executable, "-u", "-m", *args], cwd=str(tmp_path), env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def _read_until(proc, needle, timeout=20):
    deadline = time.time() + timeout
    out = []
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        out.append(line)
        if needle in line:
            return "".join(out)
    raise AssertionError(f"{needle!r} not seen in: {''.join(out)}")


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals")
def test_cli_signals(tmp_path):
    out = tmp_path / "rec"
    p = _spawn(["capmo.tools.record", "--out", str(out), "--host", "127.0.0.1", "--port", "0",
                "--wait", "--no-keys"], tmp_path)
    try:
        _read_until(p, "listening on")
        p.send_signal(signal.SIGUSR1)                     # toggle -> start
        _read_until(p, "recording to")
        p.send_signal(signal.SIGTERM)                     # clean stop + exit
        rest = p.communicate(timeout=20)[0]
        assert p.returncode == 0 and "session written to" in rest
    finally:
        if p.poll() is None:
            p.kill()
    (sess,) = os.listdir(out)
    assert os.path.isfile(out / sess / "manifest.json")


def test_headless_main_still_works(tmp_path):
    """``python -m capmo --headless`` delegates to the recorder; SIGINT stops it."""
    out = tmp_path / "cap"
    p = _spawn(["capmo", "--headless", "--host", "127.0.0.1", "--port", "0",
                "--record", str(out)], tmp_path)
    try:
        _read_until(p, "recording to")
        p.send_signal(signal.SIGINT)
        rest = p.communicate(timeout=20)[0]
        assert "session written to" in rest, rest
    finally:
        if p.poll() is None:
            p.kill()
    (sess,) = os.listdir(out)
    man = json.load(open(out / sess / "manifest.json"))
    assert man["schema_version"] == 2 and man["phones"] == []
