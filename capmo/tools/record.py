"""
Headless multi-phone recorder (no Qt): capture server + session writer + a
start/stop control.

    python -m capmo.tools.record --out DIR [--host H --port P] [--start]
        [--preset 720p60 | --width W --height H --fps F] [--wait] [--duration S]
        [--control-port 8765] [--convert] [--board-squares 5x8 ...]

* Phones dial ``host:port`` (default ``0.0.0.0:12345``); with ``--start`` each
  phone is told to START at the chosen resolution / frame rate when it connects.
* Recording starts immediately, or with ``--wait`` only when toggled:
  **Enter** or ``s`` + Enter on the terminal (when stdin is a tty; ``q`` quits),
  ``kill -USR1 <pid>``, or the HTTP control endpoint.
* ``--control-port N``: a tiny HTTP endpoint on ``--control-host`` (default
  127.0.0.1): ``POST /start``, ``POST /stop``, ``POST /toggle``, ``POST /quit``,
  ``GET /status`` (JSON).
* ``--duration S``: stop recording after S seconds of recording and exit.
* Ctrl+C / SIGTERM stop the recording cleanly (manifest written) and exit.
* Each recording is one session directory under ``--out`` (schema-2 manifest,
  see :mod:`capmo.core.session_io`); ``--convert`` then runs
  :mod:`capmo.tools.session_to_take` into ``<session>/videos/``.

A status line per connected phone is printed every 2 s, as ``python -m capmo
--headless`` always did (that mode now runs this code).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .. import protocol
from ..core.presets import PRESETS, preset as _preset
from ..recorder import RecorderController
from ..server import CaptureServer, local_ip


class Recorder:
    """Capture server + recorder with start/stop/toggle/status controls.

    Thread-safe: the controls may be called from signal handlers, the stdin
    thread or the HTTP thread.
    """

    def __init__(self, out_dir: str | None, *, host: str = "0.0.0.0", port: int = 12345,
                 width: int = 1280, height: int = 720, fps: int = 30,
                 start_phones: bool = False, board: dict | None = None,
                 preset: str | None = None, convert: bool = False,
                 server: CaptureServer | None = None, log=print) -> None:
        self.out_dir = os.path.abspath(out_dir) if out_dir else None
        self.width, self.height, self.fps = width, height, fps
        self.board, self.preset, self.convert = board, preset, convert
        self.start_phones = start_phones
        self.log = log
        self.server = server or CaptureServer(host=host, port=port)
        self.host = host
        self.rec = RecorderController(self.server, self.out_dir) if self.out_dir else None
        self.sessions: list[str] = []
        self.converted: dict[str, str] = {}
        self.quit_event = threading.Event()
        self._lock = threading.Lock()
        self._rec_started: float | None = None
        self._http: ThreadingHTTPServer | None = None
        self.server.on_connect = self._on_connect
        self.server.on_disconnect = lambda c: self.log(
            f"[capmo] - {c.label} ({c.phone_id}) gone")

    # -- server --------------------------------------------------------------

    def _on_connect(self, conn) -> None:
        self.log(f"[capmo] + {conn.label} ({conn.phone_id}) connected")
        if self.start_phones:
            conn.send(protocol.cmd_start(self.width, self.height))
            conn.send(protocol.cmd_framerate(self.fps))

    def open(self) -> None:
        self.server.start()
        self.log(f"[capmo] listening on {self.host}:{self.server.port} -- "
                 f"point each phone at  {local_ip()}:{self.server.port}")

    # -- controls ------------------------------------------------------------

    @property
    def recording(self) -> bool:
        return bool(self.rec and self.rec.recording)

    def session_info(self) -> dict:
        return {"nominal_fps": self.fps, "requested_size": [self.width, self.height],
                "preset": self.preset, "board": self.board,
                "recorder": {"tool": "capmo.tools.record", "pid": os.getpid()}}

    def start_recording(self, name: str | None = None) -> str | None:
        if self.rec is None:
            return None
        with self._lock:
            if self.rec.recording:
                return self.rec.session_dir
            d = self.rec.start(name=name, info=self.session_info())
            self._rec_started = time.time()
        self.log(f"[capmo] recording to {d}")
        return d

    def stop_recording(self) -> str | None:
        if self.rec is None:
            return None
        with self._lock:
            if not self.rec.recording:
                return None
            d = self.rec.stop()
            self._rec_started = None
        if d:
            self.sessions.append(d)
            self.log(f"[capmo] stopped; session written to {d}")
            if self.convert:
                self._convert(d)
        return d

    def toggle(self) -> str | None:
        return self.stop_recording() if self.recording else self.start_recording()

    def _convert(self, d: str) -> None:
        from . import session_to_take
        out = os.path.join(d, "videos")
        try:
            session_to_take.convert(d, out, log=self.log)
            self.converted[d] = out
        except Exception as exc:          # conversion must not lose the session
            self.log(f"[capmo] conversion failed ({type(exc).__name__}: {exc}); run "
                     f"python -m capmo.tools.session_to_take {d} --out {out}")

    def status(self) -> dict:
        conns = self.server.connections()
        return {
            "recording": self.recording,
            "session_dir": self.rec.session_dir if self.rec else None,
            "recording_s": (time.time() - self._rec_started) if self._rec_started else None,
            "sessions": list(self.sessions),
            "port": self.server.port,
            "phones": [{"label": c.label, "phone_id": c.phone_id, "fps": round(c.fps, 2),
                        "frames": c.frames, "clock_offset_ms": c.clock.offset_ms,
                        "rtt_ms": c.clock.rtt_ms, "synced": c.clock.ready} for c in conns],
        }

    def status_line(self) -> str | None:
        conns = self.server.connections()
        if not conns:
            return None
        parts = [f"{c.label}: {c.fps:4.1f}fps off={c.clock.offset_ms:+.0f}ms "
                 f"rtt={c.clock.rtt_ms:.0f} frames={c.frames}" for c in conns]
        tag = " [REC]" if self.recording else ""
        return "[capmo]" + tag + " " + " | ".join(parts)

    # -- HTTP control --------------------------------------------------------

    def serve_control(self, port: int, host: str = "127.0.0.1") -> int:
        rec = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip("/") in ("/status", ""):
                    self._send(200, rec.status())
                else:
                    self._send(404, {"error": "unknown path"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                p = self.path.rstrip("/")
                if p == "/start":
                    d = rec.start_recording()
                    self._send(200 if d else 409, {"ok": bool(d), "session_dir": d,
                                                   **rec.status()})
                elif p == "/stop":
                    d = rec.stop_recording()
                    self._send(200, {"ok": True, "stopped": d, **rec.status()})
                elif p == "/toggle":
                    rec.toggle()
                    self._send(200, {"ok": True, **rec.status()})
                elif p == "/quit":
                    self._send(200, {"ok": True})
                    rec.quit_event.set()
                else:
                    self._send(404, {"error": "unknown path"})

            def log_message(self, *a):        # keep the console for status lines
                pass

        self._http = ThreadingHTTPServer((host, port), Handler)
        self._http.daemon_threads = True
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        port = self._http.server_address[1]
        self.log(f"[capmo] control: http://{host}:{port}/status  (POST /start /stop /toggle /quit)")
        return port

    # -- lifecycle -----------------------------------------------------------

    def _stdin_loop(self) -> None:
        self.log("[capmo] keys: Enter or 's' = start/stop recording, 'q' = quit")
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in ("", "s"):
                self.toggle()
            elif cmd in ("q", "quit", "exit"):
                self.quit_event.set()
                return

    def install_signals(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        signal.signal(signal.SIGTERM, lambda *_: self.quit_event.set())
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, lambda *_: threading.Thread(
                target=self.toggle, daemon=True).start())

    def run(self, *, duration: float | None = None, status_every: float = 2.0,
            keys: bool | None = None) -> None:
        """Block until quit (signal, 'q', /quit, --duration). Ctrl+C quits."""
        if keys is None:
            keys = sys.stdin is not None and sys.stdin.isatty()
        if keys and self.rec is not None:
            threading.Thread(target=self._stdin_loop, daemon=True).start()
        last = time.time()
        try:
            while not self.quit_event.wait(0.1):
                if duration is not None and self._rec_started is not None \
                        and time.time() - self._rec_started >= duration:
                    self.log(f"[capmo] --duration {duration:g}s reached")
                    break
                if time.time() - last >= status_every:
                    last = time.time()
                    line = self.status_line()
                    if line:
                        self.log(line)
        except KeyboardInterrupt:
            self.log("\n[capmo] stopping...")

    def close(self) -> None:
        try:
            self.stop_recording()
        finally:
            if self._http:
                self._http.shutdown()
                self._http.server_close()
            self.server.stop()


def board_from_args(args) -> dict | None:
    if not (args.board_squares or args.board_square_len or args.board_marker_len):
        return None
    from ..calibration import BoardConfig
    cfg = BoardConfig()
    if args.board_squares:
        sx, sy = args.board_squares.lower().split("x")
        cfg.squares_x, cfg.squares_y = int(sx), int(sy)
    if args.board_square_len is not None:
        cfg.square_len = args.board_square_len
    if args.board_marker_len is not None:
        cfg.marker_len = args.board_marker_len
    return cfg.to_dict()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m capmo.tools.record",
                                description="Record capmo phones into session folders.")
    p.add_argument("--out", required=True, help="root folder for session directories")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=12345)
    p.add_argument("--start", action="store_true",
                   help="send START/FRAMERATE to each phone as it connects")
    p.add_argument("--preset", choices=[x[0] for x in PRESETS], default=None,
                   help="resolution@fps preset (overrides --width/--height/--fps)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--wait", action="store_true",
                   help="don't record until toggled (Enter / SIGUSR1 / POST /start)")
    p.add_argument("--name", default=None, help="session directory name (first recording)")
    p.add_argument("--duration", type=float, default=None,
                   help="stop after S seconds of recording and exit")
    p.add_argument("--control-port", type=int, default=None,
                   help="HTTP control endpoint port (0 = any free port)")
    p.add_argument("--control-host", default="127.0.0.1")
    p.add_argument("--convert", action="store_true",
                   help="run session_to_take into <session>/videos after each recording")
    p.add_argument("--no-keys", action="store_true", help="ignore stdin")
    p.add_argument("--board-squares", metavar="XxY", default=None,
                   help="ChArUco board used in this session (stored in the manifest)")
    p.add_argument("--board-square-len", type=float, default=None)
    p.add_argument("--board-marker-len", type=float, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def run_from_args(args, *, record_dir: str | None, arm: bool) -> int:
    """Shared by ``capmo.tools.record`` and ``python -m capmo --headless``."""
    w, h, fps = args.width, args.height, args.fps
    preset = getattr(args, "preset", None)
    if preset:
        w, h, fps = _preset(preset)
    rec = Recorder(record_dir, host=args.host, port=args.port, width=w, height=h, fps=fps,
                   start_phones=args.start, board=board_from_args(args), preset=preset,
                   convert=getattr(args, "convert", False))
    rec.install_signals()          # before "listening" is printed
    rec.open()
    try:
        if getattr(args, "control_port", None) is not None:
            rec.serve_control(args.control_port, getattr(args, "control_host", "127.0.0.1"))
        if arm and record_dir:
            rec.start_recording(getattr(args, "name", None))
        rec.run(duration=getattr(args, "duration", None),
                keys=False if getattr(args, "no_keys", False) else None)
    finally:
        rec.close()
    return 0


def main(argv=None) -> int:
    import logging
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    return run_from_args(args, record_dir=args.out, arm=not args.wait)


if __name__ == "__main__":
    sys.exit(main())
