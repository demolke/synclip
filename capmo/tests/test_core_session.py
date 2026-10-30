"""capmo.core: H.264 SPS parsing, per-frame timeline model, session schema/reader."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from ..core import h264, timeline as tl
from ..core.session_io import SCHEMA_VERSION, load_session, parse_cam_ids
from ..session import RecordingSession
from .session_fixtures import CamSpec, encode_h264, fabricate_session


# ------------------------------------------------------------------ h264
@pytest.mark.parametrize("w,h", [(160, 96), (162, 98), (640, 360)])
def test_sps_size_matches_encoder(w, h):
    aus = encode_h264(2, w, h, gop=2)
    sps = h264.parse_sps(aus[0])
    assert sps and (sps["width"], sps["height"]) == (w, h)
    assert sps["fps"] == pytest.approx(30.0)
    assert h264.is_idr(aus[0]) and h264.has_slice(aus[1])
    cfg = h264.codec_config(aus[0])
    assert h264.nal_types(cfg) == [h264.NAL_SPS, h264.NAL_PPS]


def test_nal_scan_without_sps():
    assert h264.parse_sps(b"\x00\x00\x01\x65abc") is None
    assert h264.nal_types(b"\x00\x00\x00\x01\x67x\x00\x00\x01\x68y\x00\x00\x01\x65z") == [7, 8, 5]


# -------------------------------------------------------------- timeline
def _synthetic_stream(seed=1, N=900, fps=30.0, drops=(100, 401, 402), gop=30,
                      offset=1234.5, meta=True):
    rng = np.random.default_rng(seed)
    P = 1000.0 / fps
    slots = np.delete(np.arange(N + len(drops)), list(drops))[:N].astype(float)
    true_phone = 1.7e12 + slots * P + rng.normal(0, 0.5, N)
    true_host = true_phone - offset
    lat = 5 + rng.exponential(12, N) + (rng.random(N) < 0.05) * rng.uniform(20, 60, N)
    recv = np.maximum.accumulate(true_host + lat)
    m = np.full(N, np.nan)
    if meta:
        key = np.arange(0, N, gop)
        m[key] = np.round(true_phone[key])
    samples = [(true_host[i], offset + rng.normal(0, 1.0), 4 + rng.exponential(3))
               for i in range(0, N, 30)]
    return true_host, recv, m, samples


def test_timeline_accuracy_on_synthetic_stream():
    """Quantifies the per-frame timestamp model (numbers quoted in the docs).

    30 fps, 0.5 ms capture jitter, integer-ms META every 30th frame, 3 encoder
    drops, 5..~100 ms arrival latency, clock samples with 1 ms noise / 4+ ms rtt.
    """
    true_host, recv, meta, samples = _synthetic_stream()
    om = tl.offset_model(samples)
    assert om.method == "clock_history" and abs(om.a - 1234.5) < 0.6
    ft = tl.host_frame_times(recv, meta, om, nominal_fps=30)
    err = ft.t_ms - true_host
    rms, mx = float(np.sqrt(np.mean(err ** 2))), float(np.abs(err).max())
    assert ft.method == "meta_anchors" and ft.dropped_slots == 3
    assert rms < 1.5 and mx < 4.0, (rms, mx)
    env, _ = tl.recv_envelope(recv, 1000 / 30)
    env_rms = float(np.sqrt(np.mean((env - true_host) ** 2)))
    raw_rms = float(np.sqrt(np.mean((recv - true_host) ** 2)))
    assert rms * 3 < env_rms < raw_rms


def test_timeline_rejects_bad_anchor_and_extrapolates():
    true_host, recv, meta, samples = _synthetic_stream(drops=(), N=300)
    meta[150] = meta[150] + 400.0                      # one corrupt META
    meta[0] = np.nan                                   # first keyframe without META
    ft = tl.host_frame_times(recv, meta, tl.offset_model(samples), nominal_fps=30)
    assert ft.n_anchors_rejected == 1
    err = ft.t_ms - true_host
    assert np.abs(err).max() < 4.0


def test_timeline_fallbacks():
    true_host, recv, meta, samples = _synthetic_stream(meta=False, N=300, drops=())
    ft = tl.host_frame_times(recv, meta, None, nominal_fps=30)
    assert ft.method == "recv_envelope"
    d = np.diff(ft.t_ms)
    assert np.all(d > 0) and np.std(d) < 3.0           # smooth, monotonic
    om = tl.offset_model(None, keyframe_pairs=[(1000.0, 1100.0), (2000.0, 2101.0)])
    assert om.method == "keyframe_server_ms" and om.a == pytest.approx(100.5)
    assert tl.offset_model(None, None, 12.0).method == "manifest"
    assert tl.offset_model(None, None, None, [(1000.0, 1050.0), (2000.0, 2060.0)]).a == 60.0
    assert tl.offset_model() is None


def test_offset_model_drift():
    t = np.linspace(0, 600_000, 300)
    samples = [(ti, 50 + 20e-6 * ti, 5.0) for ti in t]
    om = tl.offset_model(samples)
    assert om.method == "clock_history_drift"
    assert om.to_dict()["drift_ppm"] == pytest.approx(20.0, rel=1e-3)
    assert float(om.to_host(600_000 + 62.0)) == pytest.approx(600_000, abs=1e-3)


# ---------------------------------------------------------------- session
SPECS = [CamSpec("10.0.0.11", offset_ms=250.0, start_ms=0.0, config="separate"),
         CamSpec("10.0.0.12", offset_ms=-1800.0, start_ms=11.0, config="inband", drop=(17,)),
         CamSpec("10.0.0.13", offset_ms=40.0, start_ms=23.0, config="midstream")]


@pytest.fixture(scope="module")
def fab(tmp_path_factory):
    root = tmp_path_factory.mktemp("sessions")
    return fabricate_session(str(root), SPECS, n=40, fps=30, gop=10,
                             info={"nominal_fps": 30, "requested_size": [160, 96],
                                   "preset": "test", "board": {"squares_x": 5}})


def test_manifest_schema_v2(fab):
    man = json.load(open(os.path.join(fab.dir, "manifest.json")))
    assert man["schema_version"] == SCHEMA_VERSION
    assert man["nominal_fps"] == 30 and man["board"] == {"squares_x": 5}
    assert man["stopped_ms"] >= man["created_ms"]
    for p in man["phones"]:
        # schema-1 keys are all still there
        for k in ("phone_id", "slug", "h264", "index", "frames", "bytes", "first_recv_ms",
                  "last_recv_ms", "clock_offset_ms"):
            assert k in p
        assert p["cam_id"] == p["slug"]
        assert (p["width"], p["height"]) == (160, 96)
        assert p["fps_nominal"] == pytest.approx(30.0)
        assert 20 < p["fps_measured"] < 40
        assert p["rotation_deg_ccw"] is None
        assert p["clock"] == f"{p['slug']}.clock.jsonl" and p["clock_samples"] >= 2
        assert p["codec_config_b64"]
        assert os.path.isfile(os.path.join(fab.dir, p["clock"]))


def test_load_session_frame_times(fab):
    sess = load_session(fab.dir)
    assert sess.schema_version == 2 and len(sess.cameras) == 3
    by = {c.phone_id: c for c in sess.cameras}
    for s in SPECS:
        c = by[s.phone_id]
        truth = fab.truth[s.phone_id][fab.first_k[s.phone_id]:]
        assert c.n_frames == len(truth)
        assert c.times.method == "meta_anchors"
        assert c.offset.method == "clock_history"
        assert abs(c.offset.a - s.offset_ms) < 1.0
        err = c.t_ms - truth
        assert np.abs(err).max() < 4.0, (s.phone_id, np.abs(err).max())
        assert (c.width, c.height) == (160, 96)
    sep = by["10.0.0.11"]
    assert len(sep.config_rows) == 1 and sep.first_decodable == 0
    mid = by["10.0.0.13"]
    assert mid.first_decodable == 7          # recorded from k=3, next IDR at k=10
    assert by["10.0.0.12"].times.dropped_slots == 1
    rep = sess.sync_report()
    assert rep["pair_uncertainty_ms"] is not None and rep["pair_uncertainty_ms"] < 10
    assert sess.t0_ms == pytest.approx(min(c.decodable_t_ms[0] for c in sess.cameras))


def test_access_units_decode(fab):
    av = pytest.importorskip("av")
    from .session_fixtures import code_for, code_of
    sess = load_session(fab.dir)
    for c in sess.cameras:
        cc = av.CodecContext.create("h264", "r")
        codes = []
        for k, i, data in c.access_units():
            pkt = av.Packet(data)
            for fr in cc.decode(pkt):
                codes.append(code_of(fr.to_ndarray(format="bgr24")))
        for fr in cc.decode(None):
            codes.append(code_of(fr.to_ndarray(format="bgr24")))
        k0 = fab.first_k[c.phone_id] + c.first_decodable
        exp = [code_for(k) for k in range(k0, k0 + len(codes))]
        assert len(codes) == c.n_frames - c.first_decodable
        d = np.array(codes) - exp                 # (a constant few-level codec shift)
        assert np.abs(d - np.median(d)).max() < 1.5 and abs(np.median(d)) < 4


def test_cam_id_renaming(fab):
    sess = load_session(fab.dir, cam_ids="a,b,c")
    assert sess.cam_ids == ["a", "b", "c"]
    sess = load_session(fab.dir, cam_ids="10.0.0.12=right")
    assert sess.cam_ids == ["right"]
    assert parse_cam_ids("x=y, z=w") == {"x": "y", "z": "w"}
    with pytest.raises(ValueError):
        load_session(fab.dir, cam_ids="a,b")


def test_old_schema1_session_still_loads(tmp_path):
    """A session as the pre-schema-2 writer left it: no clock file, no new keys."""
    aus = encode_h264(20, gop=10)
    sess = RecordingSession(str(tmp_path), name="old")
    base = 1.7e12
    for k, au in enumerate(aus):
        idr = h264.is_idr(au)
        meta = int(base + k * 33.3 + 500) if idr else None
        sess.write_frame("Pixel 5", au, recv_ms=base + k * 33.3 + 20 + (k % 3),
                         is_keyframe=idr, meta_ms=meta,
                         server_ms=(meta - 500.0) if meta else None)
    sess.close(clock_offsets={"Pixel 5": 499.0})
    mpath = os.path.join(sess.dir, "manifest.json")
    man = json.load(open(mpath))
    old_keys = {"phone_id", "slug", "h264", "index", "frames", "bytes", "first_recv_ms",
                "last_recv_ms", "clock_offset_ms"}
    man = {"name": man["name"], "created_ms": man["created_ms"],
           "phones": [{k: v for k, v in p.items() if k in old_keys} for p in man["phones"]]}
    json.dump(man, open(mpath, "w"))
    s = load_session(sess.dir)
    assert s.schema_version == 1
    c = s.cameras[0]
    assert c.cam_id == "Pixel_5" and (c.width, c.height) == (160, 96)
    assert c.offset.method == "keyframe_server_ms" and c.offset.a == pytest.approx(500.0)
    assert c.times.method == "meta_anchors"
    assert np.abs(c.t_ms - (base + np.arange(20) * 33.3)).max() < 1.0
    assert c.codec_config and c.fps_measured == pytest.approx(30.03, rel=0.01)
    # and without any META the arrival envelope is used
    idx = os.path.join(sess.dir, "Pixel_5.index.jsonl")
    rows = [json.loads(l) for l in open(idx)]
    with open(idx, "w") as f:
        for r in rows:
            r["meta_ms"] = r["server_ms"] = None
            f.write(json.dumps(r) + "\n")
    c = load_session(sess.dir).cameras[0]
    assert c.times.method == "recv_envelope" and c.offset.method == "manifest"


def test_recorder_samples_clocks_and_caches_config(tmp_path):
    from ..recorder import RecorderController
    from ..server import CaptureServer
    from .session_fixtures import FakeConn
    aus = encode_h264(12, gop=10)
    srv = CaptureServer(host="127.0.0.1", port=0)
    conn = FakeConn("cam", 77.0)
    srv.connections = lambda: [conn]
    rec = RecorderController(srv, str(tmp_path), clock_interval=0.05)
    rec._on_frame(conn, aus[0], 1.0, True, 100, 23.0)       # before arming: config cached
    d = rec.start(info={"nominal_fps": 30})
    import time
    time.sleep(0.2)
    for k in range(1, 12):
        rec._on_frame(conn, aus[k], 1.0 + k, h264.is_idr(aus[k]), None, None)
    assert rec.stop() == d and rec.last_session_dir == d
    man = json.load(open(os.path.join(d, "manifest.json")))
    p = man["phones"][0]
    assert p["clock_samples"] >= 2 and p["clock_summary"]["offset_last_ms"] == 77.0
    assert p["codec_config_b64"] and (p["width"], p["height"]) == (160, 96)
    rows = [json.loads(l) for l in open(os.path.join(d, p["clock"]))]
    assert rows[0]["raw_offset_ms"] == 77.0


def test_session_dir_never_reused(tmp_path):
    a = RecordingSession(str(tmp_path), name="same")
    a.write_frame("cam", b"\x00\x00\x01\x65x", recv_ms=1.0, is_keyframe=True)
    a.close()
    b = RecordingSession(str(tmp_path), name="same")
    assert b.dir != a.dir and b.name == "same_2"
    b.close()
