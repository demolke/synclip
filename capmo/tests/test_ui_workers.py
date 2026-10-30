"""LatestWorker: one background thread, newest job per key wins, results come
back through a queued signal on the GUI thread."""

from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Slot  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ..ui.workers import LatestWorker  # noqa: E402


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


class _Sink(QObject):
    def __init__(self):
        super().__init__()
        self.got = []
        self.thread = None

    @Slot(object, object)
    def on_done(self, key, res):
        self.thread = threading.get_ident()
        self.got.append((key, res))


def test_stale_jobs_are_dropped_and_results_arrive_on_the_gui_thread(app):
    gate, started, ran = threading.Event(), threading.Event(), []

    def fn(key, job):
        ran.append((key, job, threading.get_ident()))
        started.set()
        gate.wait(5)
        return job * 10

    w = LatestWorker(fn, "test-worker")
    sink = _Sink()
    w.done.connect(sink.on_done)
    try:
        w.submit("a", 1)
        assert started.wait(5)
        w.submit("a", 2)                     # superseded before it ran ...
        w.submit("a", 3)                     # ... by this one
        w.submit("b", 7)
        gate.set()
        assert w.wait_idle(5)
        app.processEvents()
        assert [(k, j) for k, j, _ in ran] == [("a", 1), ("a", 3), ("b", 7)]
        assert {t for *_, t in ran} == {w.thread_ident} != {threading.get_ident()}
        assert sorted(sink.got) == [("a", 10), ("a", 30), ("b", 70)]
        assert sink.thread == threading.get_ident()
    finally:
        w.stop()


def test_failing_job_does_not_kill_the_worker_and_stop_joins(app):
    def fn(key, job):
        if job == "bad":
            raise RuntimeError("boom")
        return job

    w = LatestWorker(fn, "test-worker")
    sink = _Sink()
    w.done.connect(sink.on_done)
    w.submit("k", "bad")
    assert w.wait_idle(5)
    w.submit("k", "ok")
    assert w.wait_idle(5)
    app.processEvents()
    assert sink.got == [("k", "ok")]
    t = w._thread
    w.stop()
    assert not t.is_alive()
    w.submit("k", "late")                    # ignored after stop
    assert not t.is_alive()
