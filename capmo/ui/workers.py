"""
A tiny "latest job wins" background worker for the capmo UI.

The GUI thread :meth:`~LatestWorker.submit`\\ s jobs by key (e.g. one key per
phone); the worker thread runs *fn(key, job)* on the newest pending job of each
key -- jobs that were superseded before the worker got to them are dropped, so
nothing ever queues up behind a slow job -- and returns each result through the
queued :attr:`~LatestWorker.done` signal, so the connected slot runs on the GUI
thread. Used by the board check (ChArUco detection) and the character view
(software rendering).
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Signal

log = logging.getLogger("capmo.ui")


class LatestWorker(QObject):
    done = Signal(object, object)           # (key, result) -- emitted off the GUI thread

    def __init__(self, fn, name: str = "capmo-worker", parent=None) -> None:
        super().__init__(parent)
        self._fn = fn
        self._name = name
        self._cond = threading.Condition()
        self._pending: dict = {}
        self._busy = False
        self._stop = False
        self._thread: threading.Thread | None = None

    @property
    def thread_ident(self) -> int | None:
        return self._thread.ident if self._thread is not None else None

    def submit(self, key, job) -> None:
        """Queue *job* for *key*, replacing a not-yet-started job of that key."""
        with self._cond:
            if self._stop:
                return
            self._pending[key] = job
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
                self._thread.start()
            self._cond.notify_all()

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Block until nothing is pending or running (tests / shutdown)."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._pending and not self._busy, timeout)

    def stop(self, timeout: float = 5.0) -> None:
        """Drop pending jobs and join the thread (waits for the running job)."""
        with self._cond:
            self._stop = True
            self._pending.clear()
            self._cond.notify_all()
            t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)

    def _loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._pending or self._stop)
                if self._stop:
                    return
                key = next(iter(self._pending))
                job = self._pending.pop(key)
                self._busy = True
            try:
                res = self._fn(key, job)
            except Exception as exc:          # a bad job must not kill the worker
                log.debug("%s: job %r failed: %s", self._name, key, exc)
                res = None
            try:
                if res is not None and not self._stop:
                    self.done.emit(key, res)
            except RuntimeError:              # owner already deleted
                pass
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()
