"""Container entry point: keep exactly one API process alive, restart it after a crash.

This exists for the force-kill scenario.  Docker's own restart policy only
helps when the *container* exits; a ``kill -9`` on the API process inside an
otherwise healthy container would leave the service down.  As PID 1 this
supervisor:

* runs ``uvicorn app.main:app`` as its child and restarts it whenever it
  exits without a stop request (SIGKILL included) -- the new API instance
  then performs startup recovery against the same data directory, in the
  same PID namespace where its orphaned model workers still live;
* forwards SIGTERM/SIGINT to the child so ``docker compose stop`` stays a
  graceful shutdown (bounded, then escalates to SIGKILL);
* reaps zombies: orphaned model workers and registry managers reparent to
  PID 1, and PID 1 must reap them for their exit to be observable;
* publishes the child's pid to ``$DATA_DIR/api.pid`` so an operator (or the
  demo script) can target the API process specifically.

The supervisor never runs more than one API child at a time, and the API
itself refuses to start twice against the same data directory (file lock),
so the two-process budget keeps a single owner.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

RESTART_DELAY = float(os.environ.get("SUPERVISOR_RESTART_DELAY_SECONDS", "1.0"))
STOP_GRACE = float(os.environ.get("SUPERVISOR_STOP_GRACE_SECONDS", "25"))
DATA_DIR = os.environ.get("DATA_DIR", "/data")
PID_FILE = os.path.join(DATA_DIR, "api.pid")

_reaped: dict[int, int] = {}
_child: "subprocess.Popen[bytes] | None" = None
_stopping = False


def _log(message: str) -> None:
    print(f"supervisor: {message}", flush=True)


def _sweep() -> None:
    """Reap any exited children: our API child and adopted orphans alike."""
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except (ChildProcessError, OSError):
            return
        if pid <= 0:
            return
        _reaped[pid] = status


def _decode(status: int) -> int:
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    return 0


def _child_returncode(child: "subprocess.Popen[bytes]") -> int:
    """Accurate exit status even if a zombie sweep reaped the child first."""
    status = _reaped.pop(child.pid, None)
    if status is not None:
        return _decode(status)
    rc = child.poll()
    return 0 if rc is None else rc


def _on_signal(signum: int, _frame: object) -> None:
    global _stopping
    _stopping = True
    child = _child
    if child is not None and child.poll() is None:
        try:
            os.kill(child.pid, signal.SIGTERM)
        except OSError:
            pass


def _write_pid_file(pid: int) -> None:
    try:
        tmp = f"{PID_FILE}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(str(pid))
        os.replace(tmp, PID_FILE)
    except OSError as exc:
        _log(f"could not write {PID_FILE}: {exc}")


def main() -> int:
    global _child
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        os.environ.get("PORT", "8000"),
        "--log-level",
        os.environ.get("UVICORN_LOG_LEVEL", "info"),
    ]
    while True:
        child = subprocess.Popen(cmd)
        _child = child
        _write_pid_file(child.pid)
        _log(f"started API process pid={child.pid}")
        stop_deadline: float | None = None
        while True:
            _sweep()
            if child.poll() is not None:
                break
            if _stopping:
                if stop_deadline is None:
                    stop_deadline = time.monotonic() + STOP_GRACE
                elif time.monotonic() >= stop_deadline:
                    _log("graceful stop timed out; sending SIGKILL to the API")
                    try:
                        child.kill()
                    except OSError:
                        pass
                    stop_deadline = None
            time.sleep(0.1)
        rc = _child_returncode(child)
        _child = None
        try:
            os.unlink(PID_FILE)
        except OSError:
            pass
        if _stopping:
            _log(f"API process pid={child.pid} exited rc={rc}; supervisor stopping")
            return 0
        _log(
            f"API process pid={child.pid} exited rc={rc}; "
            f"restarting in {RESTART_DELAY:.1f}s (startup recovery will run)"
        )
        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    raise SystemExit(main())
