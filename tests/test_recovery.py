"""Recovery tests: a SIGKILLed API must not leave model processes behind.

Covers the on-call scenario end to end with real OS processes:

* API killed while a model is serving -> the worker survives as an orphan ->
  the next API instance on the same data directory reclaims it *before*
  serving, and models can be re-uploaded afterwards.
* API killed while a candidate is still loading -> both orphans reclaimed.
* API killed *during* recovery itself -> the next start finishes the job.
* The orphaned registry-manager process (a fork child) is reclaimed too.
* PID reuse never causes a wrong kill: manifests whose pid now belongs to an
  unrelated process are only cleaned up, never signalled.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest

from app import recovery

ROOT = Path(__file__).resolve().parent.parent
HAVE_PROC = Path("/proc/self/stat").exists()

pytestmark = pytest.mark.skipif(not HAVE_PROC, reason="recovery requires /proc")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _gone_or_zombie(pid: int) -> bool:
    proc = recovery.read_proc(pid)
    return proc is None or proc.state == "Z"


def _alive(pid: int) -> bool:
    proc = recovery.read_proc(pid)
    return proc is not None and proc.state != "Z"


def wait_for(predicate, timeout=15.0, interval=0.05, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"{what} not met in time")


class ApiProc:
    """One API process on a given data directory; can be SIGKILLed and replaced."""

    def __init__(self, data_dir: Path, env_extra: dict | None = None) -> None:
        self.data_dir = data_dir
        self.env_extra = env_extra or {}
        self.port = _free_port()
        self.proc: subprocess.Popen | None = None
        self.log_path = data_dir / f"api-{self.port}.log"
        self._log = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, wait: bool = True) -> "ApiProc":
        self._log = open(self.log_path, "ab")
        env = {
            **os.environ,
            "DATA_DIR": str(self.data_dir),
            "LOG_LEVEL": "WARNING",
            "PYTHONUNBUFFERED": "1",
            **self.env_extra,
        }
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if wait:
            self.wait_healthy()
        return self

    def wait_healthy(self, timeout: float = 40.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"API exited during startup; log:\n{self.log_text()}")
            try:
                if httpx.get(f"{self.base}/healthz", timeout=1.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"API did not start; log:\n{self.log_text()}")

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def sigkill(self) -> None:
        assert self.proc is not None
        os.kill(self.proc.pid, signal.SIGKILL)
        self.proc.wait(10)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(10)
        if self._log is not None:
            self._log.close()
            self._log = None


@pytest.fixture()
def api_stack(tmp_path):
    """API-process factory on one shared data dir; everything stopped at the end.

    ``api_stack(**env)`` starts a healthy API; ``api_stack.spawn(**env)``
    returns an unstarted ApiProc for tests that need to watch startup itself.
    """
    data_dir = tmp_path / "qcdata"
    data_dir.mkdir()
    servers: list[ApiProc] = []

    class Stack:
        def __call__(self, **env_extra) -> ApiProc:
            server = ApiProc(data_dir, env_extra)
            servers.append(server)
            return server.start()

        @staticmethod
        def spawn(**env_extra) -> ApiProc:
            server = ApiProc(data_dir, env_extra)
            servers.append(server)
            return server

    try:
        yield Stack()
    finally:
        for server in servers:
            server.stop()
        # Backstop: no worker of this data dir may outlive the test.
        leftovers = recovery.scan_for_workers(str(data_dir / "workers"))
        for proc in leftovers:
            try:
                os.killpg(proc.pgrp, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if leftovers:
            raise AssertionError(f"worker processes leaked the test: {leftovers}")


def upload(api: ApiProc, payload, timeout=40.0):
    return httpx.post(f"{api.base}/admin/models", json=payload, timeout=timeout)


def infer(api: ApiProc, features, **extra):
    return httpx.post(
        f"{api.base}/infer", json={"features": features, **extra}, timeout=40.0
    )


def status(api: ApiProc):
    return httpx.get(f"{api.base}/admin/status", timeout=5.0).json()


W_A = [[1.0, 0.0], [0.0, 1.0]]
W_B = [[0.0, 1.0], [1.0, 0.0]]
LABELS = ["defect-A", "defect-B"]


def artifact(version, weights, **extra):
    return {"version": version, "weights": weights, "bias": [0.0, 0.0], "labels": LABELS, **extra}


def active_worker_pid(api: ApiProc) -> int:
    st = status(api)
    return next(s["pid"] for s in st["slots"] if s["process_state"] == "active")


# ---------------------------------------------------------------------------
# End-to-end SIGKILL scenarios
# ---------------------------------------------------------------------------


def test_sigkill_while_serving_is_reclaimed(api_stack):
    api = api_stack()
    assert upload(api, artifact("v1", W_A)).status_code == 201
    assert infer(api, [0.9, 0.1]).json()["label"] == "defect-A"
    worker_pid = active_worker_pid(api)

    api.sigkill()
    # The worker really does survive the API's SIGKILL (the bug scenario).
    wait_for(lambda: _alive(worker_pid), what="orphaned worker still running")

    api2 = api_stack()
    # healthz is up => startup recovery has already finished (the API binds
    # its port only after recovery), so the orphan must be gone.
    wait_for(lambda: _gone_or_zombie(worker_pid), what="orphaned worker reclaimed")

    st = status(api2)
    assert st["slots_in_use"] == 0
    assert st["active_version"] is None
    assert st["total_accepted"] == 0  # old requests are not replayed
    reclaimed_pids = [w["pid"] for w in st["recovery"]["reclaimed_workers"]]
    assert worker_pid in reclaimed_pids

    # The same version id can be re-uploaded after recovery (registry is fresh).
    resp = upload(api2, artifact("v1", W_A))
    assert resp.status_code == 201, resp.text
    assert infer(api2, [0.9, 0.1]).json()["version"] == "v1"
    # And a normal online swap still works afterwards.
    assert upload(api2, artifact("v2", W_B)).status_code == 201
    assert infer(api2, [0.9, 0.1]).json()["version"] == "v2"


def test_sigkill_during_candidate_load_is_reclaimed(api_stack):
    api = api_stack()
    assert upload(api, artifact("v1", W_A)).status_code == 201
    serving_pid = active_worker_pid(api)

    outcome = {}

    def slow_upload():
        try:
            outcome["resp"] = upload(api, artifact("v2", W_B, slow_load_ms=8000))
        except Exception as exc:  # the API dies mid-upload; the client just breaks
            outcome["error"] = exc

    thread = threading.Thread(target=slow_upload)
    thread.start()
    # Wait until the loading candidate occupies the second slot.
    wait_for(lambda: status(api)["slots_in_use"] == 2, what="candidate loading")
    loading_pid = next(
        s["pid"] for s in status(api)["slots"] if s["process_state"] == "loading"
    )

    api.sigkill()
    thread.join(10)
    wait_for(lambda: _alive(serving_pid) and _alive(loading_pid), what="both orphaned")

    api2 = api_stack()
    wait_for(lambda: _gone_or_zombie(serving_pid), what="serving worker reclaimed")
    wait_for(lambda: _gone_or_zombie(loading_pid), what="loading worker reclaimed")

    st = status(api2)
    assert st["slots_in_use"] == 0
    reclaimed = {w["pid"] for w in st["recovery"]["reclaimed_workers"]}
    assert {serving_pid, loading_pid} <= reclaimed

    assert upload(api2, artifact("v3", W_A)).status_code == 201
    assert infer(api2, [0.9, 0.1]).json()["version"] == "v3"


def test_orphaned_registry_manager_is_reclaimed(api_stack, tmp_path):
    api = api_stack()
    assert upload(api, artifact("v1", W_A)).status_code == 201
    records = list((api.data_dir / "instances").glob("*.json"))
    assert len(records) == 1
    manager_pid = json.loads(records[0].read_text())["manager_pid"]
    assert _alive(manager_pid)

    api.sigkill()
    # The manager (a fork child) also survives the API's SIGKILL...
    wait_for(lambda: _alive(manager_pid), what="orphaned manager still running")

    api2 = api_stack()
    # ...and the next instance reclaims it during startup recovery.
    wait_for(lambda: _gone_or_zombie(manager_pid), what="orphaned manager reclaimed")
    st = status(api2)
    assert manager_pid in [m["pid"] for m in st["recovery"]["reclaimed_managers"]]
    # Only the new instance's own record remains.
    remaining = list((api.data_dir / "instances").glob("*.json"))
    assert len(remaining) == 1
    assert json.loads(remaining[0].read_text())["instance"] == st["instance"]


def test_sigkill_during_recovery_still_recovers(api_stack, tmp_path):
    data_dir = tmp_path / "qcdata"
    # Seed a leftover worker that ignores SIGTERM, so recovery takes a while
    # and can be killed in the middle of its grace wait.
    work_dir = data_dir / "workers"
    fake, _control_dir = spawn_fake_worker(work_dir, ignore_sigterm=True)
    try:
        env = {"RECOVERY_TERM_GRACE_SECONDS": "5", "RECOVERY_KILL_GRACE_SECONDS": "5"}
        api = api_stack.spawn(**env)
        api.start(wait=False)  # do not block: recovery runs before healthz
        # Wait for recovery to start terminating the fake worker, then kill
        # the API in the middle of the SIGTERM grace window.
        wait_for(
            lambda: "terminating leftover model worker" in api.log_text(),
            what="recovery to start",
        )
        api.sigkill()
        assert _alive(fake.pid)  # SIGTERM was ignored; SIGKILL never came

        api2 = api_stack(**env)
        wait_for(lambda: _gone_or_zombie(fake.pid), what="fake worker reclaimed")
        st = status(api2)
        assert fake.pid in [w["pid"] for w in st["recovery"]["reclaimed_workers"]]
        assert upload(api2, artifact("v1", W_A)).status_code == 201
        assert infer(api2, [0.1, 0.9]).json()["label"] == "defect-B"
    finally:
        if _alive(fake.pid):
            os.killpg(os.getpgid(fake.pid), signal.SIGKILL)
        fake.wait()


def test_second_instance_on_same_data_dir_is_refused(api_stack):
    api = api_stack()
    assert upload(api, artifact("v1", W_A)).status_code == 201
    challenger = ApiProc(api.data_dir)
    env = {**os.environ, "DATA_DIR": str(api.data_dir), "LOG_LEVEL": "WARNING"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
         "--port", str(_free_port()), "--log-level", "warning"],
        cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    rc = proc.wait(30)
    assert rc != 0  # refused: the data directory already has an owner
    assert infer(api, [0.9, 0.1]).status_code == 200  # first instance undisturbed


# ---------------------------------------------------------------------------
# Unit-level recovery behaviour
# ---------------------------------------------------------------------------

FAKE_WORKER_IGNORE_TERM = (
    "import signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "time.sleep(300)\n"
)
FAKE_WORKER_PLAIN = "import time; time.sleep(300)\n"


def spawn_fake_worker(work_dir: Path, ignore_sigterm: bool = False):
    """A stand-in model worker: right cmdline shape, own process group, manifest."""
    control_dir = work_dir / f"slot-9-{uuid.uuid4().hex[:8]}"
    control_dir.mkdir(parents=True, exist_ok=True)
    code = FAKE_WORKER_IGNORE_TERM if ignore_sigterm else FAKE_WORKER_PLAIN
    proc = subprocess.Popen(
        [sys.executable, "-c", code, "app.worker_launcher", str(control_dir)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    info = wait_for(lambda: recovery.read_proc(proc.pid), what="fake worker in /proc")
    manifest = {
        "pid": proc.pid,
        "pgid": os.getpgid(proc.pid),
        "started_ticks": info.starttime,
        "control_dir": str(control_dir),
        "slot_id": "slot-9",
        "version": "fake",
        "instance": "previous-instance",
        "created_at": time.time(),
    }
    (control_dir / recovery.MANIFEST_NAME).write_text(json.dumps(manifest))
    return proc, control_dir


def test_recover_kills_real_worker_and_is_idempotent(tmp_path):
    work_dir = tmp_path / "workers"
    proc, control_dir = spawn_fake_worker(work_dir)
    try:
        report = recovery.recover_leftover_workers(str(work_dir), term_grace=2, kill_grace=2)
        assert [entry["pid"] for entry in report] == [proc.pid]
        assert report[0]["version"] == "fake"
        wait_for(lambda: _gone_or_zombie(proc.pid), what="fake worker terminated")
        assert not control_dir.exists()  # control dir removed once process is gone
        # A second run (e.g. after the API itself died mid-recovery) is a no-op.
        assert recovery.recover_leftover_workers(str(work_dir)) == []
    finally:
        if _alive(proc.pid):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def test_recover_without_manifest_uses_proc_scan(tmp_path):
    work_dir = tmp_path / "workers"
    proc, control_dir = spawn_fake_worker(work_dir)
    (control_dir / recovery.MANIFEST_NAME).unlink()  # lost manifest
    try:
        report = recovery.recover_leftover_workers(str(work_dir), respawn_grace=0.1)
        assert [entry["pid"] for entry in report] == [proc.pid]
        assert report[0]["via"] == "proc-scan"
        wait_for(lambda: _gone_or_zombie(proc.pid), what="fake worker terminated")
    finally:
        if _alive(proc.pid):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def _spawn_unrelated():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            start_new_session=True)
    wait_for(lambda: recovery.read_proc(proc.pid), what="unrelated process in /proc")
    return proc


def test_pid_reuse_never_kills_unrelated_process(tmp_path):
    """A manifest whose pid now belongs to someone else is only cleaned up."""
    work_dir = tmp_path / "workers"
    unrelated = _spawn_unrelated()
    try:
        real = recovery.read_proc(unrelated.pid)
        for started_ticks in (1, real.starttime):  # wrong ticks / right ticks
            control_dir = work_dir / f"slot-0-{uuid.uuid4().hex[:8]}"
            control_dir.mkdir(parents=True)
            (control_dir / recovery.MANIFEST_NAME).write_text(json.dumps({
                "pid": unrelated.pid,
                "pgid": os.getpgid(unrelated.pid),
                "started_ticks": started_ticks,
                "control_dir": str(control_dir),
                "slot_id": "slot-0",
                "version": "ghost",
                "instance": "previous-instance",
            }))
            report = recovery.recover_leftover_workers(str(work_dir), respawn_grace=0.1)
            assert report == []
            assert _alive(unrelated.pid)  # never signalled
            assert not control_dir.exists()  # stale manifest cleaned up
    finally:
        os.killpg(os.getpgid(unrelated.pid), signal.SIGKILL)
        unrelated.wait()


def test_dead_pid_manifest_is_just_cleaned_up(tmp_path):
    work_dir = tmp_path / "workers"
    control_dir = work_dir / "slot-0-deadbeef"
    control_dir.mkdir(parents=True)
    (control_dir / recovery.MANIFEST_NAME).write_text(json.dumps({
        "pid": 999999, "pgid": 999999, "started_ticks": 1,
        "control_dir": str(control_dir), "slot_id": "slot-0", "version": "ghost",
    }))
    assert recovery.recover_leftover_workers(str(work_dir)) == []
    assert not control_dir.exists()


def test_zombie_counts_as_gone(tmp_path):
    work_dir = tmp_path / "workers"
    pid = os.fork()
    if pid == 0:  # child: exit immediately, become a zombie
        os._exit(0)
    try:
        wait_for(
            lambda: (recovery.read_proc(pid) or recovery.ProcInfo(pid, 0, "", None, [])).state
            == "Z",
            what="zombie child",
        )
        control_dir = work_dir / "slot-0-zombie"
        control_dir.mkdir(parents=True)
        info = recovery.read_proc(pid)
        (control_dir / recovery.MANIFEST_NAME).write_text(json.dumps({
            "pid": pid, "pgid": os.getpgid(pid),
            "started_ticks": info.starttime if info else None,
            "control_dir": str(control_dir), "slot_id": "slot-0", "version": "ghost",
        }))
        # Must not hang waiting for the zombie to disappear.
        assert recovery.recover_leftover_workers(str(work_dir), term_grace=1) == []
        assert not control_dir.exists()
    finally:
        os.waitpid(pid, 0)


def test_recover_instances_kills_recorded_manager_only(tmp_path):
    instances = tmp_path / "instances"
    instances.mkdir()
    # A live stand-in manager process.
    manager = _spawn_unrelated()
    try:
        info = recovery.read_proc(manager.pid)
        (instances / "old.json").write_text(json.dumps({
            "instance": "old-instance",
            "api_pid": 999998,
            "manager_pid": manager.pid,
            "manager_started_ticks": info.starttime,
        }))
        (instances / "self.json").write_text(json.dumps({
            "instance": "me", "api_pid": os.getpid(), "manager_pid": 999997,
            "manager_started_ticks": None,
        }))
        report = recovery.recover_instances(str(instances), self_instance="me")
        assert [entry["pid"] for entry in report] == [manager.pid]
        wait_for(lambda: _gone_or_zombie(manager.pid), what="manager terminated")
        # Own record survives; the old one is gone.
        assert [p.name for p in instances.glob("*.json")] == ["self.json"]
        # Idempotent: nothing left to do.
        assert recovery.recover_instances(str(instances), self_instance="me") == []
    finally:
        if _alive(manager.pid):
            os.killpg(os.getpgid(manager.pid), signal.SIGKILL)
        manager.wait()


def test_recover_instances_ignores_pid_reuse(tmp_path):
    instances = tmp_path / "instances"
    instances.mkdir()
    unrelated = _spawn_unrelated()
    try:
        (instances / "old.json").write_text(json.dumps({
            "instance": "old-instance",
            "api_pid": 999998,
            "manager_pid": unrelated.pid,
            "manager_started_ticks": 1,  # wrong start time -> not our manager
        }))
        report = recovery.recover_instances(str(instances), self_instance="me")
        assert report == []
        assert _alive(unrelated.pid)
        assert not (instances / "old.json").exists()  # stale record cleaned up
    finally:
        os.killpg(os.getpgid(unrelated.pid), signal.SIGKILL)
        unrelated.wait()
