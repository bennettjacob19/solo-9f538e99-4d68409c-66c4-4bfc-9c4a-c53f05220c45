"""Reclaim processes left behind when the API dies abruptly (SIGKILL, OOM, ...).

The API spawns every model worker with ``start_new_session=True`` so a
graceful shutdown can signal the worker's whole process group.  The flip side
is that a hard-killed API leaves its workers -- and its registry-manager
process -- running as orphans.  The workers keep their loopback ports, their
model memory and, on real hardware, the accelerator they were loaded onto.
Starting the API again against the same data directory must therefore
*reclaim* those orphans before it serves traffic or loads new models, or the
machine accumulates unmanaged model processes and the two-process budget is
fiction.

How a leftover is identified (and why PID reuse cannot cause a wrong kill):

* Every worker writes ``worker.json`` into its private control directory as
  the very first thing it does, before loading the artifact.  The manifest
  records the worker's pid, process-group id and the kernel start-time token
  from ``/proc/<pid>/stat`` (field 22).  A recycled pid always has a
  different start time, so a manifest can never condemn an unrelated process
  that merely inherited a dead worker's pid.  The command line must also
  contain ``app.worker_launcher`` and the worker's own control directory.
* The API writes one record per instance into ``instances/<id>.json`` with
  its registry-manager's pid and start-time token, so the next instance can
  stop the orphaned manager (a ``fork`` child that would otherwise leak --
  and, in the worst case, keep inherited file descriptors such as the
  data-directory lock alive forever).
* As a backstop for a worker that died before writing its manifest (or a
  corrupt manifest), ``/proc`` is scanned for ``app.worker_launcher`` command
  lines whose control-directory argument lives inside *this* data directory.

Recovery is idempotent: a record or control directory is removed only after
its process is confirmed gone, so an API that is itself killed mid-recovery
leaves enough on disk for the next start to finish the job.  A process in
zombie state counts as gone: it holds no device, and init will reap it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_NAME = "worker.json"
WORKER_MODULE = "app.worker_launcher"

module_logger = logging.getLogger("qc.recovery")


@dataclass
class ProcInfo:
    pid: int
    pgrp: int
    state: str
    starttime: Optional[int]  # kernel ticks since boot; None when unreadable
    cmdline: List[str]


def read_proc(pid: int) -> Optional[ProcInfo]:
    """Live info about ``pid`` from /proc, or None if the process is gone."""
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="replace") as fh:
            stat = fh.read()
    except OSError:
        return None
    try:
        # comm (field 2) is parenthesised and may contain spaces or parens;
        # everything after the last ')' is plain space-separated fields.
        fields = stat[stat.rindex(")") + 2 :].split()
        state = fields[0]  # field 3
        pgrp = int(fields[2])  # field 5
        starttime = int(fields[19])  # field 22
    except (ValueError, IndexError):
        return None
    cmdline: List[str] = []
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
        cmdline = [t.decode("utf-8", "replace") for t in raw.split(b"\0") if t]
    except OSError:
        pass
    return ProcInfo(pid=pid, pgrp=pgrp, state=state, starttime=starttime, cmdline=cmdline)


def self_start_ticks() -> Optional[int]:
    """This process's kernel start-time token, or None off Linux."""
    try:
        with open("/proc/self/stat", "r", encoding="utf-8", errors="replace") as fh:
            stat = fh.read()
        return int(stat[stat.rindex(")") + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _gone(pid: int) -> bool:
    proc = read_proc(pid)
    return proc is None or proc.state == "Z"


def _wait_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _gone(pid):
            return True
        time.sleep(0.05)
    return _gone(pid)


def _terminate(proc: ProcInfo, term_grace: float, kill_grace: float, group: bool) -> bool:
    """SIGTERM then SIGKILL; wait for the process to actually disappear."""
    # Paranoia: never signal ourselves, init, or our own process group,
    # whatever a (possibly corrupt) record says.
    if proc.pid in (0, 1, os.getpid()) or proc.pgrp in (0, 1):
        return False
    if group and proc.pgrp == os.getpgrp():
        return False
    for sig, grace in ((signal.SIGTERM, term_grace), (signal.SIGKILL, kill_grace)):
        try:
            if group:
                os.killpg(proc.pgrp, sig)
            else:
                os.kill(proc.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            return False
        if _wait_gone(proc.pid, grace):
            return True
    return _gone(proc.pid)


def _cmdline_is_worker(cmdline: List[str], control_dir: str) -> bool:
    return WORKER_MODULE in cmdline and bool(control_dir) and control_dir in cmdline


def _manifest_matches(proc: ProcInfo, manifest: Dict[str, Any]) -> bool:
    """True only if the live process really is the worker the manifest describes."""
    if proc.state == "Z":
        return False
    started = manifest.get("started_ticks")
    if started is not None and proc.starttime is not None:
        if int(started) != proc.starttime:
            return False  # same pid, different process: the pid was reused
    pgid = manifest.get("pgid")
    if pgid is not None and proc.pgrp != int(pgid):
        return False
    control_dir = str(manifest.get("control_dir") or "")
    return _cmdline_is_worker(proc.cmdline, control_dir)


def _worker_args(cmdline: List[str]) -> Dict[str, str]:
    """Best-effort slot/version extracted from a worker command line."""
    try:
        i = cmdline.index(WORKER_MODULE)
        return {"slot_id": cmdline[i + 1], "version": cmdline[i + 2]}
    except (ValueError, IndexError):
        return {}


def _path_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def scan_for_workers(work_dir: str) -> List[ProcInfo]:
    """Live ``app.worker_launcher`` processes belonging to this data directory.

    A process qualifies when its command line carries a control-directory
    argument inside ``work_dir`` -- a path only this service hands out.  This
    is the backstop for workers whose manifest is missing or corrupt.
    """
    root = os.path.realpath(work_dir)
    found: List[ProcInfo] = []
    if not os.path.isdir("/proc"):
        return found
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        proc = read_proc(int(entry))
        if proc is None or proc.state == "Z" or WORKER_MODULE not in proc.cmdline:
            continue
        if any(
            tok.startswith("/") and _path_within(os.path.realpath(tok), root)
            for tok in proc.cmdline
        ):
            found.append(proc)
    return found


def recover_leftover_workers(
    work_dir: str,
    *,
    term_grace: float = 3.0,
    kill_grace: float = 5.0,
    respawn_grace: float = 1.0,
    log: logging.Logger = module_logger,
) -> List[Dict[str, Any]]:
    """Stop model workers orphaned by a previous API instance.

    Returns one report entry per reclaimed process.  Raises ``RuntimeError``
    if a confirmed worker could not be stopped: the caller must not start
    serving while an old model process still holds the machine.
    """
    root = Path(work_dir)
    report: List[Dict[str, Any]] = []
    if not root.is_dir():
        return report

    def collect() -> Dict[Path, Optional[Dict[str, Any]]]:
        found: Dict[Path, Optional[Dict[str, Any]]] = {}
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            manifest: Optional[Dict[str, Any]] = None
            try:
                data = json.loads((child / MANIFEST_NAME).read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    manifest = data
            except (OSError, ValueError):
                manifest = None
            found[child] = manifest
        return found

    candidates = collect()
    if any(m is None for m in candidates.values()):
        # A worker orphaned between exec() and its first manifest write shows
        # up as a control dir without a manifest for a few hundred ms.  Give
        # such workers one grace window to publish, then look again.
        time.sleep(respawn_grace)
        candidates = collect()

    handled: set[int] = set()
    failures: List[str] = []

    for control_dir, manifest in candidates.items():
        victim: Optional[ProcInfo] = None
        if manifest is not None:
            pid = manifest.get("pid")
            if isinstance(pid, int) and pid > 0:
                live = read_proc(pid)
                if live is not None and live.state != "Z":
                    if _manifest_matches(live, manifest):
                        victim = live
                    else:
                        log.warning(
                            "recovery: %s names pid %d, but the live process does "
                            "not match it (pid reused?); leaving that process alone",
                            control_dir / MANIFEST_NAME,
                            pid,
                        )
        if victim is not None:
            log.warning(
                "recovery: terminating leftover model worker pid=%d version=%s slot=%s",
                victim.pid,
                manifest.get("version"),
                manifest.get("slot_id"),
            )
            if _terminate(victim, term_grace, kill_grace, group=True):
                handled.add(victim.pid)
                report.append(
                    {
                        "kind": "worker",
                        "via": "manifest",
                        "pid": victim.pid,
                        "pgrp": victim.pgrp,
                        "version": manifest.get("version"),
                        "slot_id": manifest.get("slot_id"),
                        "instance": manifest.get("instance"),
                        "control_dir": str(control_dir),
                    }
                )
            else:
                # Keep the control dir so the next start retries the reclaim.
                failures.append(f"worker pid {victim.pid} ({control_dir})")
                continue
        # The process (if any) is gone now; the directory is safe to remove.
        shutil.rmtree(control_dir, ignore_errors=True)

    # Backstop: workers whose manifest was never written or was unreadable.
    for proc in scan_for_workers(str(root)):
        if proc.pid in handled:
            continue
        log.warning(
            "recovery: terminating unmanifested model worker pid=%d cmdline=%s",
            proc.pid,
            proc.cmdline[:4],
        )
        if _terminate(proc, term_grace, kill_grace, group=True):
            handled.add(proc.pid)
            report.append(
                {
                    "kind": "worker",
                    "via": "proc-scan",
                    "pid": proc.pid,
                    "pgrp": proc.pgrp,
                    **_worker_args(proc.cmdline),
                }
            )
        else:
            failures.append(f"worker pid {proc.pid} (proc-scan)")

    if failures:
        raise RuntimeError(
            "could not stop leftover model process(es): " + ", ".join(failures)
        )
    return report


def recover_instances(
    instances_dir: str,
    *,
    self_instance: str,
    term_grace: float = 3.0,
    kill_grace: float = 5.0,
    log: logging.Logger = module_logger,
) -> List[Dict[str, Any]]:
    """Stop registry-manager processes of previous API instances.

    Every API instance writes ``<instance>.json`` here with its manager's pid
    and start-time token.  Records other than our own belong to dead
    instances (the data-dir lock guarantees the previous API itself is gone);
    their managers are terminated, verified by pid + start time, and their
    records removed.  Raises ``RuntimeError`` if a confirmed manager would
    not stop.
    """
    root = Path(instances_dir)
    report: List[Dict[str, Any]] = []
    if not root.is_dir():
        return report
    failures: List[str] = []
    for record_path in sorted(root.glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        if not isinstance(record, dict):
            log.warning("recovery: dropping unreadable instance record %s", record_path)
            record_path.unlink(missing_ok=True)
            continue
        if record.get("instance") == self_instance:
            continue
        mpid = record.get("manager_pid")
        if isinstance(mpid, int) and mpid > 1:
            proc = read_proc(mpid)
            if proc is not None and proc.state != "Z":
                ticks = record.get("manager_started_ticks")
                if ticks is not None and proc.starttime is not None and int(ticks) != proc.starttime:
                    log.warning(
                        "recovery: instance record %s names manager pid %d, but the "
                        "live process has a different start time (pid reused?); "
                        "leaving it alone",
                        record_path,
                        mpid,
                    )
                elif ticks is None:
                    log.warning(
                        "recovery: instance record %s has no start-time token for "
                        "pid %d; cannot verify identity, leaving it alone",
                        record_path,
                        mpid,
                    )
                else:
                    log.warning(
                        "recovery: terminating leftover registry manager pid=%d "
                        "of instance %s",
                        mpid,
                        record.get("instance"),
                    )
                    if _terminate(proc, term_grace, kill_grace, group=False):
                        report.append(
                            {
                                "kind": "manager",
                                "pid": mpid,
                                "instance": record.get("instance"),
                            }
                        )
                    else:
                        failures.append(f"manager pid {mpid}")
                        continue  # keep the record; retry on next start
        record_path.unlink(missing_ok=True)
    if failures:
        raise RuntimeError(
            "could not stop leftover registry manager(s): " + ", ".join(failures)
        )
    return report
