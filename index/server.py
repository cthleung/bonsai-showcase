"""Start and stop the local llama-server processes (one for embeddings, one for reranking).

Both bind to 127.0.0.1 only, so nothing outside this computer can reach them. They run with `--no-slots`
(no slot monitoring endpoint) and with any LLAMA_ARG_* variables removed from their environment,
so nothing set outside this package can change their flags.

    python -m index serve            # start both and leave them running
    python -m index serve --stop     # stop the ones this package started
    python -m index serve --status

**Bookkeeping.** BONSAI_STATE/index/servers.json has one entry per role this package started:
the pid with its start token (index/procs.py, so a reused pid is never mistaken for it), the
port, the model, `auto_stop`, and `users`, the runs using it right now. Every read-modify-write
of the file happens under a lock file (`servers.lock`), so two overlapping runs can't overwrite
each other's entries. The lock names its holder (pid and start token) and is never held while a
server loads, so a run killed at any point leaves a lock the next run sees is stale and removes.

- A live entry is never overwritten. A second run joins it as another user, also while it's still
  `starting` (then it waits on that start instead of making its own).
- A server that exits early, or isn't healthy within the wait, is killed and its entry removed,
  so no untracked server is left holding VRAM.
- A run leaves with `release()`. A server with `auto_stop` stops when its last user leaves, so
  one search never stops a server another run is still using. `serve` and `--keep-server` turn
  `auto_stop` off.
- A server already answering on the port that this package didn't start is used only if its
  /props names the expected model file, and it's never stopped.

Settings per server (measured on an RTX 4080 SUPER): a context of 4096 split over 2
slots of 2048 tokens each, and batch and ubatch of 2048. An embedding input has to fit in one slot,
so 2048 tokens is the longest input; the chunker aims for about 1,100 and the embed client shortens
anything that still doesn't fit. With these settings the 4B embedder takes about 5.2 GB of VRAM and
the 4B reranker about 5.2 GB more.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path

from common import pidlock
from index import procs
from index.localhttp import urlopen
from index.settings import Settings

COMMON = ["--host", "127.0.0.1", "-ngl", "999", "-c", "4096", "-np", "2", "-b", "2048", "-ub", "2048",
          "--no-webui", "--no-slots"]
ROLES = {
    "embed": ["--embedding", "--pooling", "last"],
    "rerank": ["--reranking", "--pooling", "rank"],
}
LOCK_WAIT = 120.0       # how long a run waits for the lock (held only for servers.json updates and stops)


class ServerError(RuntimeError):
    pass


def _url(s: Settings, role: str) -> str:
    return s.embed_url if role == "embed" else s.rerank_url


def _port(s: Settings, role: str) -> int:
    return s.embed_port if role == "embed" else s.rerank_port


def _model(s: Settings, role: str) -> Path | None:
    return s.embed_model if role == "embed" else s.rerank_model


def healthy(url: str, timeout: float = 2.0) -> bool:
    try:
        with urlopen(f"{url}/health", timeout=timeout) as r:
            return r.status == 200 and b"ok" in r.read()
    except (urllib.error.URLError, OSError, ValueError):
        return False


def serving_model(url: str, timeout: float = 2.0) -> str | None:
    """The file name of the model a llama-server is running, from its /props."""
    try:
        with urlopen(f"{url}/props", timeout=timeout) as r:
            path = json.load(r).get("model_path") or ""
    except (urllib.error.URLError, OSError, ValueError, AttributeError):
        return None
    return Path(str(path).replace("\\", "/")).name or None


def ready(s: Settings, role: str) -> bool:
    """Up, and running the model this role expects."""
    model = _model(s, role)
    return healthy(_url(s, role)) and model is not None and serving_model(_url(s, role)) == model.name


# --- servers.json under a lock --------------------------------------------------------------

class _Lock:
    """servers.json's lock, held only for a read-modify-write of servers.json (and a stop), never
    while a server loads. It's common/pidlock.py's lock, the one capture's run lock uses: the
    file names its holder (pid and start token), a dead holder's lock is taken over, and Windows'
    file-in-use errors on create and delete are retried. This waits up to LOCK_WAIT for a live
    holder, then raises ServerError.

    The file names a process, not a thread, so threads of one process (the gateway's search worker
    and its shutdown) first take a thread lock for that file: otherwise one thread's cleanup could
    delete the lock another thread of the same process had just taken."""

    _threads: dict[str, threading.Lock] = {}
    _threads_guard = threading.Lock()

    def __init__(self, s: Settings):
        self.path = s.index_dir / "servers.lock"
        s.index_dir.mkdir(parents=True, exist_ok=True)
        self._cm = None
        key = os.path.normcase(os.path.abspath(self.path))
        with _Lock._threads_guard:
            self._thread_lock = _Lock._threads.setdefault(key, threading.Lock())

    def __enter__(self):
        deadline = time.time() + LOCK_WAIT
        if not self._thread_lock.acquire(timeout=LOCK_WAIT):
            raise ServerError(f"another thread of this process holds {self.path}")
        try:
            while True:
                cm = pidlock.pid_lock(self.path)
                if cm.__enter__():
                    self._cm = cm
                    return self
                cm.__exit__(None, None, None)
                if time.time() > deadline:
                    raise ServerError(f"another run holds {self.path}")
                time.sleep(random.uniform(0.02, 0.08))
        except BaseException:
            self._thread_lock.release()
            raise

    def __exit__(self, *exc):
        try:
            self._leave()
        finally:
            self._thread_lock.release()

    def _leave(self):
        cm, self._cm = self._cm, None
        if cm is not None:
            cm.__exit__(None, None, None)
        # pid_lock leaves the file when its delete keeps failing, or when its last read hit a
        # waiter's open (Windows reads it as {}). A file still naming this process would hold every
        # other run off until this one takes the lock again, so make sure it's gone.
        pid, token = procs.me()
        mine = {"pid": pid, "token": token}
        for _ in range(100):
            info = pidlock.read(self.path)
            if info is None or (info and info != mine):      # gone, or someone else's now
                break
            if info == mine and pidlock._unlink(self.path):
                break
            time.sleep(0.02)


def _read(s: Settings) -> dict:
    try:
        data = json.loads(s.servers_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(s: Settings, data: dict) -> None:
    s.index_dir.mkdir(parents=True, exist_ok=True)
    tmp = s.servers_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, s.servers_file)


def _entry_alive(entry: dict | None) -> bool:
    return bool(entry) and procs.alive(int(entry.get("pid") or 0), entry.get("token"))


def _live_users(entry: dict) -> list:
    return [u for u in entry.get("users") or [] if procs.alive(int(u[0]), u[1])]


# --- starting and stopping ------------------------------------------------------------------

def command(s: Settings, role: str) -> list[str]:
    model = _model(s, role)
    exe = [str(s.llama_server)]
    if s.llama_server.suffix == ".py":       # the tests' stand-in server
        exe = [sys.executable, "-I", str(s.llama_server)]
    return [*exe, "-m", str(model), *ROLES[role], "--port", str(_port(s, role)), *COMMON]


def launch_env() -> dict[str, str]:
    """This process's environment minus LLAMA_ARG_* (llama-server reads flags from those)."""
    return {k: v for k, v in os.environ.items() if not k.upper().startswith("LLAMA_ARG_")}


def _wait_healthy(proc, url: str, wait: float) -> str:
    """'up', 'exited' or 'timeout'."""
    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            return "exited"
        if healthy(url):
            return "up"
        time.sleep(0.25)
    return "timeout"


def _wait_joined(pid: int, token: str | None, url: str, wait: float) -> str:
    """Like _wait_healthy, for a server another run is starting: 'up', 'exited' or 'timeout'."""
    deadline = time.time() + wait
    while time.time() < deadline:
        if not procs.alive(pid, token):
            return "exited"
        if healthy(url):
            return "up"
        time.sleep(0.25)
    return "timeout"


def _leave(s: Settings, role: str, user: list | None) -> None:
    """Take `user` off the role's entry (a run that waited on someone else's start and gave up)."""
    if not user:
        return
    with _Lock(s):
        data = _read(s)
        entry = data.get(role)
        if entry:
            entry["users"] = [u for u in _live_users(entry) if list(u) != list(user)]
            _write(s, data)


def start(s: Settings, role: str, wait: float = 120.0, log=print, user: list | None = None,
          keep: bool = True, tie: bool = False) -> bool:
    """Make sure the role's server is up, starting it if needed. True when it's up.
    See _start_once; when the start this run joined dies, it tries once more on its own."""
    for _ in range(2):
        up = _start_once(s, role, wait, log, user, keep, tie)
        if up is not None:
            return up
    return False


START_GRACE = 10.0   # how long past its own wait a start still counts as in progress


def _start_live(entry: dict) -> bool:
    """A `starting` entry is a live start while its starter runs and its wait (`until`) plus
    START_GRACE hasn't passed. Past that, whatever happened to the starter, it's a leftover."""
    starter = entry.get("starter") or [0, None]
    return (procs.alive(int(starter[0] or 0), starter[1])
            and time.time() < float(entry.get("until") or 0) + START_GRACE)


def _start_once(s: Settings, role: str, wait: float, log, user: list | None, keep: bool,
                tie: bool) -> bool | None:
    """One try at start(): True when it's up, False when it isn't, None when the start this run
    joined ended (so start() can try again).

    `user` ([pid, token]) joins the entry's users, so release() knows the run is using it.
    `keep=True` (serve, --keep-server) leaves it running after the last user goes; `keep=False`
    lets the last user stop it. `tie=True` ties a server this call starts to this process
    (procs.tie_to_me), so it dies with it however it ends: the gateway uses that.

    The lock is held only to read and write servers.json, never during the wait for the server to
    get healthy, so a run killed while it waits can't leave the lock behind for everyone else. An
    entry marked `starting` is another run's start in progress (`starter` is that run, `until`
    when its wait ends): while that start is live (_start_live) this run joins it as a user and
    waits on it, and only the starter ever stops it for being slow. A `starting` entry that isn't
    live any more is replaced like any server that isn't answering.

    If anything fails between launching the server and handing it on (the lock, servers.json), the
    server is stopped and its entry removed before the error goes up, so no unfinished start is
    left to hold the role."""
    url = _url(s, role)
    proc = None
    try:
        with _Lock(s):
            data = _read(s)
            entry = data.get(role)
            if entry and not _entry_alive(entry):          # left over from a server that's gone
                data.pop(role)
                entry = None
                _write(s, data)
            if healthy(url):
                if not ready(s, role):
                    log(f"index: port {_port(s, role)} answers but isn't running {_model(s, role)}; not using it")
                    return False
                if entry:                                   # ours: join it, never replace it
                    entry["users"] = _live_users(entry) + ([user] if user else [])
                    if keep:
                        entry["auto_stop"] = False
                    _write(s, data)
                return True                                 # someone else's: use it, never stop it
            if entry and entry.get("starting") and _start_live(entry):   # another run's start: wait with it
                entry["users"] = _live_users(entry) + ([user] if user else [])
                if keep:
                    entry["auto_stop"] = False
                _write(s, data)
                pid, token = int(entry["pid"]), entry.get("token")
            else:
                if entry:                                   # not answering, or a dead start: replace it
                    log(f"index: {role} server pid {entry['pid']} is running but not answering; stopping it")
                    procs.kill(int(entry["pid"]), entry.get("token"))
                    data.pop(role)
                    _write(s, data)
                model = _model(s, role)
                if model is None or not model.is_file():
                    log(f"index: no {role} model at {model}")
                    return False
                if not s.llama_server.is_file():
                    log(f"index: llama-server not found at {s.llama_server}")
                    return False
                logpath = s.index_dir / f"llama-{role}.log"
                flags = 0
                if sys.platform == "win32":
                    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                with open(logpath, "ab") as logf:
                    proc = subprocess.Popen(command(s, role), stdout=logf, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, creationflags=flags, env=launch_env(),
                                            cwd=str(s.llama_server.parent))
                if tie and not procs.tie_to_me(proc.pid) and sys.platform == "win32":
                    log(f"index: couldn't tie the {role} server (pid {proc.pid}) to this process; "
                        "it won't stop by itself if this process is killed")
                data[role] = {"pid": proc.pid, "token": procs.start_token(proc.pid), "port": _port(s, role),
                              "model": str(model), "since": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "auto_stop": not keep, "users": [user] if user else [], "starting": True,
                              "starter": procs.me(), "until": time.time() + wait}
                _write(s, data)  # recorded before the wait, so a crash here still leaves it tracked
    except BaseException:
        if proc is not None:
            _abandon(s, role, proc)
        raise

    if proc is None:                                    # someone else's start
        state = _wait_joined(pid, token, url, wait)
        if state == "up":
            return True
        _leave(s, role, user)
        if state == "timeout":
            log(f"index: {role} server another run is starting wasn't up after {wait:.0f}s")
            return False
        log(f"index: {role} server another run was starting has exited; trying again")
        return None

    try:
        state = _wait_healthy(proc, url, wait)
    except BaseException:
        _abandon(s, role, proc)
        raise
    if state == "up":
        try:
            with _Lock(s):
                data = _read(s)
                if (data.get(role) or {}).get("pid") == proc.pid:
                    for k in ("starting", "starter", "until"):
                        data[role].pop(k, None)
                    _write(s, data)
        except (ServerError, OSError) as e:   # it's up; the mark expires with `until` anyway
            log(f"index: {role} server is up, but servers.json couldn't be updated ({e})")
        return True
    if state == "exited":
        log(f"index: {role} server exited with code {proc.returncode}, see {s.index_dir / f'llama-{role}.log'}")
    else:
        log(f"index: {role} server not up after {wait:.0f}s; stopped it")
    _abandon(s, role, proc)
    return False


def _abandon(s: Settings, role: str, proc) -> None:
    """Stop a server this run launched and forget its entry. Never raises: it runs on the way out
    of an error too."""
    try:
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)
    except Exception:            # noqa: BLE001
        pass
    procs.untie(proc.pid)
    try:
        with _Lock(s):
            data = _read(s)
            if (data.get(role) or {}).get("pid") == proc.pid:
                data.pop(role)
                _write(s, data)
    except Exception:            # noqa: BLE001  the entry names a dead pid: the next run drops it
        pass


def stop(s: Settings, role: str) -> bool:
    """Stop the role's server if this package started it, whoever is using it (`serve --stop`).
    True when one was stopped."""
    with _Lock(s):
        data = _read(s)
        entry = data.pop(role, None)
        _write(s, data)
        return bool(entry) and procs.kill(int(entry["pid"]), entry.get("token"))


def release(s: Settings, role: str, user: list) -> bool:
    """`user` is done with the role's server. Stops it when it's `auto_stop` and nobody else is
    using it. True when it was stopped. When other runs are still using it, a tie this process
    made to it (procs.tie_to_me) is undone, so this process dying later doesn't take their server."""
    with _Lock(s):
        data = _read(s)
        entry = data.get(role)
        if not entry:
            return False
        if not _entry_alive(entry):
            data.pop(role)
            _write(s, data)
            return False
        entry["users"] = [u for u in _live_users(entry) if list(u) != list(user)]
        if entry.get("auto_stop") and not entry["users"]:
            data.pop(role)
            _write(s, data)
            stopped = procs.kill(int(entry["pid"]), entry.get("token"))
            procs.untie(int(entry["pid"]))
            return stopped
        _write(s, data)
        if entry["users"]:
            procs.untie(int(entry["pid"]))
        return False


def release_dead(s: Settings, user: list) -> list[str]:
    """`user` ([pid, token]) was a run that died without releasing its servers (one a scheduler
    killed, say). Stop each server that run was using when it's `auto_stop` and no live run
    uses it any more; a server from `serve` (not `auto_stop`) or one still in use stays. Returns
    the roles stopped."""
    stopped = []
    with _Lock(s):
        data = _read(s)
        for role in list(data):
            entry = data[role]
            if not _entry_alive(entry):
                data.pop(role)
                continue
            if list(user) not in [list(u) for u in entry.get("users") or []]:
                continue
            entry["users"] = _live_users(entry)
            if entry.get("auto_stop") and not entry["users"]:
                data.pop(role)
                if procs.kill(int(entry["pid"]), entry.get("token")):
                    stopped.append(role)
        _write(s, data)
    return stopped


def status(s: Settings) -> dict:
    data = _read(s)
    out = {}
    for role in ROLES:
        entry = data.get(role) or {}
        out[role] = {"up": healthy(_url(s, role)), "serving": serving_model(_url(s, role)),
                     "ours": _entry_alive(entry), "auto_stop": entry.get("auto_stop"),
                     "users": len(_live_users(entry)) if entry else 0,
                     "port": _port(s, role), "model": str(_model(s, role))}
    return out


class Servers:
    """Context manager: make sure the roles are up for this run, and release them on exit. A
    server is stopped on exit only when it's `auto_stop` and this was its last user."""

    def __init__(self, s: Settings, roles: list[str], keep: bool = False, log=print, wait: float = 120.0,
                 tie: bool = False):
        self.s, self.roles, self.keep, self.log, self.wait, self.tie = s, roles, keep, log, wait, tie
        self.user = procs.me()
        self.up: dict[str, bool] = {}

    def __enter__(self) -> "Servers":
        for role in self.roles:
            self.up[role] = start(self.s, role, wait=self.wait, log=self.log, user=self.user, keep=self.keep,
                                  tie=self.tie)
        return self

    def __exit__(self, *exc) -> None:
        for role, ok in self.up.items():
            if ok:
                release(self.s, role, self.user)
