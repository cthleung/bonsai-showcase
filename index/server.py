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
each other's entries.

- A live entry is never overwritten. A second run joins it as another user.
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
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

from index import procs
from index.localhttp import urlopen
from index.settings import Settings

COMMON = ["--host", "127.0.0.1", "-ngl", "999", "-c", "4096", "-np", "2", "-b", "2048", "-ub", "2048",
          "--no-webui", "--no-slots"]
ROLES = {
    "embed": ["--embedding", "--pooling", "last"],
    "rerank": ["--reranking", "--pooling", "rank"],
}
LOCK_WAIT = 300.0       # how long a run waits for another run's start or stop to finish
LOCK_STALE = 600.0      # a lock file older than this is from a run that died, and is removed


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
    def __init__(self, s: Settings):
        self.path = s.index_dir / "servers.lock"
        s.index_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        deadline = time.time() + LOCK_WAIT
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > LOCK_STALE:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    continue
                if time.time() > deadline:
                    raise ServerError(f"another run holds {self.path}")
                time.sleep(0.2)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


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


def start(s: Settings, role: str, wait: float = 120.0, log=print, user: list | None = None,
          keep: bool = True) -> bool:
    """Make sure the role's server is up, starting it if needed. True when it's up.

    `user` ([pid, token]) joins the entry's users, so release() knows the run is using it.
    `keep=True` (serve, --keep-server) leaves it running after the last user goes; `keep=False`
    lets the last user stop it."""
    url = _url(s, role)
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
        if entry:                                       # ours, alive, but not answering: replace it
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
        data[role] = {"pid": proc.pid, "token": procs.start_token(proc.pid), "port": _port(s, role),
                      "model": str(model), "since": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "auto_stop": not keep, "users": [user] if user else []}
        _write(s, data)          # recorded before the wait, so a crash here still leaves it tracked
        state = _wait_healthy(proc, url, wait)
        if state == "up":
            return True
        if state == "exited":
            log(f"index: {role} server exited with code {proc.returncode}, see {logpath}")
        else:
            log(f"index: {role} server not up after {wait:.0f}s; stopped it")
            proc.kill()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                pass
        data.pop(role, None)
        _write(s, data)
        return False


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
    using it. True when it was stopped."""
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
            return procs.kill(int(entry["pid"]), entry.get("token"))
        _write(s, data)
        return False


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

    def __init__(self, s: Settings, roles: list[str], keep: bool = False, log=print, wait: float = 120.0):
        self.s, self.roles, self.keep, self.log, self.wait = s, roles, keep, log, wait
        self.user = procs.me()
        self.up: dict[str, bool] = {}

    def __enter__(self) -> "Servers":
        for role in self.roles:
            self.up[role] = start(self.s, role, wait=self.wait, log=self.log, user=self.user, keep=self.keep)
        return self

    def __exit__(self, *exc) -> None:
        for role, ok in self.up.items():
            if ok:
                release(self.s, role, self.user)
