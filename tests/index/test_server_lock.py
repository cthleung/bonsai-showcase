"""index/server.py's lock and starts: the lock is never held while a server loads, a lock
left by a dead run is cleared by its pid (not its age), a run that arrives during another's start
waits on it instead of killing it, and a server tied to a process dies with it on Windows."""
import dataclasses
import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from common import pidlock
from index import procs, server

FAKE = Path(__file__).with_name("fake_llama_server.py")
REPO = Path(__file__).resolve().parents[2]
QUIET = lambda *a, **k: None   # noqa: E731


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ss(settings):
    model = settings.state / "fake-embed.gguf"
    model.write_bytes(b"x")
    s = dataclasses.replace(settings, llama_server=FAKE, embed_model=model,
                            embed_port=_free_port(), rerank_port=_free_port())
    s.index_dir.mkdir(parents=True, exist_ok=True)
    yield s
    for role in ("embed", "rerank"):
        server.stop(s, role)


def _lock(s) -> Path:
    return s.index_dir / "servers.lock"


def _dead_pid() -> tuple[int, str]:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    token = procs.start_token(p.pid) or "gone"
    p.wait()
    return p.pid, token


def test_a_lock_left_by_a_dead_run_is_cleared_at_once(ss, monkeypatch):
    pid, token = _dead_pid()
    _lock(ss).write_text(json.dumps({"pid": pid, "token": token}), encoding="utf-8")
    monkeypatch.setattr(server, "LOCK_WAIT", 2.0)
    t0 = time.time()
    with server._Lock(ss):
        assert pidlock.read(_lock(ss)) == {"pid": procs.me()[0], "token": procs.me()[1]}
    assert time.time() - t0 < 1.5 and not _lock(ss).exists()


def test_a_lock_held_by_a_live_run_is_waited_on_however_old(ss, monkeypatch):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        for _ in range(50):
            if procs.start_token(other.pid):
                break
            time.sleep(0.1)
        _lock(ss).write_text(json.dumps({"pid": other.pid, "token": procs.start_token(other.pid)}),
                             encoding="utf-8")
        old = time.time() - 10_000
        os.utime(_lock(ss), (old, old))
        monkeypatch.setattr(server, "LOCK_WAIT", 1.0)
        with pytest.raises(server.ServerError):
            with server._Lock(ss):
                pass
        assert _lock(ss).exists()          # never removed from under a live holder
    finally:
        other.kill()


def test_a_run_never_removes_a_lock_that_isnt_its_own(ss):
    lk = server._Lock(ss)
    with lk:
        _lock(ss).write_text('{"pid": 1, "token": "someone else"}', encoding="utf-8")
    assert pidlock.read(_lock(ss)) == {"pid": 1, "token": "someone else"}


def test_the_lock_is_free_while_a_server_loads_and_a_second_run_waits_on_it(ss, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    box = {}
    t = threading.Thread(target=lambda: box.update(ok=server.start(ss, "embed", wait=4, log=QUIET, keep=False)))
    t.start()
    deadline = time.time() + 5
    while time.time() < deadline and not ss.servers_file.exists():
        time.sleep(0.05)
    time.sleep(0.5)
    entry = json.loads(ss.servers_file.read_text(encoding="utf-8"))["embed"]
    assert entry["starting"] is True
    assert not _lock(ss).exists()                         # nobody holds the lock during the wait
    other, user = None, None
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        user = [other.pid, procs.start_token(other.pid)]
        t0 = time.time()
        assert server.start(ss, "embed", wait=1, log=QUIET, user=user, keep=False) is False
        assert time.time() - t0 < 3
        assert procs.alive(entry["pid"], entry["token"])   # the joiner gave up, it didn't kill the start
        t.join(10)
        assert box["ok"] is False and not procs.alive(entry["pid"], entry["token"])
        assert "embed" not in json.loads(ss.servers_file.read_text(encoding="utf-8"))
    finally:
        other.kill()


def test_a_server_that_comes_up_is_no_longer_marked_starting(ss):
    assert server.start(ss, "embed", wait=20, log=QUIET, keep=False, user=procs.me())
    entry = json.loads(ss.servers_file.read_text(encoding="utf-8"))["embed"]
    assert "starting" not in entry
    assert server.release(ss, "embed", procs.me())


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
def test_a_tied_server_dies_when_its_process_is_killed_outright(ss):
    """A child process starts a hanging server with tie=True and is then killed. The server must
    go with it, and the next run must find the lock free."""
    script = textwrap.dedent(f"""
        import dataclasses, os, sys, time
        sys.path.insert(0, {str(REPO)!r})
        os.environ["FAKE_LLAMA_MODE"] = "hang"
        from pathlib import Path
        from index import server
        from index.settings import Settings
        s = Settings(vault=Path({str(ss.vault)!r}), state=Path({str(ss.state)!r}),
                     llama_server=Path({str(FAKE)!r}), embed_model=Path({str(ss.embed_model)!r}),
                     rerank_model=None, embed_port={ss.embed_port}, rerank_port={ss.rerank_port})
        server.start(s, "embed", wait=60, log=lambda *a: None, keep=False, tie=True)
    """)
    child = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.time() + 15
        entry = None
        while time.time() < deadline:
            try:
                entry = json.loads(ss.servers_file.read_text(encoding="utf-8")).get("embed")
            except (OSError, ValueError):
                entry = None
            if entry and procs.alive(entry["pid"], entry["token"]):
                break
            time.sleep(0.1)
        assert entry and procs.alive(entry["pid"], entry["token"])
        time.sleep(0.5)
        child.kill()
        child.wait(10)
        deadline = time.time() + 10
        while time.time() < deadline and procs.alive(entry["pid"], entry["token"]):
            time.sleep(0.1)
        assert not procs.alive(entry["pid"], entry["token"])
        assert not _lock(ss).exists()
        with server._Lock(ss):
            pass
    finally:
        if child.poll() is None:
            child.kill()
        if entry:
            procs.kill(entry["pid"], entry["token"])


def test_tie_to_me_is_a_no_op_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert procs.tie_to_me(os.getpid()) is False


def test_a_lock_token_with_spaces_is_read_whole(ss, monkeypatch):
    """On macOS the start token is `ps -o lstart=` output, with spaces in it."""
    mac = "Thu Nov 20 01:06:45 2026"
    monkeypatch.setattr(procs, "alive", lambda pid, token: pid == 4242 and token == mac)
    _lock(ss).write_text(json.dumps({"pid": 4242, "token": mac}), encoding="utf-8")
    assert pidlock.held(_lock(ss)) is True
    _lock(ss).write_text(json.dumps({"pid": 4243, "token": mac}), encoding="utf-8")
    assert pidlock.held(_lock(ss)) is False


def _child_start(ss, tie: bool, untie_after: bool = False) -> subprocess.Popen:
    script = textwrap.dedent(f"""
        import os, sys, time
        sys.path.insert(0, {str(REPO)!r})
        os.environ["FAKE_LLAMA_MODE"] = "hang"
        from pathlib import Path
        from index import procs, server
        from index.settings import Settings
        s = Settings(vault=Path({str(ss.vault)!r}), state=Path({str(ss.state)!r}),
                     llama_server=Path({str(FAKE)!r}), embed_model=Path({str(ss.embed_model)!r}),
                     rerank_model=None, embed_port={ss.embed_port}, rerank_port={ss.rerank_port})
        if {untie_after!r}:
            os.environ["FAKE_LLAMA_MODE"] = ""
            server.start(s, "embed", wait=20, log=lambda *a: None, keep=False, tie=True)
            procs.untie()
        else:
            server.start(s, "embed", wait=60, log=lambda *a: None, keep=False, tie={tie!r})
    """)
    return subprocess.Popen([sys.executable, "-c", script])


def _wait_entry(ss, timeout=15) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            entry = json.loads(ss.servers_file.read_text(encoding="utf-8")).get("embed")
        except (OSError, ValueError):
            entry = None
        if entry and procs.alive(entry["pid"], entry["token"]):
            return entry
        time.sleep(0.1)
    raise AssertionError("no server entry")


def test_a_start_left_by_a_killed_run_is_replaced_not_waited_on(ss):
    child = _child_start(ss, tie=False)
    old = None
    try:
        old = _wait_entry(ss)
        assert old["starting"] is True
        child.kill()
        child.wait(10)
        assert procs.alive(old["pid"], old["token"])         # untied: the stand-in outlived its starter
        t0 = time.time()
        assert server.start(ss, "embed", wait=20, log=QUIET, keep=False, user=procs.me())
        assert time.time() - t0 < 15
        assert not procs.alive(old["pid"], old["token"])     # replaced, not joined
        assert server.release(ss, "embed", procs.me())
    finally:
        if child.poll() is None:
            child.kill()
        if old:
            procs.kill(old["pid"], old["token"])


def test_a_joiner_whose_start_dies_tries_once_on_its_own(ss, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    first = threading.Thread(target=lambda: server.start(ss, "embed", wait=2, log=QUIET, keep=False))
    first.start()
    entry = _wait_entry(ss)
    monkeypatch.delenv("FAKE_LLAMA_MODE")                     # the second try gets a healthy stand-in
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        user = [other.pid, procs.start_token(other.pid)]
        assert server.start(ss, "embed", wait=15, log=QUIET, keep=False, user=user)
        first.join(10)
        now = _wait_entry(ss)
        assert now["pid"] != entry["pid"] and "starting" not in now
    finally:
        other.kill()


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
def test_an_untied_server_outlives_a_clean_exit(ss):
    child = _child_start(ss, tie=True, untie_after=True)
    entry = None
    try:
        child.wait(30)
        entry = json.loads(ss.servers_file.read_text(encoding="utf-8")).get("embed")
        assert entry and procs.alive(entry["pid"], entry["token"])
    finally:
        if entry:
            procs.kill(entry["pid"], entry["token"])
