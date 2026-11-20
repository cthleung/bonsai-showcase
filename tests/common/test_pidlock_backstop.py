"""common/pidlock.py: the hung-holder backstop (BONSAI_LOCK_MAX_HOURS),
the stale takeover guard renamed before it's deleted, and a lock that still names this process
after a release that couldn't remove it. procs.alive and procs.kill are fakes here: nothing real is
stopped."""
from __future__ import annotations

import json
import os
import time
import types

import pytest

from capture import __main__ as capture_cli
from common import config, pidlock

HUNG = {"pid": 4242, "token": "hung since yesterday"}


def write(path, data, age=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    if age:
        old = time.time() - age
        os.utime(path, (old, old))


@pytest.fixture
def fake_procs(monkeypatch):
    """HUNG runs until procs.kill stops it; kill checks the token like the real one."""
    state = types.SimpleNamespace(alive={(HUNG["pid"], HUNG["token"])}, killed=[])
    real_alive = pidlock.procs.alive

    def alive(pid, token):
        return (pid, token) in state.alive or (pid != HUNG["pid"] and real_alive(pid, token))

    def kill(pid, token):
        if (pid, token) not in state.alive:
            return False
        state.alive.discard((pid, token))
        state.killed.append(pid)
        return True
    monkeypatch.setattr(pidlock.procs, "alive", alive)
    monkeypatch.setattr(pidlock.procs, "kill", kill)
    return state


# --- the hung-holder backstop ---------------------------------------------------------------

def test_a_live_holder_past_the_limit_is_stopped_and_taken_over(tmp_path, fake_procs):
    path, lines = tmp_path / "capture.lock", []
    write(path, HUNG, age=13 * 3600)
    with pidlock.pid_lock(path, max_age=12 * 3600, log=lines.append) as got:
        assert got
    assert fake_procs.killed == [4242]
    assert any("STOPPING pid 4242" in l and "13.0 hours" in l and "12 hour limit" in l for l in lines)


def test_a_live_holder_under_the_limit_is_left_alone(tmp_path, fake_procs):
    path = tmp_path / "capture.lock"
    write(path, HUNG, age=11 * 3600)
    with pidlock.pid_lock(path, max_age=12 * 3600) as got:
        assert not got
    assert fake_procs.killed == []


def test_without_a_limit_a_live_holder_is_never_stopped(tmp_path, fake_procs):
    path = tmp_path / "capture.lock"
    write(path, HUNG, age=30 * 24 * 3600)
    with pidlock.pid_lock(path) as got:
        assert not got
    assert fake_procs.killed == []


def test_a_tokenless_holder_is_never_stopped(tmp_path, fake_procs, monkeypatch):
    path = tmp_path / "capture.lock"
    write(path, str(HUNG["pid"]), age=2 * 3600)          # the old bare-pid format
    monkeypatch.setattr(pidlock.procs, "start_token", lambda pid: "running")
    with pidlock.pid_lock(path, max_age=60) as got:
        assert not got
    assert fake_procs.killed == []


def test_capture_uses_the_setting_and_logs_to_the_capture_log(tmp_path, fake_procs, monkeypatch):
    monkeypatch.setenv("BONSAI_VAULT", str(tmp_path))
    monkeypatch.setenv("BONSAI_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("BONSAI_LOCK_MAX_HOURS", "2")
    cfg = config.load()
    assert cfg.lock_max_hours == 2.0
    write(cfg.state / "capture.lock", HUNG, age=3 * 3600)
    with capture_cli.run_lock(cfg) as got:
        assert got
    assert fake_procs.killed == [4242]
    assert "STOPPING pid 4242" in (cfg.state / "capture.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("value, hours", [("", 12.0), ("6", 6.0), ("0.5", 0.5), ("twelve", 12.0), ("-3", 12.0)])
def test_the_lock_limit_setting_never_stops_a_command(tmp_path, monkeypatch, value, hours):
    monkeypatch.setenv("BONSAI_VAULT", str(tmp_path))
    monkeypatch.setenv("BONSAI_STATE", str(tmp_path / "state"))
    env = tmp_path / ".env"
    env.write_text(f"BONSAI_VAULT=x\nBONSAI_STATE=y\nBONSAI_LOCK_MAX_HOURS={value}\n", encoding="utf-8")
    monkeypatch.delenv("BONSAI_LOCK_MAX_HOURS", raising=False)
    assert config.load(env).lock_max_hours == hours


# --- the takeover guard ---------------------------------------------------------------------

def test_a_stale_guard_is_renamed_then_removed(tmp_path):
    guard = tmp_path / "x.lock.takeover"
    write(guard, "", age=pidlock.EMPTY_GRACE + 5)
    pidlock._break_stale_guard(guard)
    assert list(tmp_path.iterdir()) == []


def test_a_fresh_guard_is_left_alone(tmp_path):
    guard = tmp_path / "x.lock.takeover"
    write(guard, "")
    pidlock._break_stale_guard(guard)
    assert guard.exists()


def test_a_guard_that_changed_after_it_was_judged_is_given_back(tmp_path, monkeypatch):
    """Another taker swapped in a fresh guard between the age check and the rename: the renamed
    file isn't the one judged stale, so it goes back and nothing is deleted."""
    guard = tmp_path / "x.lock.takeover"
    write(guard, "fresh", age=pidlock.EMPTY_GRACE + 5)
    real = pidlock._identity
    seen = []

    def identity(path):
        seen.append(path)
        got = real(path)
        return got if len(seen) == 1 else (got[0], got[1] + 1, got[2])   # "a different file" the second time
    monkeypatch.setattr(pidlock, "_identity", identity)
    pidlock._break_stale_guard(guard)
    assert guard.read_text(encoding="utf-8") == "fresh"
    assert [p.name for p in tmp_path.iterdir()] == ["x.lock.takeover"]


def test_a_guard_someone_else_already_moved_is_not_touched(tmp_path, monkeypatch):
    guard = tmp_path / "x.lock.takeover"
    write(guard, "", age=pidlock.EMPTY_GRACE + 5)

    def rename(src, dst):
        raise FileNotFoundError(src)
    monkeypatch.setattr(pidlock.os, "rename", rename)
    pidlock._break_stale_guard(guard)
    assert guard.exists()


# --- a lock that still names this process ----------------------------------------------------

def test_a_failed_release_is_logged_and_the_lock_taken_back_next_time(tmp_path, monkeypatch):
    path, lines = tmp_path / "x.lock", []
    real_unlink = pidlock._unlink
    monkeypatch.setattr(pidlock, "_unlink", lambda p: False)       # Windows kept it open
    with pidlock.pid_lock(path, log=lines.append) as got:
        assert got
    assert path.exists()
    assert any("couldn't remove x.lock on the way out" in l for l in lines)
    monkeypatch.setattr(pidlock, "_unlink", real_unlink)
    with pidlock.pid_lock(path, log=lines.append) as again:
        assert again                                                # never waits on itself
    assert any("still named this process" in l for l in lines)
    assert not path.exists()


def test_a_lock_this_process_really_holds_is_still_refused(tmp_path):
    path = tmp_path / "x.lock"
    with pidlock.pid_lock(path) as first:
        with pidlock.pid_lock(path, max_age=0.001) as second:      # never stops itself either
            assert first and not second
    assert not path.exists()
