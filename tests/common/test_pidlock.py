"""Lock files that belong to a live process (common/pidlock.py), and capture's run_lock on top of
it. Real child processes stand in for a live holder; a pid that can't exist stands in for a dead
one. Temp folders only."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types

import pytest

from capture import __main__ as capture_cli
from common import pidlock
from index import procs

DEAD = {"pid": 999999, "token": "a process that ended long ago"}


@pytest.fixture
def child():
    """A live process that isn't this one, with its pid and start token."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    token = None
    for _ in range(50):
        token = procs.start_token(p.pid)
        if token:
            break
        time.sleep(0.05)
    yield types.SimpleNamespace(pid=p.pid, token=token, proc=p)
    p.kill()
    p.wait(10)


def write(path, data, age=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    if age:
        old = time.time() - age
        os.utime(path, (old, old))


def test_one_holder_at_a_time(tmp_path):
    path = tmp_path / "x.lock"
    with pidlock.pid_lock(path) as first:
        with pidlock.pid_lock(path) as second:
            assert first and not second
        info = json.loads(path.read_text(encoding="utf-8"))
        assert [info["pid"], info["token"]] == procs.me()
    assert not path.exists()
    with pidlock.pid_lock(path) as again:
        assert again


def test_a_dead_holder_is_taken_over(tmp_path):
    path = tmp_path / "x.lock"
    write(path, DEAD)
    with pidlock.pid_lock(path) as got:
        assert got
        assert pidlock.read(path)["pid"] == os.getpid()
    assert not path.exists()
    assert not (tmp_path / "x.lock.takeover").exists()


def test_a_live_holder_is_never_taken_over_however_old(tmp_path, child):
    path = tmp_path / "x.lock"
    write(path, {"pid": child.pid, "token": child.token}, age=30 * 24 * 3600)
    with pidlock.pid_lock(path) as got:
        assert not got
    assert pidlock.read(path) == {"pid": child.pid, "token": child.token}   # left alone
    child.proc.kill()
    child.proc.wait(10)
    with pidlock.pid_lock(path) as got:
        assert got


def test_a_reused_pid_with_another_start_time_counts_as_dead(tmp_path, child):
    path = tmp_path / "x.lock"
    write(path, {"pid": child.pid, "token": "the start time of an older process"})
    with pidlock.pid_lock(path) as got:
        assert got


def test_leaving_never_removes_someone_elses_lock(tmp_path, child):
    path = tmp_path / "x.lock"
    with pidlock.pid_lock(path) as got:
        assert got
        write(path, {"pid": child.pid, "token": child.token})   # as if another run took it over
    assert pidlock.read(path) == {"pid": child.pid, "token": child.token}


def test_old_bare_pid_locks_keep_their_age_rule(tmp_path, child):
    path = tmp_path / "capture.lock"
    write(path, str(child.pid))                       # the format before 2026-11-20
    with pidlock.pid_lock(path) as got:
        assert not got
    write(path, str(child.pid), age=4 * 3600)         # older than the old 3 hours
    with pidlock.pid_lock(path) as got:
        assert got
    write(path, "999999")                             # a bare pid that isn't running
    with pidlock.pid_lock(path) as got:
        assert got


def test_the_dream_locks_format_reads_as_a_tokenless_holder(tmp_path, child):
    path = tmp_path / "dream" / "lock"
    write(path, {"pid": child.pid, "started": "2026-11-20T03:30:00"})
    assert pidlock.held(path, legacy_stale=6 * 3600)
    write(path, {"pid": 999999, "started": "2026-11-20T03:30:00"})
    assert not pidlock.held(path, legacy_stale=6 * 3600)
    assert not pidlock.held(tmp_path / "no such lock")


def test_an_empty_lock_is_a_holder_mid_write_until_it_gets_old(tmp_path):
    path = tmp_path / "x.lock"
    write(path, "")
    with pidlock.pid_lock(path) as got:
        assert not got
    write(path, "", age=pidlock.EMPTY_GRACE + 5)
    with pidlock.pid_lock(path) as got:
        assert got


def test_two_takers_never_remove_a_fresh_lock(tmp_path, child):
    """The second taker judged the same dead holder, but by the time it acts a live run holds
    the lock: it must leave that lock alone."""
    path = tmp_path / "x.lock"
    write(path, {"pid": child.pid, "token": child.token})
    pidlock._remove_dead(path, DEAD, pidlock.LEGACY_STALE)
    assert pidlock.read(path) == {"pid": child.pid, "token": child.token}


def test_a_takeover_in_progress_blocks_a_second_one(tmp_path):
    path = tmp_path / "x.lock"
    write(path, DEAD)
    write(tmp_path / "x.lock.takeover", "")
    pidlock._remove_dead(path, DEAD, pidlock.LEGACY_STALE)
    assert path.exists()                                    # the other taker is on it
    write(tmp_path / "x.lock.takeover", "", age=pidlock.EMPTY_GRACE + 5)
    pidlock._remove_dead(path, DEAD, pidlock.LEGACY_STALE)   # a taker that died: its guard goes
    assert not (tmp_path / "x.lock.takeover").exists()
    with pidlock.pid_lock(path) as got:
        assert got


def test_a_lock_that_goes_away_between_create_and_read_is_tried_again(tmp_path, monkeypatch):
    path = tmp_path / "x.lock"
    write(path, DEAD)
    real_read = pidlock.read
    calls = []

    def read(p):
        calls.append(1)
        if len(calls) == 1:
            path.unlink()            # the holder let go just then
            return None
        return real_read(p)
    monkeypatch.setattr(pidlock, "read", read)
    with pidlock.pid_lock(path) as got:
        assert got


def test_release_retries_while_windows_has_the_file_open(tmp_path, monkeypatch):
    path = tmp_path / "x.lock"
    path.write_text("x", encoding="utf-8")
    real_unlink = type(path).unlink
    tries = []

    def unlink(self, *a, **k):
        tries.append(1)
        if len(tries) < 3:
            raise PermissionError("in use")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(type(path), "unlink", unlink)
    assert pidlock._unlink(path) is True
    assert len(tries) == 3 and not path.exists()


# --- capture's run_lock ---------------------------------------------------------------------

def test_capture_never_takes_a_live_runs_lock_however_old(tmp_path, child):
    """A long run that held capture's lock for more than 3 hours used to lose it to the hourly run,
    and then delete the hourly run's lock on its way out."""
    cfg = types.SimpleNamespace(state=tmp_path)
    path = tmp_path / "capture.lock"
    write(path, {"pid": child.pid, "token": child.token}, age=3 * 3600 + 60)
    with capture_cli.run_lock(cfg) as got:
        assert not got
    assert path.exists()


def test_capture_takes_over_a_killed_runs_lock(tmp_path):
    cfg = types.SimpleNamespace(state=tmp_path)
    write(tmp_path / "capture.lock", DEAD)
    with capture_cli.run_lock(cfg) as got:
        assert got
    assert not (tmp_path / "capture.lock").exists()
