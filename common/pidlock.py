"""Lock files that belong to a live process.

    with pid_lock(Path(state) / "capture.lock", max_age=12 * 3600, log=print) as got:
        if not got:
            ...   # another live process holds it

The file holds {"pid", "token", "since"}, where the token is the process's start time
(index/procs.py), so a reused pid is never mistaken for the holder. A lock is taken over when its
holder is gone. Leaving unlinks the file only when it still names this process.

The backstop for a hung holder: when the holder still runs but its lock is older than `max_age`
seconds, the taker stops that process (procs.kill, which checks the token, so it never stops a
process that only reuses the pid), logs one loud line naming the pid, and takes over. Without
`max_age` a live holder is never touched.

Files written before 2026-11-20 hold a bare pid (capture.lock) or {"pid", "started"} (the dream
lock), with no token. Those count as held while that pid runs and the file is younger than
`legacy_stale` seconds, the age rule they were written under. They are never killed.

The races:
  - The holder lets go between our create and our read: the read finds nothing, and we try again.
  - A file another process has open can't be deleted on Windows, and a file being deleted can't
    be created: a PermissionError counts as "held right now" and the create is tried again.
  - Two processes find the same dead holder: a short-lived guard file (`<lock>.takeover`, made with
    O_EXCL) lets one remove the dead lock, and only if the file still names that dead holder. The
    normal create then picks the winner. A guard left by a taker that died is renamed to a name of
    its own before it's deleted, and deleted only when the renamed file is the one judged stale.
  - A release that couldn't remove its own file (Windows kept it open, say) leaves a lock naming
    this process. This process takes it back next time unless it really holds it right now.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from pathlib import Path

from index import procs

LEGACY_STALE = 3 * 3600     # a tokenless lock older than this is from a crash (the old capture rule)
EMPTY_GRACE = 60            # an empty or unreadable file younger than this is a holder mid-write
ATTEMPTS = 5

_held: set[str] = set()     # the locks this process holds right now (normalized paths)


def _key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def read(path: Path) -> dict | None:
    """The holder as {"pid", "token"} ({} for an empty or unreadable file), or None when there's
    no file."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:          # open elsewhere on Windows: someone is there
        return {}
    if not text:
        return {}
    if text.isdigit():
        return {"pid": int(text), "token": None}
    try:
        data = json.loads(text)
        return {"pid": int(data["pid"]), "token": data.get("token")}
    except (ValueError, KeyError, TypeError):
        return {}


def _age(path: Path) -> float:
    try:
        return time.time() - Path(path).stat().st_mtime
    except OSError:
        return 0.0


def holder_alive(path: Path, info: dict, legacy_stale: float = LEGACY_STALE) -> bool:
    """Does the process `info` names still run? An empty file counts as alive for EMPTY_GRACE
    seconds (its holder is still writing it)."""
    pid = info.get("pid")
    if not pid:
        return _age(path) < EMPTY_GRACE
    token = info.get("token")
    if token:
        return procs.alive(int(pid), token)
    return procs.start_token(int(pid)) is not None and _age(path) < legacy_stale


def held(path: Path, legacy_stale: float = LEGACY_STALE) -> bool:
    """True when a live process holds the lock at `path`."""
    info = read(path)
    return info is not None and holder_alive(path, info, legacy_stale)


def _unlink(path: Path) -> bool:
    """Remove the file, retrying briefly when Windows says another process has it open. Never
    raises."""
    for _ in range(10):
        try:
            Path(path).unlink()
            return True
        except FileNotFoundError:
            return False
        except PermissionError:
            time.sleep(0.05)
        except OSError:
            return False
    return False


def _identity(path: Path):
    st = Path(path).stat()
    return st.st_ino, st.st_mtime_ns, st.st_size


def _break_stale_guard(guard: Path) -> None:
    """Remove a takeover guard left by a taker that died, without ever removing a fresh one: rename
    it to a name of its own, then delete it only if the renamed file is the one judged stale."""
    try:
        judged = _identity(guard)
    except OSError:
        return
    if time.time() - judged[1] / 1e9 <= EMPTY_GRACE:
        return
    mine = guard.with_name(f"{guard.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.stale")
    try:
        os.rename(guard, mine)
    except OSError:
        return                                 # someone else got there first
    try:
        same = _identity(mine) == judged
    except OSError:
        return
    if same:
        _unlink(mine)
        return
    with contextlib.suppress(OSError):       # a fresh guard came in between: give it back
        if not guard.exists():
            os.rename(mine, guard)


def _remove_dead(path: Path, dead: dict, legacy_stale: float) -> None:
    """Delete the lock if it still names the dead holder `dead`, under the takeover guard."""
    guard = Path(str(path) + ".takeover")
    try:
        fd = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except (FileExistsError, PermissionError):
        _break_stale_guard(guard)
        return
    os.close(fd)
    try:
        now = read(path)
        if now is not None and now == dead and not holder_alive(path, now, legacy_stale):
            _unlink(path)
    finally:
        _unlink(guard)


def _say(log, line: str) -> None:
    if log:
        with contextlib.suppress(Exception):
            log(line)


@contextlib.contextmanager
def pid_lock(path: Path, legacy_stale: float = LEGACY_STALE, max_age: float | None = None, log=None):
    """Yields True with the lock held, False when a live process holds it. `max_age` (seconds)
    turns on the hung-holder backstop; `log(line)` gets the lines worth keeping."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    me = procs.me()
    mine = {"pid": me[0], "token": me[1]}
    key = _key(path)
    for _ in range(ATTEMPTS):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            info = read(path)
            if info is None:                   # let go just now: try again
                continue
            if info == mine and key not in _held:
                _say(log, f"lock: {path.name} still named this process from an earlier release; "
                          f"taking it back")
                _unlink(path)
                continue
            if holder_alive(path, info, legacy_stale):
                age = _age(path)
                if not (max_age and info.get("token") and info != mine and age > max_age):
                    yield False
                    return
                _say(log, f"lock: STOPPING pid {info['pid']}, which has held {path.name} for "
                          f"{age / 3600:.1f} hours (over the {max_age / 3600:.0f} hour limit), and taking over")
                if not procs.kill(int(info["pid"]), info["token"]):
                    _say(log, f"lock: pid {info['pid']} was gone by the time it was stopped")
            _remove_dead(path, info, legacy_stale)
            time.sleep(0.02)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({**mine, "since": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        _held.add(key)
        try:
            yield True
        finally:
            _held.discard(key)
            now = read(path)
            if now == mine:
                if not _unlink(path):
                    _say(log, f"lock: couldn't remove {path.name} on the way out; this process takes "
                              f"it back next time, any other once this process ends")
            elif now not in (None, {}):
                _say(log, f"lock: {path.name} names pid {now.get('pid')} on the way out, so it was left alone")
        return
    yield False
