"""Guards and state for the night job: where its state lives, the STOP file, the lock file, the
nightly cost cap and the run log.

State lives under BONSAI_STATE/dream/, outside the vault and outside git:

    dream/lock              one run at a time (pid + start time; stale after LOCK_STALE_HOURS)
    dream/STOP              if it exists, nothing runs (so does "0 Inbox/STOP dream.md" in the vault)
    dream/spend.json        {"2026-11-18": 0.42, ...} dollars spent per run date, for the cap
    dream/night.log         one line per event, scrubbed
    dream/rehearsed.json    set after a clean run on a copy; the real vault is refused before that
    dream/                  per-vault state for the real vault: state.json, proposals/, vault-history
    dream/copy-<id>/        the same for any other vault (a copy), so a copy run never moves the
                            real vault's watermark
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

from common.notes import write_text_atomic
from common.scrub import scrub_obj, scrub_text

LOCK_STALE_HOURS = 6
DEFAULT_CAP_USD = 1.00
VAULT_STOP = Path("0 Inbox") / "STOP dream.md"


class Refused(RuntimeError):
    """A guard said no (STOP file, lock held, cap reached, real vault before a copy run)."""


# --- paths ----------------------------------------------------------------------------------

def dream_root(state: Path) -> Path:
    return Path(state) / "dream"


def same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def vault_state_dir(state: Path, vault: Path, real_vault: Path) -> Path:
    """Per-vault state folder: dream/ for the real vault, dream/copy-<id>/ for anything else."""
    root = dream_root(state)
    if same_path(vault, real_vault):
        return root
    tag = hashlib.sha256(os.path.normcase(os.path.abspath(vault)).encode()).hexdigest()[:8]
    return root / f"copy-{tag}"


def history_dir(state: Path, vault: Path, real_vault: Path) -> Path:
    """The snapshot git dir. For the real vault it's BONSAI_STATE/vault-history."""
    if same_path(vault, real_vault):
        return Path(state) / "vault-history"
    return vault_state_dir(state, vault, real_vault) / "vault-history"


# --- json state -----------------------------------------------------------------------------

def load_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def save_json(path: Path, data) -> None:
    write_text_atomic(path, json.dumps(scrub_obj(data), indent=1, ensure_ascii=False) + "\n")


# --- STOP -----------------------------------------------------------------------------------

def stop_files(state: Path, vault: Path) -> list[Path]:
    return [dream_root(state) / "STOP", Path(vault) / VAULT_STOP]


def check_stop(state: Path, vault: Path) -> None:
    for path in stop_files(state, vault):
        if path.exists():
            raise Refused(f"STOP file is there, so nothing runs: {path}")


# --- lock -----------------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lock:
    """`with Lock(state):` holds dream/lock for the run. A lock whose process is gone, or that is
    older than LOCK_STALE_HOURS, is taken over; anything else raises Refused."""

    def __init__(self, state: Path, now: _dt.datetime | None = None):
        self.path = dream_root(state) / "lock"
        self.now = now or _dt.datetime.now()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._stale():
                    self.path.unlink(missing_ok=True)
                    continue
                raise Refused(f"another dream run holds the lock: {self.path}")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "started": self.now.isoformat(timespec="seconds")}, f)
            return self
        raise Refused(f"could not take the lock: {self.path}")

    def _stale(self) -> bool:
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
            started = _dt.datetime.fromisoformat(info["started"])
            pid = int(info["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            return True
        if self.now - started > _dt.timedelta(hours=LOCK_STALE_HOURS):
            return True
        return not _pid_alive(pid)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


# --- cost cap -------------------------------------------------------------------------------

class Budget:
    """The night's dollar cap, shared by every claude call on one run date (copy runs included)."""

    MIN_CALL_USD = 0.05   # don't start a call with less than this left

    def __init__(self, state: Path, day: str, cap_usd: float = DEFAULT_CAP_USD):
        self.path = dream_root(state) / "spend.json"
        self.day = day
        self.cap = float(cap_usd)

    def spent(self) -> float:
        return float(load_json(self.path, {}).get(self.day, 0.0))

    def remaining(self) -> float:
        return max(0.0, self.cap - self.spent())

    def can_call(self) -> bool:
        return self.remaining() >= self.MIN_CALL_USD

    def add(self, usd: float) -> None:
        data = load_json(self.path, {})
        data[self.day] = round(float(data.get(self.day, 0.0)) + float(usd or 0.0), 6)
        save_json(self.path, data)


# --- the copy gate --------------------------------------------------------------------------

def rehearsed(state: Path) -> bool:
    return bool(load_json(dream_root(state) / "rehearsed.json", {}).get("ok"))


def mark_rehearsed(state: Path, copy: Path, when: str) -> None:
    save_json(dream_root(state) / "rehearsed.json", {"ok": True, "copy": str(copy), "when": when})


# --- log ------------------------------------------------------------------------------------

def log(state: Path, message: str, echo: bool = True) -> None:
    line = f"{_dt.datetime.now().isoformat(timespec='seconds')} {scrub_text(message)}"
    path = dream_root(state) / "night.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
    if echo:
        print(line, flush=True)
