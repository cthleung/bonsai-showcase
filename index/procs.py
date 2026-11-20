"""Small process helpers: is a pid still the same process, and stop it.

A pid alone isn't enough, because Windows and Unix both reuse pids. Each process is identified by
(pid, start token), where the start token is its creation time (Windows: GetProcessTimes; elsewhere:
`ps -o lstart=`). A process counts as alive only when both match, and nothing is ever killed
without that check.
"""
from __future__ import annotations

import os
import subprocess
import sys


def start_token(pid: int) -> str | None:
    """The process's creation time as a string, or None when no such process is running."""
    if pid <= 0:
        return None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:   # STILL_ACTIVE
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k32.GetProcessTimes(h, *[ctypes.byref(t) for t in times]):
                return None
            created = times[0]
            return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        finally:
            k32.CloseHandle(h)
    out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True)
    return out.stdout.strip() or None


def alive(pid: int, token: str | None) -> bool:
    """True when `pid` is running and is the same process that had `token`."""
    return token is not None and start_token(pid) == token


def kill(pid: int, token: str | None) -> bool:
    """Stop the process if it's still the one with `token`. True when it was stopped."""
    if not alive(pid, token):
        return False
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        os.kill(pid, 15)
    import time
    for _ in range(40):
        if not alive(pid, token):
            return True
        time.sleep(0.25)
    if sys.platform != "win32":
        os.kill(pid, 9)
    return True


_me: list | None = None


def me() -> list:
    """[pid, start token] for this process (worked out once per process)."""
    global _me
    if _me is None or _me[0] != os.getpid():
        _me = [os.getpid(), start_token(os.getpid())]
    return list(_me)


_jobs: dict = {}   # pid -> the job object handle tying that process to this one


def _win():
    """(ctypes, kernel32 with the job object calls typed, the extended limit structure)."""
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    return ctypes, k32, Extended


def _kill_on_close(job, on: bool) -> bool:
    ctypes, k32, Extended = _win()
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000 if on else 0     # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    return bool(k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)))   # extended limits


def tie_to_me(pid: int) -> bool:
    """Windows: put `pid` in a job object of its own that only this process holds, with
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, so Windows kills it when this process ends, even when this
    process is killed outright. One job per process, so untie() can let one go and keep the rest.
    True when it worked. Elsewhere there's no such thing: False, and the caller's own cleanup is
    all there is."""
    if sys.platform != "win32" or pid <= 0:
        return False
    untie(pid)       # an entry left for an earlier process with this pid (reused) isn't a tie to this one
    _, k32, _ = _win()
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return False
    h = None
    try:
        if not _kill_on_close(job, True):
            return False
        h = k32.OpenProcess(0x0100 | 0x0001, False, pid)          # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        if not h or not k32.AssignProcessToJobObject(job, h):
            return False
        _jobs[pid], job = job, None
        return True
    finally:
        if h:
            k32.CloseHandle(h)
        if job:
            k32.CloseHandle(job)


def untie(pid: int | None = None) -> bool:
    """Let a process tie_to_me() tied outlive this one after all (another run is using it now):
    the job's kill flag is cleared and its handle closed. `pid=None` unties every one. True when
    one was untied."""
    if sys.platform != "win32":
        return False
    _, k32, _ = _win()
    done = False
    for p in [pid] if pid is not None else list(_jobs):
        job = _jobs.pop(p, None)
        if job is None:
            continue
        _kill_on_close(job, False)
        k32.CloseHandle(job)
        done = True
    return done
