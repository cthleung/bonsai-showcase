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


def me() -> list:
    """[pid, start token] for this process."""
    return [os.getpid(), start_token(os.getpid())]
