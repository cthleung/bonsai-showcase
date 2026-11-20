"""Index server and rule edge cases: the servers.json lock under load, unfinished starts,
per-server ties and the frontmatter rules."""
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

from index import chunking, procs, server

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


# --- the lock under load -----------------------------------------------------------------------

HAMMER = textwrap.dedent("""
    import json, os, random, sys, time
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from common import pidlock
    from index import server
    from index.settings import Settings
    ghosts = []
    state, rounds, die = Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "die"
    s = Settings(vault=state, state=state, llama_server=state / "x", embed_model=state / "m", rerank_model=None)
    server.LOCK_WAIT = 20.0
    inside, both, errors = state / "inside", 0, []
    for _ in range(rounds):
        lk = server._Lock(s)
        try:
            lk.__enter__()
        except Exception as e:
            errors.append(type(e).__name__)
            continue
        try:
            fd = os.open(inside, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); mine = True
        except FileExistsError:
            both += 1; mine = False
        if mine:
            os.unlink(inside)
        if die and random.random() < 0.3:      # leave a lock naming a dead process, as a killed run would
            ghosts.append(lk._cm)                # never finalized, so its cleanup never runs: a real death
            lk._cm = None
            pidlock._held.discard(pidlock._key(lk.path))
            lk._thread_lock.release()            # a dead process holds no thread locks either
            lk.path.write_text(json.dumps({"pid": 4000001, "token": "1234"}), encoding="utf-8")
            continue
        try:
            lk.__exit__(None, None, None)
        except Exception as e:
            errors.append(type(e).__name__)
    print(json.dumps({"both": both, "errors": errors}))
""")


@pytest.mark.parametrize("die", ["live", "die"])
def test_the_lock_holds_under_load_and_never_sticks(tmp_path, die):
    """Three processes take the lock over and over. Never two holders at once, no errors, and no
    lock file left behind (on Windows a waiter's read used to make the holder's delete fail)."""
    script = tmp_path / "hammer.py"
    script.write_text(HAMMER, encoding="utf-8")
    state = tmp_path / "state"
    (state / "index").mkdir(parents=True)
    kids = [subprocess.Popen([sys.executable, "-I", str(script), str(REPO), str(state), "300", die],
                             stdout=subprocess.PIPE, text=True) for _ in range(3)]
    results = [json.loads(k.communicate(timeout=240)[0].strip().splitlines()[-1]) for k in kids]
    assert all(r["both"] == 0 for r in results), results
    assert all(r["errors"] == [] for r in results), results
    lock = state / "index" / "servers.lock"
    assert not lock.exists() or json.loads(lock.read_text(encoding="utf-8"))["pid"] == 4000001


# --- a start that fails before it finishes can't hold the role ----------------------------------

def test_an_error_right_after_launch_stops_the_server_and_forgets_it(ss, monkeypatch):
    real_write = server._write
    seen = {}

    def failing(s, data):
        if (data.get("embed") or {}).get("starting"):
            seen["pid"], seen["token"] = data["embed"]["pid"], data["embed"]["token"]
            raise OSError("disk full")
        return real_write(s, data)

    monkeypatch.setattr(server, "_write", failing)
    with pytest.raises(OSError):
        server.start(ss, "embed", wait=20, log=QUIET, keep=False, user=procs.me())
    monkeypatch.setattr(server, "_write", real_write)
    assert seen and not procs.alive(seen["pid"], seen["token"])
    assert "embed" not in server._read(ss)
    assert server.start(ss, "embed", wait=20, log=QUIET, keep=False, user=procs.me())   # role still usable
    assert server.release(ss, "embed", procs.me())


def test_a_start_past_its_wait_is_replaced_even_with_its_starter_alive(ss, monkeypatch):
    """A starter that lives on (the gateway) but whose start went wrong can't hold the role."""
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    stuck = subprocess.Popen(server.command(ss, "embed"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # reap it as soon as it's stopped: an unreaped child still shows in `ps` on macOS, and
    # procs.kill would wait for it to go
    threading.Thread(target=stuck.wait, daemon=True).start()
    try:
        for _ in range(50):
            if procs.start_token(stuck.pid):
                break
            time.sleep(0.1)
        server._write(ss, {"embed": {"pid": stuck.pid, "token": procs.start_token(stuck.pid), "users": [],
                                     "auto_stop": True, "starting": True, "starter": procs.me(),
                                     "until": time.time() - server.START_GRACE - 1}})
        monkeypatch.delenv("FAKE_LLAMA_MODE")
        t0 = time.time()
        assert server.start(ss, "embed", wait=20, log=QUIET, keep=False, user=procs.me())
        assert time.time() - t0 < 15
        deadline = time.time() + 10                         # replaced: the stuck one was stopped
        while stuck.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        assert stuck.poll() is not None
        assert server.release(ss, "embed", procs.me())
    finally:
        if stuck.poll() is None:
            stuck.kill()


def test_threads_of_one_process_never_hold_the_lock_together(ss):
    """The gateway's search worker and its shutdown can both take the lock, with the same pid and
    token in the file. A thread lock keeps them apart, while another process contends too."""
    script = ss.state / "hammer.py"
    script.write_text(HAMMER, encoding="utf-8")
    other = subprocess.Popen([sys.executable, "-I", str(script), str(REPO), str(ss.state), "200", "live"],
                             stdout=subprocess.PIPE, text=True)
    inside = ss.state / "inside"
    box = {"both": 0, "errors": []}

    def hammer():
        for _ in range(200):
            try:
                with server._Lock(ss):
                    try:
                        fd = os.open(inside, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                        os.unlink(inside)
                    except FileExistsError:
                        box["both"] += 1
            except Exception as e:   # noqa: BLE001
                box["errors"].append(type(e).__name__)

    threads = [threading.Thread(target=hammer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(240)
    theirs = json.loads(other.communicate(timeout=240)[0].strip().splitlines()[-1])
    assert box == {"both": 0, "errors": []} and theirs == {"both": 0, "errors": []}
    assert not (ss.index_dir / "servers.lock").exists()


# --- one job per server: a release that leaves other users unties it ---------------------------

CHILD = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from index import procs, server
    from index.settings import Settings
    st, role, port = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    s = Settings(vault=st, state=st, llama_server=Path(sys.argv[5]), embed_model=st / "fake-embed.gguf",
                 rerank_model=None, embed_port=port, rerank_port=port + 1)
    me = procs.me()
    ok = server.start(s, "embed", wait=30, log=lambda *a: None, keep=False, user=me, tie=(role == "G"))
    print(role, "up", ok, flush=True)
    if role == "G":
        while not (st / "go").exists():
            time.sleep(0.05)
        server.release(s, "embed", me)
        print("G released", flush=True)
    time.sleep(120)
""")


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
def test_a_released_server_another_run_uses_survives_the_gateway_being_killed(ss, tmp_path):
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")

    def child(role):
        return subprocess.Popen([sys.executable, "-I", str(script), str(REPO), str(ss.state), role,
                                 str(ss.embed_port), str(FAKE)], stdout=subprocess.PIPE, text=True)

    g = n = None
    entry = None
    try:
        g = child("G")
        assert g.stdout.readline().strip() == "G up True"
        n = child("N")
        assert n.stdout.readline().strip() == "N up True"
        (ss.state / "go").write_text("1")
        assert g.stdout.readline().strip() == "G released"
        entry = server._read(ss)["embed"]
        g.kill()
        g.wait(10)
        time.sleep(1.5)
        assert procs.alive(entry["pid"], entry["token"])      # N's server lives on
    finally:
        for p in (g, n):
            if p and p.poll() is None:
                p.kill()
        if entry:
            procs.kill(entry["pid"], entry["token"])


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
def test_untie_closes_only_that_servers_job():
    a = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    b = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert procs.tie_to_me(a.pid) and procs.tie_to_me(b.pid)
        assert procs.untie(a.pid) and a.pid not in procs._jobs and b.pid in procs._jobs
        assert not procs.untie(a.pid)
    finally:
        procs.untie(b.pid)
        a.kill()
        b.kill()


# --- frontmatter the rules must refuse ----------------------------------------------------------

REFUSED = {
    "dots closer then a type block": "---\ntitle: a\n...\ntype: diary\n---\nbody\n",
    "exact duplicate type": "---\ntype: diary\ntype: note\n---\nbody\n",
    "self-referencing alias": "---\ntype: &a [*a]\n---\nbody\n",
    "alias bomb": ("---\na: &a [x,x,x,x,x,x,x,x,x,x]\nb: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]\n"
                   "c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]\nd: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]\n"
                   "e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d,*d]\ntype: *e\n---\nbody\n"),
}
EXCLUDED = {
    "case duplicate Type then type": "---\nType: diary\ntype: note\n---\nbody\n",
    "case duplicate type then Type": "---\ntype: note\nType: diary\n---\nbody\n",
    "merge key": "---\nbase: &b {type: diary}\n<<: *b\n---\nbody\n",
    "anchor": "---\nx: &d diary\ntype: *d\n---\nbody\n",
}
PASSES = {
    "ordinary": "---\ntitle: a\ntype: note\n---\nbody\n",
    "dots closer alone": "---\ntitle: a\n...\nbody\n",
    "rule pair with a colon in prose": "---\nNote: this is prose.\n---\nbody\n",
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_ambiguous_or_runaway_frontmatter_is_unreadable(name):
    t0 = time.time()
    assert chunking.skip_reason(REFUSED[name]) == "unreadable"
    assert time.time() - t0 < 2


@pytest.mark.parametrize("name", list(EXCLUDED))
def test_every_key_that_folds_to_type_counts(name):
    assert chunking.skip_reason(EXCLUDED[name]) == "excluded"


@pytest.mark.parametrize("name", list(PASSES))
def test_plain_frontmatter_still_passes(name):
    assert chunking.skip_reason(PASSES[name]) is None


def test_skip_reason_never_raises(monkeypatch):
    def boom(meta):
        raise RecursionError("deep")
    monkeypatch.setattr(chunking, "excluded_meta", boom)
    assert chunking.skip_reason("---\ntype: note\n---\nbody\n") == "unreadable"


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
def test_a_job_left_for_a_reused_pid_is_replaced_not_trusted():
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        procs._jobs[p.pid] = k32.CreateJobObjectW(None, None)   # an entry from an earlier process with this pid
        assert procs.tie_to_me(p.pid)
        h = k32.OpenProcess(0x1000, False, p.pid)                # PROCESS_QUERY_LIMITED_INFORMATION
        inside = wintypes.BOOL()
        assert k32.IsProcessInJob(h, procs._jobs[p.pid], ctypes.byref(inside)) and inside.value
        k32.CloseHandle(h)
    finally:
        procs.untie(p.pid)
        p.kill()
