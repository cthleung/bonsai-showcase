"""The llama-server bookkeeping, run against a small stand-in server (fake_llama_server.py), so
no GPU or real llama-server is needed."""
import dataclasses
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from index import procs, server

FAKE = Path(__file__).with_name("fake_llama_server.py")


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
    yield s
    for role in ("embed", "rerank"):        # never leave a stand-in running
        server.stop(s, role)


def _entry(s, role="embed"):
    return json.loads(s.servers_file.read_text(encoding="utf-8")).get(role)


def _other_user():
    """A live process that isn't this one, to stand in for a second run."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    for _ in range(50):
        if procs.start_token(p.pid):
            break
        time.sleep(0.1)
    return p, [p.pid, procs.start_token(p.pid)]


def test_servers_stop_their_own_server_when_the_run_ends(ss):
    with server.Servers(ss, ["embed"], log=lambda m: None) as sv:
        assert sv.up["embed"]
        e = _entry(ss)
        assert e["auto_stop"] and procs.alive(e["pid"], e["token"])
    assert _entry(ss) is None and not procs.alive(e["pid"], e["token"])
    assert not server.healthy(ss.embed_url)


def test_a_second_run_joins_and_the_last_one_out_stops_it(ss):
    other, user = _other_user()
    try:
        assert server.start(ss, "embed", user=user, keep=False, log=lambda m: None)
        first = _entry(ss)
        with server.Servers(ss, ["embed"], log=lambda m: None):
            e = _entry(ss)
            assert e["pid"] == first["pid"] and len(e["users"]) == 2     # joined, not replaced
        assert procs.alive(first["pid"], first["token"])                  # the other run still has it
        assert server.release(ss, "embed", user)
        assert not procs.alive(first["pid"], first["token"])
    finally:
        other.kill()


def test_keep_and_serve_leave_it_running(ss):
    with server.Servers(ss, ["embed"], keep=True, log=lambda m: None):
        pass
    e = _entry(ss)
    assert e and not e["auto_stop"] and procs.alive(e["pid"], e["token"])
    assert server.stop(ss, "embed") and not procs.alive(e["pid"], e["token"])


def test_a_server_that_never_gets_healthy_is_killed_and_forgotten(ss, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    logs = []
    assert not server.start(ss, "embed", wait=2, log=logs.append)
    assert _entry(ss) is None and any("not up after" in m for m in logs)
    time.sleep(0.5)
    with socket.socket() as s:                 # the port is free again: the process is gone
        if sys.platform != "win32":
            # The stand-in closed each health check's connection itself (HTTP/1.0), so those sit in
            # TIME_WAIT on the port for a while after it dies. On macOS and Linux a plain bind fails
            # over them; SO_REUSEADDR lets it through, and still fails while anything listens there.
            # (On Windows the same flag would let the bind take a port that is still in use.)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", ss.embed_port))


def test_a_server_that_exits_early_is_forgotten(ss, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "exit")
    logs = []
    assert not server.start(ss, "embed", wait=10, log=logs.append)
    assert _entry(ss) is None and any("exited" in m for m in logs)


def test_a_reused_pid_is_never_killed(ss):
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        time.sleep(0.3)
        ss.index_dir.mkdir(parents=True, exist_ok=True)
        ss.servers_file.write_text(json.dumps({"embed": {"pid": victim.pid, "token": "not its start time",
                                                         "users": [], "auto_stop": True}}), encoding="utf-8")
        assert not server.stop(ss, "embed")
        assert victim.poll() is None
    finally:
        victim.kill()


def test_the_launch_env_drops_llama_arg_variables(ss, monkeypatch, tmp_path):
    out = tmp_path / "env.json"
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    monkeypatch.setenv("llama_arg_port", "1")
    monkeypatch.setenv("FAKE_LLAMA_ENV_OUT", str(out))
    assert server.start(ss, "embed", log=lambda m: None)
    names = [n.upper() for n in json.loads(out.read_text(encoding="utf-8"))]
    assert not any(n.startswith("LLAMA_ARG_") for n in names) and "FAKE_LLAMA_ENV_OUT" in names


def test_a_server_on_the_port_with_another_model_is_not_used(ss):
    other = dataclasses.replace(ss, embed_model=ss.state / "other.gguf")
    (ss.state / "other.gguf").write_bytes(b"x")
    assert server.start(other, "embed", log=lambda m: None)
    logs = []
    assert not server.start(ss, "embed", log=logs.append)
    assert any("isn't running" in m for m in logs)
    server.stop(other, "embed")


def test_command_flags(settings):
    cmd = server.command(settings, "embed")
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1" and "--no-slots" in cmd and "--no-webui" in cmd
