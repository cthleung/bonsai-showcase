"""The real `python -m gateway` process over stdio: initialize, tools/list and tools/call, with
every line on stdout checked to be JSON-RPC. Runs on a made-up vault, from a folder outside the
repo, the way Claude Code starts it in another project, with settings from its own env file."""
import dataclasses
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the gateway needs the mcp package (requirements.txt)")

from index import procs  # noqa: E402
from tests.gateway.conftest import CHAT_REL, EXCLUDED_PATH_NOTES, make_vault  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FAKE = REPO / "tests" / "index" / "fake_llama_server.py"
PROTOCOL = "2025-06-18"


def _strict(text: str):
    def no(c):
        raise ValueError(f"not strict JSON: {c}")
    return json.loads(text, parse_constant=no)


class Client:
    def __init__(self, env_file: Path, cwd: Path, env: dict):
        # the registered form: isolated, by file path, with no PYTHONPATH
        self.proc = subprocess.Popen([sys.executable, "-I", str(REPO / "gateway" / "__main__.py"),
                                      "--env-file", str(env_file)],
                                     cwd=str(cwd), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self.lines: queue.Queue = queue.Queue()
        self.raw: list[bytes] = []
        self.err: list[bytes] = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: self.err.extend(self.proc.stderr), daemon=True).start()
        self.next_id = 0

    def _read(self):
        for line in self.proc.stdout:
            self.raw.append(line)
            self.lines.put(line)
        self.lines.put(None)

    def send(self, msg) -> None:
        self.proc.stdin.write(((msg if isinstance(msg, str) else json.dumps(msg)) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def wait_for(self, rid: int, timeout: float = 60) -> dict:
        while True:
            line = self.lines.get(timeout=timeout)
            assert line is not None, f"the server exited: {b''.join(self.err).decode(errors='replace')}"
            msg = json.loads(line)
            if msg.get("id") == rid:
                return msg

    def request(self, method: str, params: dict | None = None, timeout: float = 60) -> dict:
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, **({"params": params} if params else {})})
        return self.wait_for(self.next_id, timeout)

    def start(self) -> None:
        init = self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                           "clientInfo": {"name": "pytest", "version": "0"}})
        assert init["result"]["serverInfo"]["name"] == "bonsai"
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def close(self, timeout: float = 20) -> int:
        self.proc.stdin.close()
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise


def _env_file(path: Path, s, **extra) -> Path:
    lines = {"BONSAI_VAULT": s.vault, "BONSAI_STATE": s.state, "BONSAI_LLAMA_SERVER": s.llama_server,
             "BONSAI_EMBED_MODEL": s.embed_model, "BONSAI_RERANK_MODEL": "none",
             "BONSAI_EMBED_PORT": s.embed_port, "BONSAI_RERANK_PORT": s.rerank_port, "BONSAI_GATEWAY_WAIT": 2,
             **extra}
    path.write_text("".join(f"{k}={v}\n" for k, v in lines.items()), encoding="utf-8")
    return path


@pytest.fixture
def decoy(tmp_path):
    """Another vault, set in the process environment the way a project could: the gateway must
    ignore it."""
    return make_vault(tmp_path / "decoy vault")


def _client(built, tmp_path, decoy, **extra) -> Client:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BONSAI_", "PYTHON"))}
    env.update({"PYTHONPATH": str(tmp_path / "nothing here"), "PYTHONIOENCODING": "utf-8",
                "BONSAI_VAULT": str(decoy), "BONSAI_STATE": str(tmp_path / "decoy state"),
                "BONSAI_LLAMA_SERVER": str(tmp_path / "evil.exe")})
    env.update({k: str(v) for k, v in extra.pop("process_env", {}).items()})
    elsewhere = tmp_path / "some other project"
    elsewhere.mkdir(exist_ok=True)
    return Client(_env_file(tmp_path / "gateway.env", built, **extra), elsewhere, env)


@pytest.fixture
def client(built, tmp_path, decoy):
    c = _client(built, tmp_path, decoy)
    yield c
    if c.proc.poll() is None:
        c.proc.kill()


def _payload(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


def test_stdio_round_trip(client, built, tmp_path, decoy):
    client.start()
    tools = client.request("tools/list")["result"]["tools"]
    assert {t["name"] for t in tools} == {"search", "read_note", "recent"}
    assert all(t["annotations"]["readOnlyHint"] is True for t in tools)

    res = client.request("tools/call", {"name": "search", "arguments": {"query": "synology nas"}})["result"]
    assert not res.get("isError")
    assert res.get("structuredContent") is None      # the result goes once, as text, not twice
    out = _payload(res)
    assert out["mode_asked"] == "hybrid" and out["mode_used"] == "bm25"
    assert any("keyword search only" in n for n in out["notes"])
    assert [h["path"] for h in out["hits"]] == [CHAT_REL]

    refused = client.request("tools/call", {"name": "read_note",
                                            "arguments": {"path": EXCLUDED_PATH_NOTES["0 Inbox/"]}})["result"]
    assert refused["isError"] is True and "no note an agent can read" in refused["content"][0]["text"]

    # a NaN in the raw JSON (Python's json and pydantic both accept it) is refused, and logged strictly
    client.send('{"jsonrpc":"2.0","id":900,"method":"tools/call","params":{"name":"recent","arguments":{"days":NaN}}}')
    nan = client.wait_for(900)
    assert nan.get("error") or nan["result"]["isError"] is True

    assert client.close() == 0
    for line in client.raw:                       # nothing but the protocol went to stdout
        assert json.loads(line)["jsonrpc"] == "2.0"
    log = (built.state / "gateway" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    calls = [_strict(line) for line in log]
    assert [c["tool"] for c in calls][:2] == ["search", "read_note"]
    assert calls[0]["results"] == 1 and calls[0]["args"]["query"] == "synology nas" and "ms" in calls[0]
    assert calls[1]["results"] == 0 and calls[1]["error"].startswith("refused: ")
    # the project's environment pointed at another vault and state folder: both untouched
    assert not (tmp_path / "decoy state").exists()


def _wait_entry(state: Path, timeout: float = 20) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            entry = json.loads((state / "index" / "servers.json").read_text(encoding="utf-8")).get("embed")
            if entry and procs.alive(entry["pid"], entry["token"]):
                return entry
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    raise AssertionError("the gateway never started the stand-in llama-server")


def _gone(entry: dict, timeout: float = 15) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not procs.alive(entry["pid"], entry["token"]):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def hanging(built, tmp_path, decoy):
    """A gateway whose llama-server never gets healthy, with a long wait, mid-search."""
    model = built.state / "fake-embed.gguf"
    model.write_bytes(b"x")
    s = dataclasses.replace(built, llama_server=FAKE)
    c = _client(s, tmp_path, decoy, BONSAI_GATEWAY_WAIT=60, process_env={"FAKE_LLAMA_MODE": "hang"})
    c.start()
    c.next_id += 1
    c.send({"jsonrpc": "2.0", "id": c.next_id, "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "hometown"}}})
    entry = _wait_entry(built.state)
    yield c, entry
    if c.proc.poll() is None:
        c.proc.kill()
    procs.kill(entry["pid"], entry["token"])


def test_the_gateway_exits_at_once_when_stdin_closes_and_stops_its_server(hanging, built):
    c, entry = hanging
    t0 = time.time()
    assert c.close(timeout=15) == 0
    assert time.time() - t0 < 10
    assert _gone(entry)
    assert not (built.state / "index" / "servers.lock").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="the job object that takes the server along is Windows only")
def test_a_killed_gateway_takes_its_server_along(hanging, built):
    c, entry = hanging
    c.proc.kill()
    c.proc.wait(10)
    assert _gone(entry)
    assert not (built.state / "index" / "servers.lock").exists()
