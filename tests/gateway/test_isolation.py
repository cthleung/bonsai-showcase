"""Gateway isolation and edge cases: end-to-end frontmatter refusals, settings that can't be read
from the calling project's environment or folder, and a stale snippet an edit with the same size
and mtime used to keep. Every gateway process here reads a temp --env-file, never the real .env."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from common.config import ConfigError
from gateway import access, searching
from index import build, settings as index_settings
from tests.gateway.conftest import make_vault
from tests.index.conftest import FakeEmbedder
from tests.index.test_server_edge_cases import EXCLUDED, REFUSED

REPO = Path(__file__).resolve().parents[2]
QUIET = lambda *a, **k: None   # noqa: E731


@pytest.mark.parametrize("name", list(REFUSED) + list(EXCLUDED))
def test_ambiguous_and_excluded_frontmatter_is_refused_everywhere(vault, settings, name):
    rel = f"2 Areas/t/{name}.md"
    (vault / "2 Areas" / "t").mkdir(parents=True, exist_ok=True)
    (vault / rel).write_text((REFUSED | EXCLUDED)[name].replace("body", "quokka body"), encoding="utf-8")
    with pytest.raises(access.Refused) as e:
        access.read_note(vault, rel)
    assert str(e.value) == access.REFUSED
    assert rel not in {r.path for r in access.recent(vault, days=1, limit=100)}
    build.build(settings, FakeEmbedder(), log=QUIET)                # never crashes the build
    assert searching.run(settings, "quokka", mode="bm25", log=QUIET)["hits"] == []


def test_any_failure_in_read_note_is_the_fixed_refusal(vault, monkeypatch):
    from index import chunking

    def boom(text):
        raise RecursionError("deep")
    monkeypatch.setattr(chunking, "skip_reason", boom)
    with pytest.raises(access.Refused) as e:
        access.read_note(vault, "2 Areas/Profile/hometown.md")
    assert str(e.value) == access.REFUSED and "RecursionError" in e.value.reason


def test_an_edit_that_keeps_size_and_mtime_still_gets_a_fresh_snippet(vault, settings):
    target = vault / "2 Areas" / "Profile" / "stale.md"
    target.write_bytes(b"---\nname: stale\ndescription: d\ntype: user\n---\nwombatword OLDSECRET text\n")
    build.build(settings, FakeEmbedder(), log=QUIET)
    st = target.stat()
    target.write_bytes(b"---\nname: stale\ndescription: d\ntype: user\n---\nwombatword NEWWORDS0 text\n")
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert target.stat().st_size == st.st_size and target.stat().st_mtime_ns == st.st_mtime_ns
    hits = [h for h in searching.run(settings, "wombatword", 5, "bm25", log=QUIET)["hits"] if "stale" in h["path"]]
    assert hits and "OLDSECRET" not in hits[0]["snippet"] and "NEWWORDS0" in hits[0]["snippet"]


# --- settings that only the env file decides ------------------------------------------------------

@pytest.mark.parametrize("line", ["BONSAI_VAULT=~/vault", "BONSAI_VAULT=vault", "BONSAI_STATE=~/state",
                                  "BONSAI_LLAMA_SERVER=llama/llama-server.exe", "BONSAI_EMBED_MODEL=~/m.gguf"])
def test_tilde_and_relative_paths_are_refused_when_the_environment_is_ignored(tmp_path, line):
    key = line.split("=")[0]
    lines = {"BONSAI_VAULT": str(tmp_path / "v"), "BONSAI_STATE": str(tmp_path / "s")}
    lines[key] = line.split("=", 1)[1]
    env = tmp_path / "x.env"
    env.write_text("".join(f"{k}={v}\n" for k, v in lines.items()), encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        index_settings.load(env, environ=False)
    index_settings.load(env)          # the CLI's own loading still expands and resolves them


needs_mcp = pytest.mark.skipif(importlib.util.find_spec("mcp") is None,
                               reason="the gateway needs the mcp package (requirements.txt)")


def _run_gateway(env_file: Path, cwd: Path, env: dict, calls: list[dict]) -> list[dict]:
    """Send the calls, wait for every answer, then close stdin (closing it first would cancel the
    calls still running: the gateway leaves as soon as its client does)."""
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"}]
    msgs += [{"jsonrpc": "2.0", "id": 10 + i, "method": "tools/call", "params": c} for i, c in enumerate(calls)]
    p = subprocess.Popen([sys.executable, "-I", str(REPO / "gateway" / "__main__.py"), "--env-file", str(env_file)],
                         cwd=str(cwd), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    try:
        p.stdin.write("".join(json.dumps(m) + "\n" for m in msgs).encode())
        p.stdin.flush()
        out = []
        while len(out) < 1 + len(calls):
            line = p.stdout.readline()
            if not line:
                break
            out.append(json.loads(line))
        p.stdin.close()
        p.wait(30)
        return out
    finally:
        if p.poll() is None:
            p.kill()


@needs_mcp
def test_the_calling_projects_home_folder_and_imports_change_nothing(tmp_path):
    """A project that sets USERPROFILE and HOME to a decoy home, puts packages
    named like Bonsai's in its folder and on PYTHONPATH, and an env file written with ~ paths."""
    real = make_vault(tmp_path / "real" / "vault")
    (real / "3 Resources" / "realonly.md").write_text("---\ntitle: r\n---\nREALVAULT marker\n", encoding="utf-8")
    decoy_home = tmp_path / "decoyhome"
    decoy = make_vault(decoy_home / "vault")
    (decoy / "3 Resources" / "decoyonly.md").write_text("---\ntitle: d\n---\nDECOYVAULT marker\n", encoding="utf-8")
    (decoy_home / "state").mkdir()
    project = tmp_path / "project"
    project.mkdir()
    marker = tmp_path / "IMPORTED_FROM_PROJECT"
    for pkg in ("common", "index", "gateway", "mcp", "anyio", "pydantic"):
        (project / pkg).mkdir()
        (project / pkg / "__init__.py").write_text(f"open({str(marker)!r}, 'a').write('{pkg}')\n", encoding="utf-8")
    for mod in ("yaml", "json", "argparse", "sqlite_vec"):
        (project / f"{mod}.py").write_text(f"open({str(marker)!r}, 'a').write('{mod}')\n", encoding="utf-8")
    env = dict(os.environ)
    env.update(BONSAI_VAULT=str(decoy), BONSAI_STATE=str(decoy_home / "state"), PYTHONPATH=str(project),
               USERPROFILE=str(decoy_home), HOME=str(decoy_home))
    reads = [{"name": "read_note", "arguments": {"path": "3 Resources/realonly.md"}},
             {"name": "read_note", "arguments": {"path": "3 Resources/decoyonly.md"}}]

    absolute = tmp_path / "absolute.env"
    absolute.write_text(f"BONSAI_VAULT={real}\nBONSAI_STATE={tmp_path / 'real' / 'state'}\nBONSAI_RERANK_MODEL=none\n",
                        encoding="utf-8")
    out = {m["id"]: m for m in _run_gateway(absolute, project, env, reads)}
    assert "REALVAULT" in out[10]["result"]["content"][0]["text"]
    assert out[11]["result"]["isError"] is True

    tilde = tmp_path / "tilde.env"
    tilde.write_text("BONSAI_VAULT=~/vault\nBONSAI_STATE=~/state\nBONSAI_RERANK_MODEL=none\n", encoding="utf-8")
    out = {m["id"]: m for m in _run_gateway(tilde, project, env, reads)}
    for rid in (10, 11):
        assert out[rid]["result"]["isError"] is True
        assert out[rid]["result"]["content"][0]["text"].endswith(access.REFUSED)
    assert not marker.exists()
    assert list((decoy_home / "state").iterdir()) == []


@needs_mcp
def test_a_relative_env_file_is_refused(tmp_path):
    (tmp_path / "x.env").write_text("BONSAI_VAULT=C:/nowhere\n", encoding="utf-8")
    p = subprocess.run([sys.executable, "-I", str(REPO / "gateway" / "__main__.py"), "--env-file", "x.env"],
                       cwd=str(tmp_path), input=b"", capture_output=True, timeout=30)
    assert p.returncode == 2 and b"absolute" in p.stderr
