"""demo/run_demo.py end to end, in its own process: the sample vault, the night job, the stand-in
claude, and the digest it prints. No network, no claude CLI."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "demo" / "run_demo.py"
SAMPLE = REPO / "demo" / "sample-vault"


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


def test_the_demo_runs_the_night_job_and_prints_the_digest(tmp_path):
    before = tree_hash(SAMPLE)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BONSAI_", "CLAUDE_CLI"))}
    env["TMPDIR"] = env["TEMP"] = env["TMP"] = str(tmp_path)        # the temp folder lands here
    env["PATH"] = str(tmp_path / "no-claude-here")                  # no claude CLI to find, even by accident
    out = subprocess.run([sys.executable, str(DEMO)], cwd=tmp_path, env=env, capture_output=True,
                         encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    text = out.stdout
    assert "rehearsal on a copy of the vault: done" in text
    assert "The digest is at " in text and "digest 2026-11-18.md" in text
    assert "# Digest 2026-11-18" in text
    assert "- [ ] add | 1 Projects/Garden/memory/check-lower-leaves-on-saturdays.md |" in text
    assert "- [ ] flag | 1 Projects/Garden/memory/water-in-the-morning.md |" in text
    boxes = [line for line in text.splitlines() if line.startswith("- [ ] ")]
    assert len(boxes) == 5 and not any("capital of peru" in b for b in boxes)   # the trivia chat gives nothing
    assert tree_hash(SAMPLE) == before                               # the sample vault is never written
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("bonsai-demo-")]   # the temp folder is gone
    assert not (REPO / "demo" / "state").exists()


def test_the_sample_vault_is_in_the_note_formats():
    from common.notes import read_note
    chats = sorted((SAMPLE / "3 Resources" / "chats").rglob("*.md"))
    assert 3 <= len(chats) <= 5
    for path in chats:
        meta, body = read_note(path)
        assert {"title", "source", "chat_id", "created", "updated", "captured", "message_count",
                "capture", "raw", "visibility", "tags"} <= set(meta)
        assert path.name.endswith(f" ({meta['chat_id'][:8]}).md") and meta["visibility"] == "private"
        assert body.startswith(f"# {meta['title']}\n") and "## 1. You (" in body
    for path in (SAMPLE / "1 Projects").rglob("memory/*.md"):
        meta, _ = read_note(path)
        assert list(meta) == ["name", "description", "type", "source", "date", "visibility"]
