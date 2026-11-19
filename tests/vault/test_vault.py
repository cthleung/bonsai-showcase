"""Tests for vault/: folders and templates, and Index.md.
Everything runs on temp folders with made-up notes; nothing reads a real vault."""
from pathlib import Path

import pytest

from common import notes
from vault import index, make_folders
from vault.templates import FOLDERS, TEMPLATES, TEMPLATES_DIR

BANNED = ["\u2014", "\u2013", " half", "weight", "gravity", "structural", "load-bearing",
          "scaffold", "framework", "architecture"]


@pytest.fixture
def vault(tmp_path, monkeypatch):
    v = tmp_path / "My Vault"   # a space, like the real one
    v.mkdir()
    monkeypatch.setenv("BONSAI_VAULT", str(v))
    monkeypatch.setenv("BONSAI_STATE", str(tmp_path / "state"))
    return v


# --- make_folders ---------------------------------------------------------------------------

def test_make_folders_creates_everything(vault):
    made = make_folders.make(vault)
    for rel in FOLDERS:
        assert (vault / rel).is_dir()
    for name in TEMPLATES:
        assert (vault / TEMPLATES_DIR / name).is_file()
    assert len(made) == len(FOLDERS) + len(TEMPLATES)


def test_make_folders_never_overwrites_and_reruns_clean(vault):
    (vault / TEMPLATES_DIR).mkdir()
    mine = vault / TEMPLATES_DIR / "Task.md"
    mine.write_text("my own task template\n", encoding="utf-8")
    make_folders.make(vault)
    assert mine.read_text(encoding="utf-8") == "my own task template\n"
    assert make_folders.make(vault) == []


def test_make_folders_dry_run_writes_nothing(vault):
    made = make_folders.make(vault, dry_run=True)
    assert made and list(vault.iterdir()) == []


def test_templates_match_the_plan_formats():
    for name, text in TEMPLATES.items():
        meta, body = notes.parse(text)
        assert meta, name
        assert meta.get("visibility", "private") == "private"
    task_meta, task_body = notes.parse(TEMPLATES["Task.md"])
    assert set(task_meta) >= {"title", "status", "priority", "projects", "tags"}
    assert task_meta["tags"] == ["task"]
    assert task_body.index("## Story") < task_body.index("## Steps") < task_body.index("## Comments")
    assert "- [ ]" in task_body
    mem_meta, _ = notes.parse(TEMPLATES["Memory.md"])
    assert list(mem_meta) == ["name", "description", "type", "source", "date", "visibility"]


def test_templates_follow_writing_rules():
    for text in TEMPLATES.values():
        for word in BANNED:
            assert word not in text.lower()


# --- index ----------------------------------------------------------------------------------

def test_index_lists_notes_and_counts_chats(vault):
    make_folders.make(vault)
    notes.write_note(vault / "North Star.md", {"title": "North Star"}, "# North Star\n")
    notes.write_note(vault / "1 Projects" / "Garden" / "Garden.md",
                     {"title": "Garden", "description": "Garden project", "status": "active"}, "# Garden\n")
    notes.write_note(vault / "2 Areas" / "Cooking" / "tea-rule.md",
                     {"name": "tea-rule", "description": "Tea first, mail me at a@b.example"}, "x\n")
    for i in range(3):
        notes.write_note(vault / "3 Resources" / "chats" / "chatgpt" / "2026" / f"c{i}.md", {"title": "c"}, "x\n")
    notes.write_note(vault / "4 Archive" / "old.md", {}, "x\n")
    (vault / ".obsidian-win").mkdir()
    (vault / ".obsidian-win" / "x.md").write_text("x\n", encoding="utf-8")

    changed, _ = index.write(vault)
    assert changed
    text = (vault / "Index.md").read_text(encoding="utf-8")
    assert "- [[North Star]]" in text
    assert "- [[1 Projects/Garden/Garden]] - Garden project" in text
    assert "- [[2 Areas/Cooking/tea-rule]] - Tea first, mail me at `a@b.example`" in text
    assert "Chats from chatgpt: 3" in text and "Archived notes: 1" in text
    assert "Templates/" not in text and ".obsidian" not in text and "c0" not in text
    assert text.index("North Star") < text.index("1 Projects") < text.index("2 Areas")
    # every @ sits in a code span, so Claude Code won't read it as an import
    for line in text.splitlines():
        stripped = "".join(line.split("`")[0::2])
        assert "@" not in stripped
    assert index.write(vault)[0] is False   # unchanged on a second run
    assert "Index" not in [l for l in text.splitlines() if l.startswith("- [[Index")]


def test_index_survives_bad_frontmatter(vault):
    (vault / "broken.md").write_text("---\n: : :\n---\nbody\n", encoding="utf-8")
    assert "- [[broken]]" in index.build(vault)
