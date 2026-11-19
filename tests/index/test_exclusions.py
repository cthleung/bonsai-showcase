"""The never-indexed list (index/chunking.py EXCLUDED_PATHS and EXCLUDED_TYPES): each entry is
skipped, and a note already in the index that now matches one is removed on the next build."""
import pytest

from common.notes import render
from index import build, chunking, store
from tests.index.conftest import FakeEmbedder, write

PROFILE_NOTE = {"name": "x", "description": "a profile line", "type": "user", "source": "agent",
                "date": "2026-11-19", "visibility": "private"}

# One made-up note for every entry on the list.
EXCLUDED_NOTES = {
    "0 Inbox/": "0 Inbox/digest 2026-11-19.md",
    "4 Archive/": "4 Archive/chats-excluded/junk/old.md",
    "2 Areas/Profile/diary/": "2 Areas/Profile/diary/2026-11-19.md",
    "2 Areas/Profile/Long view.md": "2 Areas/Profile/Long view.md",
    "2 Areas/Profile/rejected.md": "2 Areas/Profile/rejected.md",
}


def _indexed(settings) -> set[str]:
    db = store.connect(settings.db_path)
    try:
        return {r["path"] for r in db.execute("SELECT path FROM notes")}
    finally:
        db.close()


def test_every_excluded_path_has_a_test_note():
    assert set(EXCLUDED_NOTES) == set(chunking.EXCLUDED_PATHS)


@pytest.mark.parametrize("rel", list(EXCLUDED_NOTES.values()))
def test_each_excluded_path_is_skipped_even_if_a_glob_reaches_it(vault, monkeypatch, rel):
    monkeypatch.setattr(chunking, "MEMORY_GLOBS", chunking.MEMORY_GLOBS + ("0 Inbox/**/*.md", "4 Archive/**/*.md"))
    write(vault, rel, render(PROFILE_NOTE, "a sentence about the profile\n"))
    rels = {p.relative_to(vault).as_posix() for p, _ in chunking.note_files(vault)}
    assert rel not in rels
    assert "2 Areas/Profile/hometown.md" in rels       # the rest of the profile folder still counts


def test_matching_ignores_case():
    assert chunking.excluded_path("2 areas/profile/LONG VIEW.md")
    assert chunking.excluded_path("2 Areas/Profile/Diary/2026-01-01.md")
    assert not chunking.excluded_path("2 Areas/Profile/long view notes.md")


@pytest.mark.parametrize("rel", [r for k, r in EXCLUDED_NOTES.items() if k.startswith("2 Areas/")])
def test_an_indexed_note_that_now_matches_is_removed(settings, monkeypatch, rel):
    write(settings.vault, rel, render(PROFILE_NOTE, "a sentence about the profile\n"))
    monkeypatch.setattr(chunking, "EXCLUDED_PATHS", ())     # an index built before the list existed
    build.build(settings, FakeEmbedder())
    assert rel in _indexed(settings)
    monkeypatch.undo()
    rep = build.build(settings, FakeEmbedder())
    assert rel not in _indexed(settings) and rep.deleted == 1


def test_type_diary_is_skipped_and_removed_once_set(settings):
    rel = "2 Areas/Profile/a reflection.md"
    write(settings.vault, rel, render(PROFILE_NOTE, "a sentence about the profile\n"))
    build.build(settings, FakeEmbedder())
    assert rel in _indexed(settings)
    write(settings.vault, rel, render({**PROFILE_NOTE, "type": "Diary"}, "a sentence about the profile\n"))
    rep = build.build(settings, FakeEmbedder())
    assert rel not in _indexed(settings) and rep.skipped == 1
    assert chunking.note_chunks(settings.vault, settings.vault / rel, "memory") is None
