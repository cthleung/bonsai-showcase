"""Fixtures for the gateway tests: the index tests' made-up vault, plus one made-up note for every
rule that keeps a note away from agents. Nothing here reads the real vault or the repo's .env."""
from __future__ import annotations

from pathlib import Path

import pytest

from common.notes import render
from index.settings import Settings
from tests.index.conftest import (CHAT_REL, MEMORY_REL, PROFILE_REL, FakeEmbedder, chat_body, chat_meta,  # noqa: F401
                                  write, write_chat)

PROFILE_META = {"name": "x", "description": "a profile line", "type": "user", "source": "agent",
                "date": "2026-11-19", "visibility": "private"}
SECRET = "zebrafish"     # a word only the off-limits notes hold, so a search that finds it found a leak

# One note under every entry of chunking.EXCLUDED_PATHS (test_access checks this covers the list).
EXCLUDED_PATH_NOTES = {
    "0 Inbox/": "0 Inbox/win/digest 2026-11-19.md",
    "4 Archive/": "4 Archive/chats-excluded/a friend/2025-01-01 their chat (11112222).md",
    "2 Areas/Profile/diary/": "2 Areas/Profile/diary/2026-11-19.md",
    "2 Areas/Profile/Long view.md": "2 Areas/Profile/Long view.md",
    "2 Areas/Profile/rejected.md": "2 Areas/Profile/rejected.md",
}
HIDDEN_NOTES = [".obsidian-win/workspace.md", ".trash/old chat.md", "2 Areas/.hidden note.md"]
DIARY_TYPE_REL = "2 Areas/AI workflow/feelings.md"           # type: diary outside the diary folder
EXCLUDED_META_REL = "3 Resources/chats/chatgpt/2025/2025-05-05 for a friend (33334444).md"
BROKEN_REL = "2 Areas/People/broken frontmatter.md"
RESEARCH_REL = "3 Resources/research/options comparison.md"  # not indexed, but agents may read it

OFF_LIMITS = (list(EXCLUDED_PATH_NOTES.values()) + HIDDEN_NOTES
              + [DIARY_TYPE_REL, EXCLUDED_META_REL, BROKEN_REL])


def make_vault(v: Path) -> Path:
    write_chat(v)
    write(v, MEMORY_REL, render({"name": "water-in-the-morning", "description": "garden watering time",
                                 "type": "project", "source": "author", "date": "2026-10-01",
                                 "visibility": "private"}, "Water the tomatoes before 08:00.\n"))
    write(v, PROFILE_REL, render(PROFILE_META, "Grew up near the lake, moved for university.\n"))
    write(v, RESEARCH_REL, "# Options\n\nSQLite beats a server for this.\n")
    for rel in EXCLUDED_PATH_NOTES.values():
        if rel.startswith("4 Archive/"):
            write(v, rel, render(chat_meta("their chat", "chatgpt", "11112222", "2025-01-01", excluded="person"),
                                 chat_body("their chat", "ChatGPT", [("2025-01-01", f"{SECRET} plans", "ok")])))
        else:
            write(v, rel, render(PROFILE_META, f"a {SECRET} line nobody should read\n"))
    for rel in HIDDEN_NOTES:
        write(v, rel, f"{SECRET} in a dot folder\n")
    write(v, DIARY_TYPE_REL, render({**PROFILE_META, "type": "diary"}, f"{SECRET} diary entry\n"))
    write(v, EXCLUDED_META_REL, render(chat_meta("for a friend", "chatgpt", "33334444", "2025-05-05", excluded="person"),
                                       chat_body("for a friend", "ChatGPT", [("2025-05-05", f"{SECRET} question", "x")])))
    write(v, BROKEN_REL, f"---\nname: [unclosed\n---\n{SECRET} behind broken frontmatter\n")
    return v


@pytest.fixture
def vault(tmp_path) -> Path:
    return make_vault(tmp_path / "vault")


@pytest.fixture
def settings(vault, tmp_path) -> Settings:
    state = tmp_path / "state"
    state.mkdir()
    return Settings(vault=vault, state=state, llama_server=state / "no-llama-server.exe",
                    embed_model=state / "fake-embed.gguf", rerank_model=None,
                    embed_port=59981, rerank_port=59982)   # ports nothing real listens on


@pytest.fixture
def built(settings):
    """The vault indexed with the fake embedder."""
    from index import build
    build.build(settings, FakeEmbedder(), log=lambda *a: None)
    return settings

