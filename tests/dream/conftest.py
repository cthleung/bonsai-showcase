"""Fixtures for the dream tests: a made-up vault, a state folder, and a fake claude.
Nothing here is real vault content."""
from __future__ import annotations

from pathlib import Path

import pytest

from common import config
from common.notes import render


CHAT_REL = "3 Resources/chats/google-ai-mode/2026/2026-11-10 ramen in sapporo (AbCdEfGh).md"
INBOX_REL = "0 Inbox/win/garden idea.md"
MEMORY_REL = "1 Projects/Garden/memory/water-in-the-morning.md"


def chat_body(turns: int, size: int = 40) -> str:
    out = "# ramen in sapporo\n\nGoogle AI Mode conversation.\n\n"
    for n in range(1, turns + 1):
        out += f"## {n}. You (2026-11-10)\n\nquestion {n} {'q' * size}\n\n### AI Mode\n\nanswer {n} {'a' * size}\n\n"
    return out


def chat_meta(updated="2026-11-17", visibility="private") -> dict:
    """Updated and captured 2026-11-17, inside the night job's first-run window for the tests'
    usual night (2026-11-18, three days back)."""
    return {"title": "ramen in sapporo", "source": "google-ai-mode", "chat_id": "AbCdEfGh1234567",
            "url": "https://example.invalid/x", "created": "2026-11-10", "updated": updated,
            "captured": updated, "model": "unknown", "capture": "cdp-share-link",
            "raw": "x.json", "visibility": visibility, "tags": ["chat"]}


def chat_note(title, turns, updated="2026-11-17", captured=None, created=None, **extra):
    """A chat note whose turns are (the user's words, the answer) pairs."""
    meta = chat_meta(updated=updated)
    meta.update(title=title, captured=captured or updated, created=created or updated,
                message_count=len(turns))   # like capture: the number of turns written
    meta.update(extra)
    body = f"# {title}\n\n"
    for n, (you, answer) in enumerate(turns, 1):
        body += f"## {n}. You ({updated})\n\n{you}\n\n### AI Mode\n\n{answer}\n\n"
    return render(meta, body)


def chat_rel(name):
    return f"3 Resources/chats/google-ai-mode/2026/{name}.md"


def write(vault: Path, rel: str, text: str) -> Path:
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def vault(tmp_path) -> Path:
    v = tmp_path / "vault"
    for d in ("0 Inbox/win", "0 Inbox/mac", "1 Projects/Garden/memory", "2 Areas/Profile", "3 Resources", "4 Archive", "Tasks"):
        (v / d).mkdir(parents=True, exist_ok=True)
    write(v, CHAT_REL, render(chat_meta(), chat_body(3)))
    write(v, INBOX_REL, "Plant tomatoes along the south fence on 2026-11-12.\n")
    write(v, MEMORY_REL, render({"name": "water-in-the-morning", "description": "garden watering time",
                                 "type": "project", "source": "author", "date": "2026-10-01",
                                 "visibility": "private"}, "Water the garden before 08:00.\n"))
    return v


@pytest.fixture
def state(tmp_path) -> Path:
    s = tmp_path / "state"
    s.mkdir()
    return s


@pytest.fixture
def cfg(vault, state) -> config.Config:
    return config.Config(vault=vault, state=state, machine="win")


class FakeClaude:
    """Stands in for `claude -p`: returns queued answers, records argv and stdin."""

    def __init__(self, *answers, cost: float = 0.01):
        self.answers = list(answers)
        self.cost = cost
        self.calls: list[tuple[list[str], str, Path]] = []

    def __call__(self, args, stdin_text, cwd):
        self.calls.append((args, stdin_text, cwd))
        ans = self.answers.pop(0) if self.answers else {"proposals": []}
        if isinstance(ans, dict) and ans.get("type") == "result":
            return ans
        return {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": self.cost,
                "modelUsage": {"claude-sonnet-5-5": {}}, "structured_output": ans}


@pytest.fixture
def fake():
    return FakeClaude
