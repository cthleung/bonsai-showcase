"""Fixtures for the index tests: a made-up vault, hand-built Settings, and fake models.
No GPU, no llama-server and no real vault content. Settings are built by hand so the tests never
read the repo's .env or the real BONSAI_STATE."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from common.notes import render
from index.settings import Settings

CHAT_REL = "3 Resources/chats/chatgpt/2025/2025-03-02 home server plans (ab12cd34).md"
CHAT2_REL = "3 Resources/chats/google-ai-mode/2026/2026-10-10 dumplings (ZZyyXXww).md"
MEMORY_REL = "1 Projects/Garden/memory/water-in-the-morning.md"
PROFILE_REL = "2 Areas/Profile/hometown.md"


def chat_meta(title: str, source: str, chat_id: str, created: str, **extra) -> dict:
    meta = {"title": title, "source": source, "chat_id": chat_id, "url": "https://example.invalid/x",
            "created": created, "updated": created, "captured": created, "model": "unknown",
            "message_count": 2, "capture": "http-api", "raw": "x.json", "visibility": "private",
            "tags": ["chat"]}
    meta.update(extra)
    return meta


def chat_body(title: str, who: str, turns: list[tuple[str, str, str]]) -> str:
    out = f"# {title}\n\n{who} chat, captured automatically.\n\n"
    for n, (date, prompt, answer) in enumerate(turns, 1):
        out += f"## {n}. You ({date})\n\n{prompt}\n\n### {who}\n\n{answer}\n\n"
    return out


def write(vault: Path, rel: str, text: str) -> Path:
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_chat(vault: Path, rel: str = CHAT_REL, extra_turn: bool = False, **meta_extra) -> Path:
    turns = [
        ("2025-03-02", "how should I set up raid on my synology nas",
         "Use SHR with two drives. Open Storage Manager and create a pool.\n\n#### Sources\n\n- [Synology](https://example.invalid/syno)"),
        ("2025-03-03", "what about backing up photos from the nas to the cloud",
         "Hyper Backup can send photos to a cloud bucket every night."),
    ]
    if extra_turn:
        turns.append(("2025-03-05", "and which ups should I buy", "Any UPS with USB signalling works."))
    meta = chat_meta("home server plans", "chatgpt", "ab12cd34ef", "2025-03-02", **meta_extra)
    return write(vault, rel, render(meta, chat_body("home server plans", "ChatGPT", turns)))


@pytest.fixture
def vault(tmp_path) -> Path:
    v = tmp_path / "vault"
    write_chat(v)
    write(v, CHAT2_REL, render(
        chat_meta("dumplings", "google-ai-mode", "ZZyyXXwwvv", "2026-10-10"),
        chat_body("dumplings", "AI Mode", [
            ("2026-10-10", "where to get 小籠包 and 餃子 in Sapporo", "Try a few spots in Susukino for soup dumplings."),
            ("2026-10-11", "how do I say thank you in Japanese, arigatou or domo", "Use arigatou with friends, domo for a quick thanks."),
        ])))
    write(v, MEMORY_REL, render({"name": "water-in-the-morning", "description": "garden watering time",
                                 "type": "project", "source": "author", "date": "2026-10-01",
                                 "visibility": "private"}, "Water the tomatoes before 08:00.\n"))
    write(v, PROFILE_REL, render({"name": "hometown", "description": "where they grew up", "type": "user",
                                  "source": "author", "date": "2026-11-01", "visibility": "private"},
                                 "Grew up near the lake, moved for university.\n"))
    write(v, "4 Archive/chats-excluded/junk/2025-01-01 junk (11112222).md",
          render(chat_meta("junk", "chatgpt", "11112222", "2025-01-01", excluded="junk"),
                 chat_body("junk", "ChatGPT", [("2025-01-01", "synology nas raid junk", "junk")])))
    write(v, "0 Inbox/win/inbox note.md", "synology nas raid in the inbox, not indexed\n")
    return v


@pytest.fixture
def settings(vault, tmp_path) -> Settings:
    state = tmp_path / "state"
    state.mkdir()
    return Settings(vault=vault, state=state, llama_server=state / "none.exe",
                    embed_model=state / "fake-embed.gguf", rerank_model=None,
                    embed_port=59991, rerank_port=59992)   # ports nothing real listens on


DIMS = 2560
WORD = re.compile(r"[\w]+")


def _vec(text: str) -> list[float]:
    """A bag of words hashed into 2560 dims, so texts sharing words have nearby vectors. Only the
    first 1024 dims get values, so the Matryoshka cut keeps the signal."""
    v = [0.0] * DIMS
    for w in WORD.findall(text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % 1024] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        from index.models import cut
        self.calls.append(list(texts))
        return [cut(_vec(t)) for t in texts]

    def embed_query(self, query):
        from index.models import cut
        return cut(_vec(query))

    @property
    def embedded(self) -> int:
        return sum(len(c) for c in self.calls)


class FakeReranker:
    """Scores by how many query words appear in the doc; `reverse` flips it to prove the order comes
    from the reranker."""

    def __init__(self, reverse: bool = False, broken: bool = False):
        self.reverse, self.broken = reverse, broken

    def rerank(self, query, docs):
        if self.broken:
            return [0.0 for _ in docs]
        q = set(WORD.findall(query.lower()))
        out = []
        for d in docs:
            s = len(q & set(WORD.findall(d.lower()))) / (len(q) or 1)
            out.append(-s if self.reverse else s)
        return out


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
