from common.notes import render
from index import chunking
from index.chunking import est_tokens
from tests.index.conftest import CHAT_REL, MEMORY_REL, PROFILE_REL, chat_body, chat_meta, write


def _chunks(vault, rel, kind="chat"):
    return chunking.note_chunks(vault, vault / rel, kind)


def test_one_chunk_per_turn_with_prefix_and_turn_dates(vault):
    nc = _chunks(vault, CHAT_REL)
    assert [c.turn for c in nc.chunks] == [1, 2]
    first, second = nc.chunks
    assert first.text.startswith("[ChatGPT | 2025-03-02 | home server plans]\nYou: how should I set up raid")
    assert "ChatGPT: Use SHR with two drives" in first.text
    assert second.date == "2025-03-03"
    assert nc.kind == "chat" and nc.source == "chatgpt" and nc.date == "2025-03-02"


def test_sources_lists_and_link_urls_are_dropped(vault):
    text = _chunks(vault, CHAT_REL).chunks[0].text
    assert "Sources" not in text and "example.invalid" not in text


def test_quoted_answer_lines_are_left_out_of_the_prompt(vault):
    body = ("# t\n\nChatGPT chat.\n\n## 1. You (2025-01-01)\n\n> the model said this\n\nmy own words\n\n"
            "### ChatGPT\n\nreply\n")
    write(vault, "3 Resources/chats/chatgpt/2025/q.md", render(chat_meta("t", "chatgpt", "q1", "2025-01-01"), body))
    [c] = _chunks(vault, "3 Resources/chats/chatgpt/2025/q.md").chunks
    assert "You: my own words" in c.text and "the model said this" not in c.text


def test_long_turn_is_windowed_with_overlap_and_the_prompt_repeated(vault):
    answer = "\n".join(f"Line {i} about drives, pools and backups for the home server." for i in range(400))
    body = chat_body("long", "ChatGPT", [("2025-02-02", "explain everything about my nas please", answer)])
    rel = "3 Resources/chats/chatgpt/2025/long.md"
    write(vault, rel, render(chat_meta("long", "chatgpt", "long1", "2025-02-02"), body))
    chunks = _chunks(vault, rel).chunks
    assert len(chunks) > 3
    assert {c.turn for c in chunks} == {1} and [c.part for c in chunks] == list(range(len(chunks)))
    for c in chunks[1:]:
        assert c.text.startswith("[ChatGPT | 2025-02-02 | long]\nYou: explain everything about my nas please\n[...]\n")
    for c in chunks:
        assert est_tokens(c.text) < 1100
    # the windows overlap: the last line of one window shows up again in the next
    last_line = chunks[0].text.strip().split("\n")[-1]
    assert last_line in chunks[1].text


def test_cjk_text_without_spaces_is_windowed_by_characters():
    text = "信用卡" * 2000
    wins = chunking.windows(text, 800, 100)
    assert len(wins) > 5 and all(est_tokens(w) <= 801 for w in wins)
    for a, b in zip(wins, wins[1:]):
        assert a.endswith(b[:50])          # each window starts with the last piece of the one before
    assert sum(len(w) for w in wins) > len(text)


def test_memory_and_profile_notes_are_one_chunk_each(vault):
    [m] = _chunks(vault, MEMORY_REL, "memory").chunks
    assert m.text.startswith("[memory: Garden | 2026-10-01 | water-in-the-morning]\ngarden watering time")
    assert "tomatoes" in m.text and m.turn is None
    [p] = _chunks(vault, PROFILE_REL, "memory").chunks
    assert p.source == "profile"
    assert chunking.memory_source("2 Areas/People/Sam.md") == "people"


def test_note_files_cover_chats_and_memory_but_not_archive_or_inbox(vault):
    write(vault, "3 Resources/chats/chatgpt/.hidden.md", "x")
    rels = {p.relative_to(vault).as_posix(): k for p, k in chunking.note_files(vault)}
    assert rels == {CHAT_REL: "chat", "3 Resources/chats/google-ai-mode/2026/2026-10-10 dumplings (ZZyyXXww).md": "chat",
                    MEMORY_REL: "memory", PROFILE_REL: "memory"}


def test_excluded_chat_gives_no_chunks(vault):
    rel = "3 Resources/chats/chatgpt/2025/x.md"
    write(vault, rel, render(chat_meta("x", "chatgpt", "x1", "2025-01-01", excluded="junk"),
                             chat_body("x", "ChatGPT", [("2025-01-01", "q", "a")])))
    assert _chunks(vault, rel) is None


def test_chat_without_turn_headings_is_still_indexed(vault):
    rel = "3 Resources/chats/chatgpt/2025/flat.md"
    write(vault, rel, render(chat_meta("flat", "chatgpt", "f1", "2025-01-01"), "# flat\n\nJust some text.\n"))
    [c] = _chunks(vault, rel).chunks
    assert c.turn is None and "Just some text." in c.text
