"""Every chat note reader splits turns with common/turns.py: the night job's cursor, the quote
exemption, the filter, reflect's quote check and the index. Made-up chats."""
from __future__ import annotations

from common.notes import render
from dream import collect, filter as filt, quotes, reflect
from index import chunking
from tests.dream.conftest import CHAT_REL, chat_meta, write

FENCE = "```"


def turn(n, you, answer, date="2026-11-10"):
    return f"## {n}. You ({date})\n\n{you}\n\n### AI Mode\n\n{answer}\n\n"


def body_of(*turns):
    return "# ramen in sapporo\n\nGoogle AI Mode conversation.\n\n" + "".join(turns)


FAKE_9 = f"An example note:\n\n{FENCE}\n## 9. You (2026-11-10)\n\nfake words in an answer\n{FENCE}\n"
FAKE_2 = f"An example note:\n\n{FENCE}\n## 2. You (2026-11-10)\n\nfake words in an answer\n{FENCE}\n"


def chat(*turns, count=None):
    meta = chat_meta()
    meta["message_count"] = len(turns) if count is None else count
    return render(meta, body_of(*turns))


# --- collect: the cursor ------------------------------------------------------------------------

def test_a_fenced_turn_9_does_not_move_the_cursor_past_real_turns(vault):
    three = [turn(1, "question 1", "a"), turn(2, "question 2", FAKE_9), turn(3, "question 3", "c")]
    write(vault, CHAT_REL, chat(*three))
    [p] = [p for p in collect.collect(vault, {}) if p.kind == "chat"]
    assert p.turns == (1, 3) and p.turns_total == 3
    seen = collect.mark_seen({}, [p])
    assert seen[CHAT_REL]["turns_done"] == 3
    write(vault, CHAT_REL, chat(*three, turn(4, "question 4", "d")))
    [p] = [p for p in collect.collect(vault, seen) if p.kind == "chat"]
    assert p.turns == (4, 4) and "question 4" in p.text


def test_a_fenced_fake_turn_2_in_a_one_turn_chat_does_not_hide_the_real_turn_2(vault):
    one = turn(1, "question 1", FAKE_2)
    write(vault, CHAT_REL, chat(one))
    [p] = [p for p in collect.collect(vault, {}) if p.kind == "chat"]
    assert p.turns == (1, 1)
    seen = collect.mark_seen({}, [p])
    write(vault, CHAT_REL, chat(one, turn(2, "real question 2", "b")))
    [p] = [p for p in collect.collect(vault, seen) if p.kind == "chat"]
    assert p.turns == (2, 2) and "real question 2" in p.text


# --- quotes: the user's own words ----------------------------------------------------------------

def test_fenced_heading_text_in_an_answer_is_not_the_users_words(vault):
    write(vault, CHAT_REL, chat(turn(1, "question 1", FAKE_2), turn(2, "question 2", "b")))
    words = quotes.OwnWords(vault).of(CHAT_REL)
    assert "question 2" in words and "fake words" not in words


def test_a_my_role_heading_does_not_cut_the_users_words(vault):
    prompt = "### My role\n\nI run the slow cooker soup club at work"
    write(vault, CHAT_REL, chat(turn(1, prompt, "Nice.")))
    words = quotes.OwnWords(vault)
    assert quotes.found_in("slow cooker soup club at work", words.of(CHAT_REL))
    assert words.chats_with("I run the slow cooker soup club at work") == [CHAT_REL]


def test_an_ambiguous_chat_gives_no_words(vault):
    write(vault, CHAT_REL, chat(turn(1, "the slow cooker soup club", "a"), count=5))
    words = quotes.OwnWords(vault)
    assert words.of(CHAT_REL) == "" and words.chats_with("I run the slow cooker soup club") == []
    assert "slow cooker" in quotes.you_turns(body_of(turn(1, "the slow cooker soup club", "a")))


# --- filter: what the filter sees ---------------------------------------------------------------

def test_the_filter_reads_session_notes_and_leaves_out_every_quote_line():
    t = "## 1. You (2026-11-10)\n\nfix the build\n\n> a quoted answer\n\nplease\n\n### Claude Code\n\nDone.\n"
    assert filt.you_text(t, "claude-code") == "fix the build\n\n\nplease"
    text, n = filt.you_turns(body_of(turn(1, "q1", FAKE_9), turn(2, "q2", "b")), "google-ai-mode")
    assert n == 2 and "fake words" not in text


# --- reflect: the quote check ---------------------------------------------------------------------

def test_reflect_finds_a_quote_after_a_my_role_heading(vault):
    write(vault, CHAT_REL, chat(turn(1, "### My role\n\nI run the slow cooker soup club at work", "Nice.")))
    c = reflect.load_chat(vault, CHAT_REL)
    assert not c.ambiguous
    found, why = reflect.find_quote("I run the slow cooker soup club at work", c)
    assert found and found.n == 1, why


def test_reflect_refuses_every_quote_from_an_ambiguous_chat(vault):
    write(vault, CHAT_REL, chat(turn(1, "I run the slow cooker soup club at work", "Nice."), count=4))
    c = reflect.load_chat(vault, CHAT_REL)
    assert c.ambiguous and [t.n for t in c.turns] == [1]
    found, why = reflect.find_quote("I run the slow cooker soup club at work", c)
    assert found is None and "told apart" in why


def test_reflect_never_takes_fenced_answer_text_for_the_user():
    turns = reflect.read_turns(body_of(turn(1, "q1", FAKE_2), turn(2, "q2", "b")), "google-ai-mode", 2)
    assert [t.n for t in turns] == [1, 2] and all("fake words" not in t.text for t in turns)


# --- index: chunks ----------------------------------------------------------------------------------

def test_index_chunks_follow_the_real_turns():
    meta = {"source": "google-ai-mode", "title": "ramen", "message_count": 2}
    chunks = chunking.chat_chunks(CHAT_REL, meta, body_of(turn(1, "q1", FAKE_2), turn(2, "q2", "b")))
    assert sorted({c.turn for c in chunks}) == [1, 2]
    assert [n for n, _, _ in chunking.split_turns(body_of(turn(1, "q1", FAKE_9)))] == [1]
