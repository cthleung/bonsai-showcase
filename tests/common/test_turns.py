"""common/turns.py: one turn parser for every chat note reader. Which headings are real turns, and
where the answer starts in each turn. Made-up chats only."""
from __future__ import annotations

from capture import chatnote, sessionnote
from capture.chatnote import Chat, Turn as CapTurn
from common.turns import parse_turns, split_turn

FENCE = "```"


def turn(n, you, answer, who="ChatGPT", date="2026-11-10"):
    return f"## {n}. You ({date})\n\n{you}\n\n### {who}\n\n{answer}\n\n"


def note(*turns):
    return "# a made-up chat\n\nChatGPT chat, captured automatically.\n\n" + "".join(turns)


# --- which headings are turns ------------------------------------------------------------

NOTE_FORMAT_ANSWER = (f"A chat note looks like this:\n\n{FENCE}markdown\n## 2. You (2026-11-10)\n\nfake question"
                      f"\n\n### ChatGPT\n\nfake answer\n\n## 3. You (2026-11-10)\n\nfake again\n{FENCE}\n\nThat's it.")


def test_a_fenced_next_number_heading_in_an_answer_is_not_a_turn():
    body = note(turn(1, "how are notes laid out", NOTE_FORMAT_ANSWER), turn(2, "real second question", "ok"))
    chat = parse_turns(body, 2)
    assert [t.n for t in chat.turns] == [1, 2] and not chat.ambiguous
    one, two = chat.turns
    assert "fake question" in one.answer and "fake question" not in one.you
    assert two.you == "real second question" and two.answer == "ok"


def test_a_fenced_wrong_number_heading_is_not_a_turn():
    answer = f"Example:\n\n{FENCE}\n## 9. You (2026-11-10)\n\nnot the user\n{FENCE}\n"
    body = note(turn(1, "first", "a"), turn(2, "second", answer), turn(3, "third", "c"))
    chat = parse_turns(body, 3)
    assert [t.n for t in chat.turns] == [1, 2, 3] and not chat.ambiguous
    assert "not the user" in chat.turns[1].answer
    assert all("not the user" not in t.you for t in chat.turns)


def test_a_real_heading_after_a_fence_that_never_closes_is_still_a_turn():
    body = note(turn(1, "first", f"code:\n\n{FENCE}python\nprint(1)\n"), turn(2, "second", "b"),
                turn(3, "third", "c"))
    chat = parse_turns(body, 3)
    assert [t.n for t in chat.turns] == [1, 2, 3] and not chat.ambiguous
    assert [t.you for t in chat.turns] == ["first", "second", "third"]


def test_without_message_count_a_heading_only_inside_a_fence_is_read_but_ambiguous():
    body = note(turn(1, "first", f"code:\n\n{FENCE}python\nprint(1)\n"), turn(2, "second", "b"))
    chat = parse_turns(body)
    assert [t.n for t in chat.turns] == [1, 2] and chat.ambiguous


def test_a_fenced_fake_in_the_last_answer_is_dropped_when_the_count_says_so():
    body = note(turn(1, "how are notes laid out", NOTE_FORMAT_ANSWER))
    chat = parse_turns(body, 1)
    assert [t.n for t in chat.turns] == [1] and not chat.ambiguous
    assert "That's it." in chat.turns[0].answer


def test_two_headings_outside_a_fence_for_one_turn_make_it_ambiguous():
    pasted = "look at this note\n\n## 2. You (2026-11-01)\n\nold pasted question"
    body = note(turn(1, "first", "a"), turn(2, pasted, "b"), turn(3, "third", "c"))
    chat = parse_turns(body, 3)
    assert chat.ambiguous


def test_message_count_mismatch_makes_it_ambiguous():
    body = note(turn(1, "first", "a"), turn(2, "second", "b"))
    assert not parse_turns(body, 2).ambiguous
    assert not parse_turns(body, "2").ambiguous
    assert parse_turns(body, 3).ambiguous
    assert not parse_turns(body, "not a number").ambiguous


def test_a_note_without_turn_1_is_read_from_its_first_heading_but_ambiguous():
    chat = parse_turns(note(turn(2, "second", "b"), turn(3, "third", "c")))
    assert [t.n for t in chat.turns] == [2, 3] and chat.ambiguous


def test_no_headings_gives_no_turns():
    chat = parse_turns("# just a note\n\nsome text\n")
    assert chat.turns == [] and not chat.ambiguous


def test_dates_and_raw_text():
    body = note(turn(1, "first", "a", date="2026-10-01"), "## 2. You\n\nno date\n\n### ChatGPT\n\nb\n")
    one, two = parse_turns(body).turns
    assert one.date == "2026-10-01" and two.date is None
    assert one.text.startswith("## 1. You (2026-10-01)") and "### ChatGPT" in one.text
    assert "## 2. You" not in one.text


# --- where the answer starts -------------------------------------------------------------

MY_ROLE = "### My role\n\nI lead the \"slow cooker soup club\" at work.\n\nwhat should I cook"


def test_my_role_heading_in_a_chat_note_prompt():
    chat = Chat(source="chatgpt", chat_id="x1", title="soup", url="https://example.invalid/c/x1",
                created="2026-11-10", updated="2026-11-10", capture="cdp-api",
                turns=[CapTurn(prompt=MY_ROLE, answer="## Ideas\n\nTry lentils.", date="2026-11-10",
                               quote="an earlier answer")])
    [t] = parse_turns(chatnote.render_body(chat), 1).turns
    assert "slow cooker soup club" in t.you and "what should I cook" in t.you
    assert "an earlier answer" not in t.you
    assert t.answer.startswith("#### Ideas") and "Try lentils." in t.answer


def test_my_role_heading_in_a_session_note_prompt():
    s = sessionnote.Session(source="claude-code", chat_id="s1", title="soup", created="2026-11-10",
                            updated="2026-11-10", agent="claude-code", agent_name="Claude Code",
                            folder="bonsai", raw=["x.jsonl"], visibility="private", host=None,
                            capture="specstory", intro="A Claude Code session.",
                            turns=[CapTurn(prompt=MY_ROLE, answer="Done.\n\n### Notes\n\nAll good.",
                                           date="2026-11-10")])
    [t] = parse_turns(sessionnote.render_body(s), 1).turns
    assert "slow cooker soup club" in t.you and "what should I cook" in t.you
    assert t.answer.startswith("Done.") and "All good." in t.answer


def test_a_turn_with_no_answer_heading_is_all_the_users():
    body = "## 1. You (2026-11-10)\n\nonly a question\n\nmore of it\n"
    [t] = parse_turns(body, 1).turns
    assert t.you == "only a question\n\nmore of it" and t.answer == ""


def test_every_quote_line_is_left_out_of_the_users_side():
    you, answer, unsure = split_turn("## 1. You\n\n> quoted answer\n\nmine\n\n> another quoted bit\n\n"
                                     "still mine\n\n### ChatGPT\n\n> answer quote\n")
    assert you == "mine\n\n\nstill mine" and answer == "> answer quote" and not unsure


def test_a_heading_inside_a_closed_fence_in_the_answer_does_not_move_the_split():
    you, answer, unsure = split_turn(f"## 1. You\n\nq\n\n### ChatGPT\n\n{FENCE}\n### a comment\n{FENCE}\n")
    assert you == "q" and "### a comment" in answer and not unsure


def test_an_answer_with_an_unclosed_fence_never_leaks_into_the_users_side():
    you, answer, unsure = split_turn(f"## 1. You\n\nq\n\n### ChatGPT\n\nsecret answer\n\n{FENCE}\n### deep\nmore\n")
    assert you == "q" and "secret answer" in answer and "### deep" in answer
    assert unsure   # an unclosed fence in the prompt would read the other way round


def test_an_unclosed_fence_with_one_answer_heading_is_not_ambiguous():
    you, answer, unsure = split_turn(f"## 1. You\n\nq\n\n### ChatGPT\n\nanswer\n\n{FENCE}\ncode\n")
    assert you == "q" and not unsure
