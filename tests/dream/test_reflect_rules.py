"""dream/reflect.py: the quote matcher, question and claim rules, visibility, failed runs, month
ranges and stale scores. Every chat, quote and claim here is made up."""
import re

import pytest

from common.notes import read_note, render
from dream import filter as filt
from dream import guards, reflect
from dream.reflect import ChatDoc, Claim, Turn
from tests.dream.conftest import write  # noqa: F401
from tests.dream.test_reflect import (DAY, JULY, NOW, QUIET, Model, chat, fact, months,  # noqa: F401
                                      no_real_claude, run, score)


def doc(text, rel="a.md", vis="private", key="C1", date="2025-07-02"):
    return ChatDoc(rel, rel[:-3], rel[:-3], date, "chatgpt", vis, 5, [Turn(1, date, text, "")], key)


def record(cfg):
    return guards.load_json(filt.profile_state(cfg.state, cfg.vault, cfg.vault) / "reflect" / f"run {DAY}.json", {})


def told(quote, rel="a.md", vis="private", month="2025-07", kind="told", text="The user wants short answers."):
    return Claim(f"{month}#01", month, kind, "work", text,
                 [{"rel": rel, "stem": rel[:-3], "turn": 1, "date": f"{month}-02", "quote": quote}],
                 status="supported", visibility=vis)


# --- one quote matcher with word edges ------------------------------------------------------

@pytest.mark.parametrize("hay, quote, found", [
    ("I really dislike tomatoes in my salad", "like tomatoes in my salad", False),
    ("I want to use JavaScript for the front", "I want to use Java", False),
    ("I really like tomatoes in my salad", "like tomatoes in my salad", True),
    ("我真的很喜欢番茄沙拉", "很喜欢番茄沙拉", True),
    ("我用Rust写代码很久了，真的", "Rust写代码很久了", True),           # a CJK neighbour is an edge
    ("我真的很喜欢用Rust写代码", "我真的很喜欢用Rust", True),
    ("我用Rustlings写代码好多年", "我用Rust写代码好多年", False),
    ("나는Rust를 정말 오래 썼어요 진짜로", "Rust를 정말 오래 썼어요", True),
])
def test_find_quote_needs_word_edges(hay, quote, found):
    assert (reflect.find_quote(quote, doc(hay))[0] is not None) is found


def test_find_quote_edges_hold_around_an_ellipsis():
    c = doc("I really dislike tomatoes in my salad, and then I left the table early")
    assert reflect.find_quote("like tomatoes in ... I left the table", c)[0] is None
    assert reflect.find_quote("dislike tomatoes in ... I left the table", c)[0] is not None


def test_check_question_uses_the_same_matcher():
    cands = {"G1": told("I really dislike tomatoes in my salad")}
    q = lambda t: reflect.check_question({"kind": "working", "based_on": ["G1"], "text": t}, cands)
    assert q('You said "I really dislike tomatoes in my salad". Is that still true?')[0]
    # inside a word: "like" is the end of "dislike"
    assert q('You said "like tomatoes in my salad". Is that still true?') == \
        (None, "a quote isn't one of the checked quotes")


def test_check_question_needs_the_minimum_length():
    cands = {"G1": told("I really dislike tomatoes in my salad")}
    item = {"kind": "working", "based_on": ["G1"], "text": 'You said "tomatoes in my". Is that still true?'}
    assert reflect.check_question(item, cands) == (None, "a quote isn't one of the checked quotes")


# --- a merged claim's visibility comes from all its chats ------------------------------------

NEOVIM = "I use Neovim for all my editing work"


@pytest.mark.parametrize("first, second", [("public", "private"), ("private", "public"), ("public", "creative")])
def test_a_merged_claim_takes_the_visibility_of_all_its_chats(first, second):
    a = doc(NEOVIM, "A.md", first, "C1")
    b = doc(NEOVIM + ", at the clinic", "B.md", second, "C2", "2025-07-05")
    raw = {"claims": [fact("The user uses Neovim.", ("C1", NEOVIM)),
                      fact("The user uses Neovim.", ("C2", NEOVIM + ", at the clinic"))]}
    kept, _, counts = reflect.check_claims(raw, "2025-07", [a, b], [], [])
    assert counts["merged repeats"] == 1 and len(kept) == 1
    assert kept[0].chats == ["A.md", "B.md"] and kept[0].visibility == "private"


def test_a_merge_of_two_public_chats_stays_public():
    a, b = doc(NEOVIM, "A.md", "public", "C1"), doc(NEOVIM, "B.md", "public", "C2")
    raw = {"claims": [fact("The user uses Neovim.", ("C1", NEOVIM)), fact("The user uses Neovim.", ("C2", NEOVIM))]}
    kept, _, _ = reflect.check_claims(raw, "2025-07", [a, b], [], [])
    assert kept[0].visibility == "public"


# --- questions carry their claims' visibility into the diary ---------------------------------

def test_check_question_stores_the_visibility_of_its_claims():
    cands = {"G1": told("keep answers short please every time", "a.md", "creative"),
             "G2": told("private august words here for sure", "b.md", "private", "2025-08")}
    q, _ = reflect.check_question({"kind": "contradiction", "based_on": ["G1", "G2"],
                                   "text": 'You said "keep answers short please" and "private august words here"?'},
                                  cands)
    assert q["visibility"] == "private" and q["months"] == ["2025-07", "2025-08"]
    q, _ = reflect.check_question({"kind": "working", "based_on": ["G1"],
                                   "text": 'You said "keep answers short please". Does that hold?'}, cands)
    assert q["visibility"] == "creative"


def test_a_question_quoting_another_month_makes_the_diary_private():
    c = told("keep answers short please every time", vis="creative")
    q = {"kind": "contradiction", "based_on": ["G1", "G2"], "text": 'You said "keep answers short please"?',
         "stems": ["a", "b"], "months": ["2025-07", "2025-08"], "visibility": "private"}
    meta, body = reflect.render_diary("2025-07", [c], 1, DAY, [q])
    assert meta["visibility"] == "private" and "keep answers short" in body.split("## Questions")[1]
    assert reflect.render_diary("2025-07", [c], 1, DAY, [dict(q, visibility="creative")])[0]["visibility"] == "creative"
    assert reflect.render_diary("2025-07", [c], 1, DAY, [])[0]["visibility"] == "creative"
    no_vis = {k: v for k, v in q.items() if k != "visibility"}
    assert reflect.render_diary("2025-07", [c], 1, DAY, [no_vis])[0]["visibility"] == "private"

