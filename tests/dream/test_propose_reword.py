"""dream/propose.py and dream/digest.py: quotes the rewording pass adds get traced, the digest check
covers only what a run writes, and the credential rule sits after the quote exception. Every vault
here is made up."""
from __future__ import annotations

import datetime as dt

import pytest

from common.notes import render
from dream import collect, digest, guards, night, propose

from tests.dream.conftest import CHAT_REL, INBOX_REL, chat_note, chat_rel, write

DAY = "2026-11-18"
QUOTE = "i want the garden watered before eight, maybe earlier in july"
AGENT_NOTE = "0 Inbox/win/2026-11-17 how the user gardens.md"
GARDEN_CHAT = chat_rel("2026-11-16 watering plan (Ww000001)")


def _agent_batch(vault, you=QUOTE, answer="Sounds good.", visibility=None):
    """An agent's note with no quote in it, and a chat that holds QUOTE (in the You turn by default)."""
    meta = {"title": "how the user gardens", "date": "2026-11-17", "source": "codex"}
    if visibility:
        meta["visibility"] = visibility
    write(vault, AGENT_NOTE, render(meta, "The user wants the garden watered early.\n"))
    write(vault, GARDEN_CHAT, chat_note("watering plan", [(you, answer)], updated="2026-11-16",
                                       message_count=1))
    batches, _ = collect.pack([p for p in collect.collect(vault, {}) if p.rel == AGENT_NOTE])
    return batches[0]


def _run(vault, state, fake, batch, *answers, existing=()):
    claude = fake(*answers)
    budget = guards.Budget(state, DAY, 1.0)
    res = propose.propose([batch], vault, list(existing), collect.homes(vault), DAY, budget, state / "work",
                          claude, log=lambda m: None)
    return res, claude


def _water(text):
    return {"action": "add", "target": "1 Projects/Garden/memory/water-before-eight.md",
            "reason": "The user set a watering time", "sources": ["S1"], "description": "watering time",
            "type": "project", "date": "2026-11-17", "text": text}


def _reworded(prop, text):
    return {"items": [{"id": "0", "reason": prop["reason"], "description": prop.get("description", "d"),
                       "text": text}]}


FIRST = "The user wants the garden watered before eight, maybe earlier in july."   # "maybe" breaks a rule
FIXED = f'The user said "{QUOTE}".'


# --- quotes added by the rewording pass get traced ------------------------------------

def test_a_quote_added_by_reword_is_traced_to_the_you_turn_it_came_from(vault, state, fake):
    b = _agent_batch(vault)
    first = _water(FIRST)
    res, claude = _run(vault, state, fake, b, {"proposals": [first]}, _reworded(first, FIXED))
    assert len(claude.calls) == 2   # the batch call, then the wording pass
    assert not res["held"]
    [p] = res["proposals"]
    assert p["text"] == FIXED
    assert p["sources"] == [AGENT_NOTE, GARDEN_CHAT]
    assert p["traced"] == [GARDEN_CHAT]
    assert p["quoted"] == [QUOTE]


def test_a_merge_into_a_career_note_becomes_a_flag_once_reword_traces_a_private_chat(vault, state, fake):
    target = "2 Areas/Career/garden-hours.md"
    write(vault, target, render({"name": "garden-hours", "description": "when the user gardens", "type": "user",
                                 "source": "author", "date": "2026-10-01", "visibility": "career"},
                                "The user gardens in the morning.\n"))
    b = _agent_batch(vault, visibility="career")   # the chat stays private (conftest's default)
    assert b.pieces[0].visibility == "career"
    first = {"action": "merge", "target": target, "reason": "The user set a watering time", "sources": ["S1"],
             "text": FIRST}
    res, _ = _run(vault, state, fake, b, {"proposals": [first]}, _reworded(first, FIXED),
                  existing=collect.existing_notes(vault))
    assert not res["held"]
    [p] = res["proposals"]
    assert p["action"] == "flag"
    assert p["visibility"] == "private"
    assert GARDEN_CHAT in p["sources"] and p["traced"] == [GARDEN_CHAT]
    assert p["reason"].endswith("(merge held: the note is career, the source is private)")


def test_the_same_merge_stays_a_merge_when_reword_adds_no_quote(vault, state, fake):
    target = "2 Areas/Career/garden-hours.md"
    write(vault, target, render({"name": "garden-hours", "description": "when the user gardens", "type": "user",
                                 "source": "author", "date": "2026-10-01", "visibility": "career"},
                                "The user gardens in the morning.\n"))
    b = _agent_batch(vault, visibility="career")
    first = {"action": "merge", "target": target, "reason": "The user set a watering time", "sources": ["S1"],
             "text": FIRST}
    res, _ = _run(vault, state, fake, b, {"proposals": [first]},
                  _reworded(first, "The user wants the garden watered before eight."),
                  existing=collect.existing_notes(vault))
    [p] = res["proposals"]
    assert p["action"] == "merge" and p["visibility"] == "career" and "traced" not in p


def test_a_reworded_quote_found_only_in_an_ai_answer_stays_held(vault, state, fake):
    b = _agent_batch(vault, you="when should I water", answer=QUOTE)
    first = _water(FIRST)
    res, _ = _run(vault, state, fake, b, {"proposals": [first]}, _reworded(first, FIXED))
    assert not res["proposals"]
    [p] = res["held"]
    assert p["sources"] == [AGENT_NOTE] and "traced" not in p

