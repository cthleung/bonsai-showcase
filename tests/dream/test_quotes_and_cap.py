"""The user's quotes are exempt from the writing rules, the cap holds, and a fact from an agent's
note also cites the chat its quote came from. Every note here is made up."""
import datetime as dt
import json
import subprocess

import pytest

from common.notes import render
from dream import collect, digest, guards, night, propose, quotes
from tests.dream.conftest import chat_note, chat_rel, write

NOW = dt.datetime(2026, 11, 18, 3, 0, 0)
OQ, CQ = quotes.OPEN_QUOTE, quotes.CLOSE_QUOTE   # curly double quotes
DAY = "2026-11-18"


# --- the user's quotes are exempt -----------------------------------------------------------

TEAM_TURN = "i joined the Seed Library Architecture group at the community garden, starting in July"
TEAM_CHAT = chat_rel("2026-11-17 garden group (Kk000001)")


def group_vault(vault):
    write(vault, TEAM_CHAT, chat_note("garden group", [
        (TEAM_TURN, "Great! In an architecture group you might plan the beds."),
    ]))
    batches, _ = collect.pack([p for p in collect.collect(vault, {}) if p.rel == TEAM_CHAT])
    return batches[0]


def group(text, description="The user joined a garden group"):
    return {"action": "profile", "target": "2 Areas/Profile/garden-group.md", "reason": "The user stated their group",
            "sources": ["S1"], "description": description, "date": "2026-11-17", "text": text}


def run_propose(vault, state, fake, batch, *answers):
    claude = fake(*answers)
    budget = guards.Budget(state, DAY, 1.0)
    res = propose.propose([batch], vault, [], collect.homes(vault), DAY, budget, state / "work", claude,
                          log=lambda m: None)
    return res, claude


def test_an_unquoted_group_name_from_the_user_is_still_held(vault, state, fake):
    # a group name with a banned word, copied from the user's turn without quote marks
    b = group_vault(vault)
    text = "The user is on the Seed Library Architecture group at the community garden."
    res, claude = run_propose(vault, state, fake, b, {"proposals": [group(text)]},
                              {"items": [{"id": "0", "reason": "The user stated their group",
                                          "description": "The user joined a garden group", "text": text}]})
    [held] = res["held"]
    assert held["held"] == ['banned word: "Architectur"'] and not res["proposals"]
    assert "exactly as written" in claude.calls[1][1], "the reword prompt says how to quote the user"


def test_the_same_name_in_quotes_from_the_user_passes_and_the_digest_is_written(vault, state, fake):
    b = group_vault(vault)
    text = 'the user is on the "Seed Library Architecture" group at the community garden.'
    res, claude = run_propose(vault, state, fake, b, {"proposals": [group(text)]})
    [p] = res["proposals"]
    assert p["quoted"] == ["Seed Library Architecture"] and not res["held"]
    assert len(claude.calls) == 1, "nothing to reword"
    path, body = digest.write(vault, state, DAY, "03:00", res["proposals"], res["held"], [], "Read 1 chat.")
    assert '"Seed Library Architecture"' in body


def test_reword_can_fix_it_by_quoting_the_user(vault, state, fake):
    b = group_vault(vault)
    res, _ = run_propose(vault, state, fake, b,
                         {"proposals": [group("The user is on the Seed Library Architecture group.")]},
                         {"items": [{"id": "0", "reason": "The user stated their group",
                                     "description": "The user joined a garden group",
                                     "text": "The user is on the " + OQ + "seed library architecture" + CQ + " group."}]})
    [p] = res["proposals"]
    assert p["text"] == "The user is on the " + OQ + "seed library architecture" + CQ + " group." and not res["held"]


@pytest.mark.parametrize("text", [
    'The AI said "you might plan the beds".',            # only in the answer, not the user's turn
    'The user said "my hobby is architecture for gardens".',  # in no source at all: the model quoting itself
])
def test_quotes_not_from_the_user_get_no_exemption(vault, state, fake, text):
    b = group_vault(vault)
    res, _ = run_propose(vault, state, fake, b, {"proposals": [group(text)]},
                         {"items": [{"id": "0", "reason": "r", "description": "d", "text": text}]})
    assert not res["proposals"] and res["held"]


def test_the_user_quoting_an_answer_back_is_not_their_own_words():
    body = ("## 1. You (2026-11-17)\n\n> In an architecture group you might plan the beds.\n\n"
            "what does that mean\n\n### AI Mode\n\nIt means plans.\n")
    words = quotes.you_turns(body)
    assert "what does that mean" in words and "architecture" not in words and "plans" not in words


def test_an_exempt_quote_from_an_earlier_run_does_not_stop_the_next_digest(vault, state, fake):
    b = group_vault(vault)
    res, _ = run_propose(vault, state, fake, b,
                         {"proposals": [group('the user is on the "Seed Library Architecture" group.')]})
    digest.write(vault, state, DAY, "03:00", res["proposals"], [], [], "Read 1 chat.")
    path, body = digest.write(vault, state, DAY, "04:00", [], [], [], "Read 0 chats.")
    assert body.count('"Seed Library Architecture"') == 1 and "# Run at 04:00" in body


def test_a_banned_word_outside_the_quote_is_still_caught(vault, state, fake):
    b = group_vault(vault)
    text = 'the user is on the "Seed Library Architecture" group and might stay.'
    res, _ = run_propose(vault, state, fake, b, {"proposals": [group(text)]},
                         {"items": [{"id": "0", "reason": "r", "description": "d", "text": text}]})
    [held] = res["held"]
    assert held["held"] == ['hedging: "might"']


@pytest.mark.parametrize("text", [
    'The user "might" move to the coast.',              # one word the user used somewhere: a scare quote
    'The house "floor is structural" in their words.',  # the user wrote "structurally": not a whole word
])
def test_one_word_quotes_and_word_pieces_get_no_exemption(vault, state, fake, text):
    rel = chat_rel("2026-11-17 moving (Mm000001)")
    write(vault, rel, chat_note("moving", [("i might move, the new floor is structurally sound", "Nice.")]))
    batches, _ = collect.pack([p for p in collect.collect(vault, {}) if p.rel == rel])
    res, _ = run_propose(vault, state, fake, batches[0], {"proposals": [group(text)]},
                         {"items": [{"id": "0", "reason": "r", "description": "d", "text": text}]})
    assert not res["proposals"] and res["held"] and res["held"][0]["quoted"] == []


# --- the cap holds --------------------------------------------------------------------------

def test_a_call_that_wont_fit_whats_left_is_never_started(state, fake):
    budget = guards.Budget(state, DAY, 0.50)
    budget.add(0.42)
    claude = fake({"proposals": []})
    prompt = "x" * 100_000   # about $0.26 by the estimate, with $0.08 left
    with pytest.raises(propose.CapReached, match="is left"):
        propose.call(prompt, propose.SCHEMA, budget, state, claude)
    assert claude.calls == [] and budget.spent() == pytest.approx(0.42)


def test_the_cli_gets_whats_left_rounded_down(state, fake):
    budget = guards.Budget(state, DAY, 0.50)
    budget.add(0.4226)
    claude = fake({"proposals": []})
    propose.call("x", propose.SCHEMA, budget, state, claude)
    args = claude.calls[0][0]
    assert args[args.index("--max-budget-usd") + 1] == "0.07"


def test_below_the_floor_no_call_starts(state, fake):
    budget = guards.Budget(state, DAY, 0.50)
    budget.add(0.46)
    claude = fake({"proposals": []})
    with pytest.raises(propose.CapReached, match="cap is reached"):
        propose.call("x", propose.SCHEMA, budget, state, claude)
    assert claude.calls == []


def test_a_timeout_charges_what_the_call_was_allowed(state, monkeypatch):
    monkeypatch.setenv("CLAUDE_CLI", "claude-for-tests")

    def too_slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(propose.subprocess, "run", too_slow)
    budget = guards.Budget(state, DAY, 0.50)
    budget.add(0.10)
    with pytest.raises(propose.ProposeError, match="charged \\$0.40"):
        propose.call("x", propose.SCHEMA, budget, state / "work")
    assert budget.spent() == pytest.approx(0.50)


def test_no_claude_cli_charges_nothing(state, monkeypatch):
    monkeypatch.delenv("CLAUDE_CLI", raising=False)
    monkeypatch.setattr(propose.shutil, "which", lambda name: None)
    budget = guards.Budget(state, DAY, 0.50)
    with pytest.raises(propose.NotStarted, match="isn't on PATH"):
        propose.call("x", propose.SCHEMA, budget, state / "work")
    assert budget.spent() == 0.0


@pytest.mark.parametrize("bad", [
    {"type": "result", "subtype": "success", "is_error": False, "structured_output": {"proposals": []}},
    ["not", "a", "result"],
])
def test_bad_output_with_no_cost_charges_what_the_call_was_allowed(state, bad):
    budget = guards.Budget(state, DAY, 0.30)
    with pytest.raises(propose.ProposeError, match="no cost"):
        propose.call("x", propose.SCHEMA, budget, state, lambda a, s, c: bad)
    assert budget.spent() == pytest.approx(0.30)


def test_a_per_call_cap_limits_the_cli_and_the_charge_on_failure(state):
    # the profile filter passes max_call_usd; a failed call is charged that, not what's left of the cap
    seen = []

    def no_cost(args, stdin, cwd):
        seen.append(args[args.index("--max-budget-usd") + 1])
        return {"type": "result", "subtype": "success", "is_error": False, "structured_output": {}}

    budget = guards.Budget(state, DAY, 6.00)
    with pytest.raises(propose.ProposeError, match="charged \\$0.25"):
        propose.call("x", propose.SCHEMA, budget, state, no_cost, max_call_usd=0.25)
    assert seen == ["0.25"]
    assert budget.spent() == pytest.approx(0.25)


def test_the_night_stops_at_the_cap_without_an_error_and_never_passes_it(vault, state, fake):
    pieces = collect.collect(vault, {})
    batches, _ = collect.pack(pieces, max_call=1)   # one piece per call
    claude = fake({"proposals": []}, {"proposals": []}, cost=0.04)
    budget = guards.Budget(state, DAY, 0.10)
    res = propose.propose(batches, vault, [], [], DAY, budget, state, claude, log=lambda m: None)
    assert len(claude.calls) == 1 and res["error"] is None and "call 2 of 2 not started" in res["capped"]
    assert res["read"] == batches[0].pieces
    assert budget.spent() <= budget.cap and res["cost"] == pytest.approx(0.04)


def test_a_capped_night_writes_its_digest_and_keeps_the_rest(cfg, fake, monkeypatch):
    real_pack = collect.pack
    monkeypatch.setattr(night.collect, "pack", lambda ps: real_pack(ps, max_call=1))
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    claude = fake({"proposals": []}, {"proposals": []}, cost=0.04)
    assert night.run(cfg, runner=claude, now=NOW, cap_usd=0.10) == 0
    text = (cfg.vault / "0 Inbox" / f"digest {DAY}.md").read_text(encoding="utf-8")
    assert "1 note wait for the next night" in text
    st = guards.load_json(guards.dream_root(cfg.state) / "state.json", {})
    assert st["watermark"] == "2026-11-15"


# --- provenance -----------------------------------------------------------------------------

QUOTE = "i want the garden watered before eight, maybe earlier in july"
AGENT_NOTE = "0 Inbox/win/2026-11-17 how the user gardens.md"
GARDEN_CHAT = chat_rel("2026-11-16 watering plan (Ww000001)")


def agent_vault(vault, source="codex", chat_vis="private", turn=QUOTE, answer="Sounds good."):
    write(vault, AGENT_NOTE, render({"title": "how the user gardens", "date": "2026-11-17", "source": source},
                                    "The user said " + OQ + QUOTE + CQ + ".\n"))
    meta_note = chat_note("watering plan", [(turn, answer)], updated="2026-11-16")
    write(vault, GARDEN_CHAT, meta_note.replace("visibility: private", f"visibility: {chat_vis}"))
    batches, _ = collect.pack([p for p in collect.collect(vault, {}) if p.rel == AGENT_NOTE])
    return batches[0]


def water(text=f'the user said "{QUOTE}".'):
    return {"action": "add", "target": "1 Projects/Garden/memory/water-before-eight.md",
            "reason": "The user set a watering time", "sources": ["S1"], "description": "watering time",
            "type": "project", "date": "2026-11-17", "text": text}


def test_a_fact_from_an_agent_note_also_cites_the_chat_its_quote_came_from(vault, state, fake):
    b = agent_vault(vault, chat_vis="career")
    res, claude = run_propose(vault, state, fake, b, {"proposals": [water()]})
    [p] = res["proposals"]
    assert p["sources"] == [AGENT_NOTE, GARDEN_CHAT]
    assert p["links"] == ["2026-11-17 how the user gardens", "2026-11-16 watering plan (Ww000001)"]
    assert p["visibility"] == "private", "a mixed set of sources is private"
    assert p["quoted"] == [QUOTE] and len(claude.calls) == 1, "the hedge inside the user's quote is exempt"
    digest.write(vault, state, DAY, "03:00", res["proposals"], [], [], "Read 1 inbox note.")
    side = json.loads(digest.sidecar_path(state, DAY).read_text(encoding="utf-8"))
    assert "[[2026-11-16 watering plan (Ww000001)]] ^p01" in side["proposals"][0]["line"]


def test_the_agent_note_alone_does_not_make_its_quote_exempt(vault, state, fake):
    b = agent_vault(vault, turn="when should I water", answer=QUOTE)   # the words are only in an answer
    res, _ = run_propose(vault, state, fake, b, {"proposals": [water()]},
                         {"items": [{"id": "0", "reason": "r", "description": "d", "text": water()["text"]}]})
    assert not res["proposals"] and res["held"][0]["sources"] == [AGENT_NOTE]


def test_a_note_the_user_wrote_gets_no_extra_source_but_its_words_count(vault, state, fake):
    b = agent_vault(vault, source="author")
    res, _ = run_propose(vault, state, fake, b, {"proposals": [water()]})
    [p] = res["proposals"]
    assert p["sources"] == [AGENT_NOTE] and p["quoted"] == [QUOTE]


def test_short_quotes_and_excluded_chats_never_name_a_chat(vault):
    agent_vault(vault)
    words = quotes.OwnWords(vault)
    assert words.chats_with("before eight") == []
    assert words.chats_with(QUOTE) == [GARDEN_CHAT]
    text = (vault / GARDEN_CHAT).read_text(encoding="utf-8").replace("tags: [chat]", "tags: [chat]\nexcluded: junk")
    write(vault, GARDEN_CHAT, text)
    assert quotes.OwnWords(vault).chats_with(QUOTE) == []


def test_matching_ignores_case_quote_marks_and_dash_spacing():
    assert quotes.norm("I" + chr(0x2019) + "d rather " + chr(0x2014) + " go") == quotes.norm("i'd  rather - go")
    assert quotes.quotes_in('a "b c" and ' + OQ + 'd e' + CQ) == ["b c", "d e"]
    assert quotes.strip_quoted('x "Big Architecture" y "other"', ["big architecture"]) == 'x "" y "other"'
