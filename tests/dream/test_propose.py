import json

import pytest

from common.notes import render
from dream import collect, guards, propose
from tests.dream.conftest import CHAT_REL, INBOX_REL, MEMORY_REL, write

DAY = "2026-11-18"


def batch_for(vault):
    batches, _ = collect.pack(collect.collect(vault, {}))
    return batches[0]


def keys(batch):
    return {p.rel: p.key for p in batch.pieces}


def add(target="1 Projects/Garden/memory/tomatoes-by-the-fence.md", sources=("S2",), **kw):
    item = {"action": "add", "target": target, "reason": "The user picked a spot", "sources": list(sources),
            "description": "where the tomatoes go", "type": "project", "date": "2026-11-12",
            "text": "Tomatoes go along the south fence."}
    item.update(kw)
    return item


# --- the call -------------------------------------------------------------------------------

def test_args_pin_the_model_turn_tools_off_and_keep_content_off_the_command_line(vault, state, fake):
    claude = fake({"proposals": []})
    b = batch_for(vault)
    budget = guards.Budget(state, DAY, 1.0)
    prompt = propose.build_prompt(b, collect.existing_notes(vault), collect.homes(vault), DAY)
    propose.call(prompt, propose.SCHEMA, budget, state / "work", claude)
    args, stdin_text, _ = claude.calls[0]
    assert args[args.index("--model") + 1] == "claude-sonnet-5-5"
    assert args[args.index("--tools") + 1] == ""
    for flag in ("-p", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in args
    assert args[args.index("--output-format") + 1] == "json"
    assert json.loads(args[args.index("--json-schema") + 1]) == propose.SCHEMA
    assert "--fallback-model" not in args
    assert "question 1" in stdin_text and "Plant tomatoes" in stdin_text
    assert not any("tomatoes" in a or "question" in a for a in args)
    assert "visibility" not in json.dumps(propose.SCHEMA)


def test_cost_goes_on_the_ledger_and_the_cap_stops_calls(state, fake):
    budget = guards.Budget(state, DAY, 0.10)
    claude = fake({"proposals": []}, cost=0.07)
    propose.call("x", propose.SCHEMA, budget, state, claude)
    assert budget.spent() == pytest.approx(0.07)
    with pytest.raises(propose.ProposeError, match="cap"):
        propose.call("x", propose.SCHEMA, budget, state, claude)
    assert len(claude.calls) == 1


def test_budget_stop_and_missing_output_are_errors_but_still_cost(state, fake):
    budget = guards.Budget(state, DAY, 5.0)
    stopped = {"type": "result", "subtype": "error_max_budget_usd", "is_error": True, "total_cost_usd": 0.3}
    with pytest.raises(propose.ProposeError, match="error_max_budget_usd"):
        propose.call("x", propose.SCHEMA, budget, state, fake(stopped))
    bare = {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.2}
    with pytest.raises(propose.ProposeError, match="no structured output"):
        propose.call("x", propose.SCHEMA, budget, state, fake(bare))
    assert budget.spent() == pytest.approx(0.5)


# --- validation -----------------------------------------------------------------------------

def test_valid_add_gets_visibility_from_its_source_not_the_model(vault):
    b = batch_for(vault)
    k = keys(b)
    raw = {"proposals": [add(sources=[k[INBOX_REL]], visibility="public")]}
    [p], rejected = propose.validate(raw, b, vault, collect.existing_notes(vault), DAY)
    assert not rejected
    assert p["visibility"] == "private" and p["name"] == "tomatoes-by-the-fence"
    assert p["links"] == ["garden idea"] and p["sources"] == [INBOX_REL]


def test_mixed_source_visibility_is_private(vault):
    meta_note = render({"visibility": "career"}, "career note\n")
    write(vault, "0 Inbox/win/career.md", meta_note)
    write(vault, INBOX_REL, render({"visibility": "career"}, "other\n"))
    b = batch_for(vault)
    k = keys(b)
    raw = {"proposals": [add(sources=[k[INBOX_REL], k["0 Inbox/win/career.md"]])]}
    [p], _ = propose.validate(raw, b, vault, [], DAY)
    assert p["visibility"] == "career"
    raw = {"proposals": [add(sources=[k[INBOX_REL], k[CHAT_REL]])]}
    [p], _ = propose.validate(raw, b, vault, [], DAY)
    assert p["visibility"] == "private"


@pytest.mark.parametrize("item, why", [
    (add(sources=["S9"]), "no known source"),
    (add(target="../outside.md"), "bad target"),
    (add(target="C:/Windows/x.md"), "bad target"),
    (add(target="3 Resources/chats/x.md"), "not a memory home"),
    (add(target=MEMORY_REL), "already exists"),
    (add(target="2 Areas/Profile/Not Kebab.md"), "kebab"),
    ({"action": "merge", "target": "1 Projects/Garden/memory/nope.md", "reason": "r", "sources": ["S1"], "text": "t"}, "no such note"),
    ({"action": "move", "target": "3 Resources/chats/google-ai-mode/2026/x.md", "reason": "r", "sources": ["S1"], "move_to": "4 Archive"}, "no such note"),
    ({"action": "move", "target": INBOX_REL, "reason": "r", "sources": ["S2"], "move_to": "0 Inbox/mac"}, "bad move_to"),
    ({"action": "delete", "target": MEMORY_REL, "reason": "r", "sources": ["S1"]}, "unknown action"),
])
def test_bad_proposals_are_rejected(vault, item, why):
    b = batch_for(vault)
    out, rejected = propose.validate({"proposals": [item]}, b, vault, collect.existing_notes(vault), DAY)
    assert out == [] and why in rejected[0]


def test_profile_goes_to_the_profile_area_as_a_user_fact(vault):
    b = batch_for(vault)
    raw = {"proposals": [add(action="profile", target="2 Areas/Food/likes-ramen.md", sources=["S1"], type="project")]}
    [p], _ = propose.validate(raw, b, vault, [], DAY)
    assert p["target"] == "2 Areas/Profile/likes-ramen.md" and p["type"] == "user"


def test_merge_into_a_less_private_note_becomes_a_flag(vault):
    write(vault, MEMORY_REL, render({"name": "water-in-the-morning", "visibility": "public"}, "Water early.\n"))
    b = batch_for(vault)
    raw = {"proposals": [{"action": "merge", "target": MEMORY_REL, "reason": "more detail", "sources": ["S2"],
                          "text": "Also water the tomatoes."}]}
    [p], _ = propose.validate(raw, b, vault, collect.existing_notes(vault), DAY)
    assert p["action"] == "flag" and "public" in p["reason"]


def test_dashes_and_pipes_are_tidied(vault):
    b = batch_for(vault)
    raw = {"proposals": [add(reason="spot " + chr(0x2014) + " chosen | done", text="a " + chr(0x2013) + " b")]}
    [p], _ = propose.validate(raw, b, vault, [], DAY)
    assert p["reason"] == "spot - chosen / done" and p["text"] == "a - b"


def test_dedupe_drops_repeats_and_ones_waiting_in_an_earlier_digest():
    a = {"action": "add", "target": "x.md"}
    b = {"action": "add", "target": "y.md"}
    assert propose.dedupe([a, dict(a), b], taken={("add", "y.md")}) == [a]


# --- the whole step -------------------------------------------------------------------------

def test_propose_runs_batches_rewords_and_holds(vault, state, fake):
    b = batch_for(vault)
    k = keys(b)
    first = {"proposals": [add(sources=[k[INBOX_REL]]),
                           add(target="1 Projects/Garden/memory/ramen-maybe.md", sources=[k[CHAT_REL]],
                               text="The user might like ramen."),
                           add(target="1 Projects/Garden/memory/second.md", sources=[k[CHAT_REL]],
                               text="It seems fine.")]}
    reworded = {"items": [{"id": "0", "reason": "The user said so", "description": "ramen", "text": "The user likes ramen."},
                          {"id": "1", "reason": "r", "description": "d", "text": "It probably works."}]}
    claude = fake(first, reworded)
    budget = guards.Budget(state, DAY, 1.0)
    res = propose.propose([b], vault, [], collect.homes(vault), DAY, budget, state / "work", claude, log=lambda m: None)
    assert [p["target"].rsplit("/", 1)[1] for p in res["proposals"]] == ["tomatoes-by-the-fence.md", "ramen-maybe.md"]
    assert res["proposals"][1]["text"] == "The user likes ramen."
    [held] = res["held"]
    assert held["target"].endswith("second.md") and held["held"]
    assert len(claude.calls) == 2 and res["read"] == b.pieces and res["error"] is None


def test_a_failed_call_stops_the_night_and_keeps_earlier_work(vault, state, fake):
    pieces = collect.collect(vault, {})
    batches, _ = collect.pack(pieces, max_call=1)   # one piece per call
    assert len(batches) == 2
    stopped = {"type": "result", "subtype": "error_during_execution", "is_error": True, "total_cost_usd": 0.0}
    claude = fake({"proposals": []}, stopped)
    res = propose.propose(batches, vault, [], [], DAY, guards.Budget(state, DAY, 1.0), state, claude, log=lambda m: None)
    assert res["read"] == batches[0].pieces and "call 2 of 2" in res["error"]


def test_moves_only_from_this_computers_inbox_or_outside_the_inbox(vault):
    write(vault, "0 Inbox/mac/from the mac.md", "A note from the Mac.\n")
    b = batch_for(vault)
    k = keys(b)

    def move(target):
        return {"action": "move", "target": target, "reason": "belongs with the garden",
                "sources": [k[INBOX_REL]], "move_to": "1 Projects/Garden/"}

    raw = {"proposals": [move(INBOX_REL), move("0 Inbox/mac/from the mac.md"), move(MEMORY_REL)]}
    raw["proposals"][2]["move_to"] = "2 Areas/Garden/"
    own, other, outside = propose.validate(raw, b, vault, collect.existing_notes(vault), DAY, inbox="win")[0]
    assert own["action"] == "move" and own["move_to"] == "1 Projects/Garden/"
    assert other["action"] == "flag" and other["target"] == "0 Inbox/mac/from the mac.md"
    assert "move_to" not in other and other["reason"] == "in the mac inbox; move it from the Mac"
    assert "only the Mac writes there" in other["text"]
    assert outside["action"] == "move" and outside["target"] == MEMORY_REL

    [mac_own, mac_other, _] = propose.validate(raw, b, vault, collect.existing_notes(vault), DAY, inbox="mac")[0]
    assert mac_own["action"] == "flag" and mac_other["action"] == "move"
    assert mac_own["reason"] == "in the win inbox; move it from the PC"


def test_prompt_names_this_computers_inbox(vault):
    b = batch_for(vault)
    on_pc = propose.build_prompt(b, [], [], DAY, inbox="win")
    assert "under 0 Inbox/win/" in on_pc and "(0 Inbox/mac/) can't be moved from here" in on_pc
    on_mac = propose.build_prompt(b, [], [], DAY, inbox="mac")
    assert "under 0 Inbox/mac/" in on_mac and "(0 Inbox/win/) can't be moved from here" in on_mac
