"""dream/reflect.py against made-up chat notes and a fake model. No real claude call, no real vault.
Every chat, quote and claim here is made up."""
import datetime as _dt
import json
import re

import pytest

from capture.chatnote import Chat, Turn, write_chat
from common import notes
from common.notes import read_note, render
from dream import filter as filt
from dream import guards, propose, reflect
from tests.dream.conftest import write

DAY = "2026-11-19"
NOW = _dt.datetime(2026, 11, 19, 11, 0)
QUIET = lambda m: None


@pytest.fixture(autouse=True)
def no_real_claude(monkeypatch):
    def refuse(*a, **k):
        pytest.fail("a test tried to run the real claude CLI")
    monkeypatch.setattr(propose, "subprocess_runner", refuse)


def chat(cfg, cid, title, turns, created):
    t = [Turn(p, a, created, quote=q) for p, a, q in turns]
    path, _ = write_chat(cfg, Chat("chatgpt", cid, title, "https://x.invalid", created, created, t, "http-api"),
                         {}, today=created)
    return path


def score(cfg, path, n=8, whose="user"):
    pstate = filt.profile_state(cfg.state, cfg.vault, cfg.vault)
    scores = filt.load_scores(pstate)
    rel = path.relative_to(cfg.vault).as_posix()
    meta, body = read_note(path)
    scores[rel] = {"score": n, "whose": whose, "reason": "made up", "hash": filt.chat_hash(meta, body),
                   "chat_id": meta["chat_id"],
                   "date": meta["created"], "month": meta["created"][:7]}
    filt.save_scores(pstate, scores)


@pytest.fixture
def months(cfg):
    a = chat(cfg, "c-a", "garden beds", [
        ("I water the tomatoes every morning before work", "Watering in the morning is good.", None),
        ("please always answer me in metric units from now on", "Sure, metric it is.", None),
    ], "2025-07-03")
    b = chat(cfg, "c-b", "seed order", [
        ("I water the beans every morning too, same routine", "Great routine. You are very disciplined.", None),
        ("ok I will try the yellow ones this year", "Yellow ones are lovely.",
         "You are clearly a very patient gardener who loves plants"),
    ], "2025-07-20")
    c = chat(cfg, "c-c", "shed paint", [("I painted the shed green last weekend with my dad", "Nice.", None)],
             "2025-08-02")
    d = chat(cfg, "c-d", "lawn", [("I never water anything in the morning, only at night", "Ok.", None)],
             "2025-08-15")
    for p in (a, b, c, d):
        score(cfg, p)
    return {"a": a, "b": b, "c": c, "d": d}


class Model:
    """A fake `claude -p`. Answers by the schema it was handed: extract (claims), fact-check
    (verdicts), gist (questions), reword (items)."""

    def __init__(self, claims=None, verdicts=None, questions=None, cost=0.05):
        self.claims = claims or {}
        self.verdicts = verdicts
        self.questions = questions
        self.cost = cost
        self.calls = []

    def __call__(self, args, stdin_text, cwd):
        schema = json.loads(args[args.index("--json-schema") + 1])
        kind = next(iter(schema["properties"]))
        self.calls.append((kind, args, stdin_text))
        if kind == "claims":
            month = re.search(r"# Chats from (\d{4}-\d{2})", stdin_text).group(1)
            out = {"claims": self.claims.get(month, [])}
        elif kind == "verdicts":
            ids = re.findall(r"^- (K\d+) ", stdin_text, re.M)
            out = {"verdicts": [{"id": k, "status": (self.verdicts or {}).get(k, "supported"),
                                 "evidence": "the turn says so", "source": "C1"} for k in ids]}
        elif kind == "questions":
            out = {"questions": self.questions if self.questions is not None else []}
        else:
            out = {"items": []}
        return {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": self.cost,
                "modelUsage": {"claude-sonnet-5-5": {}}, "structured_output": out}

    def prompts(self, kind):
        return [c[2] for c in self.calls if c[0] == kind]


def fact(text, *quotes, kind="said", theme="taste"):
    return {"kind": kind, "theme": theme, "text": text, "quotes": [{"chat": c, "quote": q} for c, q in quotes]}


JULY = [
    fact("The user waters the tomatoes every morning before work.",
         ("C1", "I water the tomatoes every morning before work")),
    fact("The user wants answers in metric units.", ("C1", "please always answer me in METRIC units"), kind="told"),
    fact("The user is a patient gardener.", ("C2", "You are clearly a very patient gardener")),       # quoted the AI
    fact("The user is disciplined.", ("C2", "Great routine. You are very disciplined")),               # the AI's answer
    fact("The user grows corn.", ("C1", "I grow corn on the north side of the yard")),                 # invented
    fact("The user waters.", ("C1", "I water")),                                                       # too short
    fact("The user waters things.", ("C9", "I water the tomatoes every morning")),                     # no such chat
    {"kind": "interpretation", "theme": "life", "text": "Waters plants in the morning in C1 and C2.",
     "quotes": [{"chat": "C1", "quote": "I water the tomatoes every morning"},
                {"chat": "C2", "quote": "I water the beans every morning too"}]},
    {"kind": "interpretation", "theme": "life", "text": "Waters in the morning, from one chat.",
     "quotes": [{"chat": "C1", "quote": "I water the tomatoes every morning"}]},                    # 1 chat
    {"kind": "interpretation", "theme": "life", "text": "Feels calm when gardening in C1 and C2.",
     "quotes": [{"chat": "C1", "quote": "I water the tomatoes every morning"},
                {"chat": "C2", "quote": "I water the beans every morning too"}]},                   # a feeling
]


def run(cfg, model, out, months=("2025-07",), **kw):
    return reflect.run(cfg, months=list(months), out=out, cap_usd=3.0, runner=model, now=NOW, log=QUIET, **kw)


def test_the_quote_check_keeps_real_quotes_and_drops_the_rest(cfg, months, tmp_path):
    model = Model({"2025-07": JULY})
    assert run(cfg, model, tmp_path / "out") == 0
    rec = guards.load_json(filt.profile_state(cfg.state, cfg.vault, cfg.vault) / "reflect" / f"run {DAY}.json", {})
    kept = {c["text"] for c in rec["claims"]}
    assert "The user waters the tomatoes every morning before work." in kept
    assert "The user wants answers in metric units." in kept        # case folding only
    # chat keys mean nothing outside the call, so they become links
    assert "Waters plants in the morning in [[2025-07-03 garden beds (c-a)]] and [[2025-07-20 seed order (c-b)]]." in kept
    dropped = {c["text"]: c["dropped"] for c in rec["dropped"]}
    assert dropped["The user is a patient gardener."].startswith("quote check")      # a > line is the AI's
    assert dropped["The user is disciplined."].startswith("quote check")
    assert dropped["The user grows corn."].startswith("quote check")
    assert dropped["The user waters."] == "quote check: quote too short"
    assert dropped["The user waters things."].startswith("quote check")
    assert dropped["Waters in the morning, from one chat."] == "needs quotes from 2 or more chats"
    assert dropped["Feels calm when gardening in C1 and C2."].startswith("names a motive")
    assert rec["stats"]["dropped by quote check"] == 5
    assert rec["stats"]["dropped by rules"] == 2


def test_find_quote_rules():
    c = reflect.ChatDoc("r", "s", "t", "2025-07-01", "chatgpt", "private", 5,
                        [reflect.Turn(1, "2025-07-01", "I said “no thanks” to the  offer, then I left early", "")])
    assert c.turns and reflect.find_quote('"I said "no thanks" to the offer"', c)[0]
    assert reflect.find_quote("I said ... to the offer, then I left", c)[0] is None       # a 2-word piece
    assert reflect.find_quote("I said “no thanks” ... then I left early.", c)[0]
    assert reflect.find_quote("then I left early ... I said no thanks", c)[0] is None     # out of order
    assert reflect.find_quote("I said no to the offer", c)[0] is None                     # changed words


def test_one_bad_quote_drops_the_whole_claim(cfg, months, tmp_path):
    claim = fact("The user waters every morning.", ("C1", "I water the tomatoes every morning"),
                 ("C2", "I water the cucumbers every morning"))
    model = Model({"2025-07": [claim]})
    run(cfg, model, tmp_path / "out")
    assert model.prompts("verdicts") == []


def test_the_fact_checker_is_cold_and_its_verdicts_count(cfg, months, tmp_path):
    model = Model({"2025-07": JULY}, verdicts={"K2": "unsupported"})
    run(cfg, model, tmp_path / "out")
    check = model.prompts("verdicts")[0]
    assert "Kinds of claim" not in check            # never the extractor's instructions
    assert "Watering in the morning is good." not in check   # never the AI's answers
    assert "I water the tomatoes every morning before work" in check
    rec = guards.load_json(filt.profile_state(cfg.state, cfg.vault, cfg.vault) / "reflect" / f"run {DAY}.json", {})
    by_text = {c["text"]: c for c in rec["claims"]}
    assert by_text["The user wants answers in metric units."]["status"] == "unsupported"
    digest_text = (tmp_path / "out" / "spike facts digest.md").read_text(encoding="utf-8")
    assert "metric" not in digest_text.lower()
    assert "- [ ] The user waters the tomatoes every morning before work." in digest_text


def test_a_claim_with_no_verdict_is_unsupported():
    c = reflect.Claim("m#01", "2025-07", "said", "taste", "x")
    counts = reflect.apply_verdicts({"verdicts": [{"id": "K9", "status": "supported", "evidence": "", "source": ""}]},
                                    {"K1": c})
    assert c.status == "unsupported" and c.dropped and counts["no verdict"] == 1


def test_the_diary_has_the_five_headings_and_links(cfg, months, tmp_path):
    run(cfg, Model({"2025-07": JULY}), tmp_path / "out")
    meta, body = read_note(tmp_path / "out" / "spike diary 2025-07.md")
    assert meta["visibility"] == "private" and meta["date"] == "2025-07-31"
    for _, head in reflect.HEADINGS:
        assert f"## {head}" in body
    assert "[fact] The user waters the tomatoes" in body
    assert "[interpretation, low confidence: 2 chats]" in body
    assert "[[2025-07-03 garden beds (c-a)]]" in body
    assert "patient gardener" not in body and "disciplined" not in body
    assert reflect.check_written(body) == []


def test_replay_passes_earlier_facts_but_not_interpretations(cfg, months, tmp_path):
    aug = [fact("The user painted the shed green.", ("C1", "I painted the shed green last weekend"), kind="did")]
    model = Model({"2025-07": JULY, "2025-08": aug})
    run(cfg, model, tmp_path / "out", months=("2025-07", "2025-08"))
    july, aug_prompt = model.prompts("claims")
    assert "Today is 2025-07-31." in july and "Today is 2025-08-31." in aug_prompt
    noted = aug_prompt.split("# Already noted")[1].split("# Chats from")[0]
    assert "The user waters the tomatoes every morning before work. (2025-07-03)" in noted
    assert "Waters plants in the morning" not in noted
    assert "I water the tomatoes" not in aug_prompt.split("# Chats from")[1]   # July's chats stay in July
    aug_diary = (tmp_path / "out" / "spike diary 2025-08.md").read_text(encoding="utf-8")
    observed = aug_diary.split("## What Bonsai observed")[1].split("## What you explicitly told it")[0]
    assert "[observed] The user painted the shed green." in observed
    assert "shed" not in (tmp_path / "out" / "spike facts digest.md").read_text(encoding="utf-8")


def test_contradictions_need_two_chats_and_go_first_in_the_gist(cfg, months, tmp_path):
    aug = [{"kind": "contradiction", "theme": "life", "text": "The user described two different watering times.",
            "quotes": [{"chat": "C1", "quote": "I painted the shed green last weekend"},
                       {"chat": "C2", "quote": "I never water anything in the morning"}]}]
    questions = [
        {"kind": "interesting", "based_on": ["G3"], "text": 'You said "I water the tomatoes every morning". Is that every day?'},
        {"kind": "contradiction", "based_on": ["G1"],
         "text": 'You said "I never water anything in the morning" in one chat. Did your routine change?'},
        {"kind": "working", "based_on": ["G2"], "text": 'You said "please always answer me in metric units". Does that still hold?'},
        {"kind": "interesting", "based_on": ["G3"], "text": 'You said "I love my garden so much". Is that right?'},
        {"kind": "interesting", "based_on": ["G3"], "text": 'You said "I water the tomatoes every morning". How do you feel about it?'},
        {"kind": "interesting", "based_on": ["G3"], "text": 'You said "I water the tomatoes every morning", don\'t you agree it helps?'},
    ]
    model = Model({"2025-07": JULY, "2025-08": aug}, questions=questions)
    run(cfg, model, tmp_path / "out", months=("2025-07", "2025-08"))
    gist = (tmp_path / "out" / "spike gist.md").read_text(encoding="utf-8")
    qs = re.findall(r"^\d\. (.*)$", gist, re.M)
    assert len(qs) == 3
    assert "never water anything" in qs[0]          # the contradiction first
    assert "metric" in qs[1]
    assert "love my garden" not in gist             # a quote nobody checked
    assert "feel" not in gist.split("## Questions")[1].lower()
    assert "agree" not in gist
    assert "[[" in qs[0]
    aug_diary = (tmp_path / "out" / "spike diary 2025-08.md").read_text(encoding="utf-8")
    assert aug_diary.rstrip().split("## Questions")[1].count("never water anything") == 1   # entries end in questions
    assert "metric" not in aug_diary.split("## Questions")[1]                           # that one is July's


def test_outputs_are_new_files_only_and_never_under_profile(cfg, months, tmp_path):
    out = tmp_path / "out"
    assert run(cfg, Model({"2025-07": JULY}), out) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["README.md", "spike claims to label.md", "spike diary 2025-07.md", "spike facts digest.md",
                     "spike gist.md"]
    before = {p.name: p.read_text(encoding="utf-8") for p in out.iterdir()}
    assert run(cfg, Model({"2025-07": JULY}), out) == 2           # refuses to overwrite
    assert {p.name: p.read_text(encoding="utf-8") for p in out.iterdir()} == before
    model = Model({"2025-07": JULY})
    assert run(cfg, model, cfg.vault / "2 Areas/Profile/diary") == 2
    assert model.calls == []
    assert not (cfg.vault / "2 Areas/Profile/diary").exists()


def test_every_output_passes_the_writing_bans(cfg, months, tmp_path):
    run(cfg, Model({"2025-07": JULY}), tmp_path / "out")
    for p in (tmp_path / "out").iterdir():
        _, body = read_note(p)
        assert reflect.check_written(body) == [], p.name


def test_a_rerun_uses_the_cache(cfg, months, tmp_path):
    run(cfg, Model({"2025-07": JULY}), tmp_path / "one")
    model = Model({"2025-07": JULY})
    assert run(cfg, model, tmp_path / "two") == 0
    assert model.calls == []
    assert (tmp_path / "two" / "spike diary 2025-07.md").read_text(encoding="utf-8") == \
        (tmp_path / "one" / "spike diary 2025-07.md").read_text(encoding="utf-8")


def test_spend_goes_on_the_profile_line_and_calls_are_locked_down(cfg, months, tmp_path):
    model = Model({"2025-07": JULY}, cost=0.1)
    run(cfg, model, tmp_path / "out")
    spend = guards.load_json(guards.dream_root(cfg.state) / "spend.json", {})
    assert list(spend) == [f"{DAY} profile"] and spend[f"{DAY} profile"] == pytest.approx(0.1 * len(model.calls))
    for _, args, _ in model.calls:
        assert args[args.index("--tools") + 1] == ""
        assert "--strict-mcp-config" in args and "--no-session-persistence" in args
        assert args[args.index("--model") + 1] == propose.MODEL


def test_the_cap_stops_the_run(cfg, months, tmp_path):
    guards.Budget(cfg.state, filt.spend_key(DAY), 1.0).add(1.0)
    model = Model({"2025-07": JULY})
    assert reflect.run(cfg, months=["2025-07"], out=tmp_path / "out", cap_usd=1.0, runner=model, now=NOW,
                       log=QUIET) == 1
    assert model.calls == []


def test_rejected_claims_are_dropped(cfg, months, tmp_path):
    write(cfg.vault, reflect.REJECTED_REL, render({"name": "rejected", "source": "author"},
                                                  "- The user wants answers in metric units.\n"))
    model = Model({"2025-07": JULY})
    run(cfg, model, tmp_path / "out")
    rec = guards.load_json(filt.profile_state(cfg.state, cfg.vault, cfg.vault) / "reflect" / f"run {DAY}.json", {})
    assert any(c["dropped"].startswith("matches the rejected list") for c in rec["dropped"])
    assert "The user wants answers in metric units." in model.prompts("claims")[0]   # the prompt lists it too


def test_other_peoples_and_low_scored_chats_are_never_read(cfg, months, tmp_path):
    p = chat(cfg, "c-x", "riley bike chain", [("riley here, my chain slips on every hill ok", "Sure.", None)], "2025-07-09")
    score(cfg, p, n=9, whose="other")
    q = chat(cfg, "c-y", "trivia", [("what is the tallest tree in the world", "A redwood.", None)], "2025-07-10")
    score(cfg, q, n=1)
    model = Model({"2025-07": []})
    run(cfg, model, tmp_path / "out")
    prompt = model.prompts("claims")[0]
    assert "chain slips on every hill" not in prompt and "tallest tree" not in prompt
    assert "I water the tomatoes" in prompt


def test_the_readme_has_the_counts(cfg, months, tmp_path):
    run(cfg, Model({"2025-07": JULY}), tmp_path / "out")
    _, body = read_note(tmp_path / "out" / "README.md")
    assert "Claims extracted: 10" in body
    assert "Dropped by the quote check: 5 claims" in body
    assert "Fact-checker verdicts: supported 3" in body
    assert "$" in body.split("## Cost")[1]


def test_a_chats_later_turns_go_to_their_own_month(cfg, months, tmp_path):
    turns = [Turn("I started the compost bin this week for the garden", "Nice.", "2025-07-25"),
             Turn("the compost bin is finally ready to use in the beds", "Great.", "2025-08-20")]
    p, _ = write_chat(cfg, Chat("chatgpt", "c-long", "compost", "https://x.invalid", "2025-07-25", "2025-08-20", turns,
                                "http-api"), {}, today="2025-08-20")
    score(cfg, p)
    late = [fact("The user's compost was ready.", ("C1", "the compost bin is finally ready to use"))]
    model = Model({"2025-07": late, "2025-08": late})
    run(cfg, model, tmp_path / "out", months=("2025-07", "2025-08"))
    july, aug = model.prompts("claims")
    assert "I started the compost bin" in july and "finally ready" not in july
    assert "finally ready" in aug and "I started the compost bin" not in aug
    rec = guards.load_json(filt.profile_state(cfg.state, cfg.vault, cfg.vault) / "reflect" / f"run {DAY}.json", {})
    assert [c["month"] for c in rec["claims"] if c["text"] == "The user's compost was ready."] == ["2025-08"]


def test_a_checker_note_that_breaks_the_writing_rules_is_left_out():
    c = reflect.Claim("m#01", "2025-07", "said", "taste", "x")
    reflect.apply_verdicts({"verdicts": [{"id": "K1", "status": "supported", "source": "C1",
                                          "evidence": "they said they had only today to prepare"}]}, {"K1": c})
    assert c.status == "supported" and "today" not in c.evidence and "today" in c.checker_note


def test_the_readme_counts_cached_costs_and_the_filter(cfg, months, tmp_path):
    pstate = filt.profile_state(cfg.state, cfg.vault, cfg.vault)
    guards.save_json(pstate / filt.FILTER_RUNS, [{"cost": 0.5}, {"cost": 0.25}])
    run(cfg, Model({"2025-07": JULY}, cost=0.1), tmp_path / "one")
    run(cfg, Model({"2025-07": JULY}), tmp_path / "two")         # all from the cache
    _, body = read_note(tmp_path / "two" / "README.md")
    assert "- extract: $0.1000" in body and "- filter (every run on this copy): $0.7500" in body


def test_the_reword_pass_is_cached_and_holds_what_still_breaks(cfg, months, tmp_path):
    hedged = [fact("The user probably waters the tomatoes every morning.",
                   ("C1", "I water the tomatoes every morning before work"))]

    class Rewords(Model):
        def __call__(self, args, stdin_text, cwd):
            r = super().__call__(args, stdin_text, cwd)
            if "Rewrite these" in stdin_text:
                r["structured_output"] = {"items": [{"id": "0", "reason": "", "description": "",
                                                     "text": "The user waters the tomatoes every morning."}]}
            return r
    model = Rewords({"2025-07": hedged})
    run(cfg, model, tmp_path / "one")
    assert "- [ ] The user waters the tomatoes every morning." in \
        (tmp_path / "one" / "spike facts digest.md").read_text(encoding="utf-8")
    again = Rewords({"2025-07": hedged})
    run(cfg, again, tmp_path / "two")
    assert again.calls == []
