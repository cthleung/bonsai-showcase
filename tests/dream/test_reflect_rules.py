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


# --- single-quoted spans in questions are checked too ---------------------------------------

def _ask(text, quote="I don't like long answers at all"):
    return reflect.check_question({"kind": "working", "based_on": ["G1"], "text": text}, {"G1": told(quote)})


def test_a_single_quoted_span_must_be_a_checked_quote():
    q, why = _ask("You said \"I don't like long answers at all\" and 'I never read the docs'. Is that still true?")
    assert q is None and why == "a quote isn't one of the checked quotes"


def test_a_checked_single_quoted_span_passes_and_apostrophes_are_not_spans():
    q, why = _ask("You said 'I don't like long answers at all'. Is that still true?")
    assert q and why == ""
    # the apostrophes in don't, it's and the user's are not quote marks
    q, why = _ask("You said \"I don't like long answers at all\". Is it still the user's rule, or isn't that so?")
    assert q and why == ""


def test_words_inside_a_checked_single_quote_skip_the_word_checks():
    q, why = _ask("You said 'I really love long answers here'. Is that still true?",
                  quote="I really love long answers here")
    assert q and why == ""


# --- the leading-question ban --------------------------------------------------------------

@pytest.mark.parametrize("text", [
    'You said "I don\'t like long answers at all", right?',
    'You said "I don\'t like long answers at all", right ? Yes',
    'Don’t you still think "I don\'t like long answers at all"?',
    'Don\'t you still think "I don\'t like long answers at all"?',
])
def test_leading_questions_are_dropped(text):
    if not text.endswith("?"):
        text += "?"
    q, why = _ask(text)
    assert q is None and why.startswith("leading:")


def test_alright_and_a_plain_question_pass():
    assert _ask('You said "I don\'t like long answers at all". Is that still alright?')[0]
    assert _ask('You said "I don\'t like long answers at all". Does that still hold?')[0]


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


# --- the feeling ban covers interpretations, unknowns and contradictions only ---------------

@pytest.mark.parametrize("text, said", [
    ("The user asked for a stress test of the API.", "please write a stress test for the API we built"),
    ("The user asked how Rust traits work.", "how do Rust traits work with generic functions"),
    ("The user asked to sum the values in a column.", "how do I sum the values in a column here"),
    ("The user built an identity provider.", "I built an identity provider for the app last week"),
    ("The user asked to fix a deeply nested loop.", "can you fix this deeply nested loop in my code"),
])
def test_a_did_claim_keeps_ordinary_words(text, said):
    kept, dropped, _ = reflect.check_claims({"claims": [fact(text, ("C1", said), kind="did")]}, "2025-07",
                                            [doc(said)], [], [])
    assert [k.text for k in kept] == [text] and dropped == []


@pytest.mark.parametrize("kind", ["interpretation", "contradiction", "unknown"])
def test_an_interpretation_naming_a_feeling_is_still_dropped(kind):
    a, b = doc("I water the tomatoes every morning before work"), doc("I water the beans every morning too", "b.md",
                                                                       key="C2")
    claim = {"kind": kind, "theme": "life", "text": "The user is stressed about watering in two chats.",
             "quotes": [{"chat": "C1", "quote": "I water the tomatoes every morning"},
                        {"chat": "C2", "quote": "I water the beans every morning"}]}
    kept, dropped, _ = reflect.check_claims({"claims": [claim]}, "2025-07", [a, b], [], [])
    assert kept == [] and dropped[0].dropped == "names a motive, feeling or trait: 'stressed'"


# --- a reworded claim goes through the claim rules again -------------------------------------

class Rewords(Model):
    """Rewrites each claim sent to the reword call with the text in `new`, by its position."""

    def __init__(self, claims, new, **kw):
        super().__init__(claims, **kw)
        self.new = new

    def __call__(self, args, stdin_text, cwd):
        r = super().__call__(args, stdin_text, cwd)
        if "Rewrite these" in stdin_text:
            r["structured_output"] = {"items": [{"id": str(i), "reason": "", "description": "", "text": t}
                                                for i, t in enumerate(self.new)]}
        return r


TWO_CHATS = [{"chat": "C1", "quote": "I water the tomatoes every morning"},
             {"chat": "C2", "quote": "I water the beans every morning too"}]


def test_a_reword_that_adds_a_feeling_is_dropped(cfg, months, tmp_path):
    claims = [{"kind": "interpretation", "theme": "life", "quotes": TWO_CHATS,
               "text": "The user probably waters plants in the morning in two chats."}]
    model = Rewords({"2025-07": claims}, ["The user loves watering plants in the morning, in two chats."])
    assert run(cfg, model, tmp_path / "out") == 0
    assert model.prompts("verdicts") == []                   # never reached the fact-checker
    rec = record(cfg)
    assert [c["dropped"] for c in rec["dropped"]] == ["names a motive, feeling or trait: 'loves'"]
    assert rec["stats"]["dropped by rules"] == 1 and rec["stats"].get("held by writing rules", 0) == 0
    assert "loves" not in (tmp_path / "out" / "spike diary 2025-07.md").read_text(encoding="utf-8")


def test_a_reword_that_matches_the_rejected_list_is_dropped(cfg, months, tmp_path):
    write(cfg.vault, reflect.REJECTED_REL, render({"name": "rejected", "source": "author"},
                                                  "- The user wants answers in metric units.\n"))
    claims = [fact("The user maybe likes the metric system for replies.",
                   ("C1", "please always answer me in metric units"), kind="told")]
    model = Rewords({"2025-07": claims}, ["The user wants answers in metric units."])
    run(cfg, model, tmp_path / "out")
    assert model.prompts("verdicts") == []
    assert record(cfg)["dropped"][0]["dropped"].startswith("matches the rejected list")


def test_rule_drop_is_the_one_rule_check():
    c = Claim("m#01", "2025-07", "did", "code", "The user asked for a stress test.")
    assert reflect.rule_drop(c, [], []) is None
    c.kind = "interpretation"
    assert reflect.rule_drop(c, [], []).startswith("names a motive")
    c.kind, c.text = "said", "The user likes the metric system."
    assert reflect.rule_drop(c, ["The user likes the metric system."], []).startswith("matches the rejected list")


# --- links and the user's own checked quotes skip the writing bans ---------------------------

QUOTES = [{"quote": "I will look at the bug maybe later tonight"}]


def test_links_and_checked_quotes_are_left_out_of_the_ban_check():
    assert reflect.claim_hits("The user asked for a recipe in [[2025-07-20 What to cook today (ab12cd34)]].", []) == []
    assert reflect.claim_hits('The user called it "maybe later" in a chat.', QUOTES) == []
    assert reflect.claim_hits("The user called it 'maybe later' in a chat.", QUOTES) == []
    # a quoted span nobody checked still counts, and so do the claim's own words
    assert reflect.claim_hits('The user called it "maybe tomorrow" in a chat.', QUOTES)
    assert reflect.claim_hits('The user maybe called it "maybe later" in a chat.', QUOTES)
    assert reflect.claim_hits('The user called it "like the bug maybe" in a chat.', QUOTES)   # not on word edges


@pytest.fixture
def cook(cfg, months):
    p = chat(cfg, "c-r", "what to cook today", [("please give me a dinner recipe with the beans", "Sure.", None)],
             "2025-07-25")
    score(cfg, p)
    return "[[" + p.stem + "]]"


def test_a_claim_with_a_link_is_not_reworded(cfg, cook, tmp_path):
    model = Model({"2025-07": [fact("The user asked for a dinner recipe in C3.",
                                    ("C3", "please give me a dinner recipe with the beans"), kind="did")]})
    assert run(cfg, model, tmp_path / "out") == 0
    assert model.prompts("items") == []
    assert [c["text"] for c in record(cfg)["claims"]] == [f"The user asked for a dinner recipe in {cook}."]


def test_a_reword_must_leave_links_alone(cfg, cook, tmp_path):
    claims = [fact("The user probably asked for a dinner recipe in C3.",
                   ("C3", "please give me a dinner recipe with the beans"), kind="did"),
              fact("The user probably asked for a recipe with beans in C3.",
                   ("C3", "please give me a dinner recipe with the beans"), kind="did")]
    model = Rewords({"2025-07": claims}, ["The user asked for a dinner recipe.",
                                          f"The user asked for a recipe with beans in {cook}."])
    run(cfg, model, tmp_path / "out")
    assert cook in model.prompts("items")[0]
    rec = record(cfg)
    assert [c["text"] for c in rec["claims"]] == [f"The user asked for a recipe with beans in {cook}."]
    assert [c["dropped"] for c in rec["dropped"]] == ["writing rules: the reword changed a link"]


# --- a failed run writes no notes ----------------------------------------------------------

FAILED = {"type": "result", "subtype": "error_max_budget_usd", "is_error": True, "total_cost_usd": 0.01}


class FailsOn(Model):
    """Fails every call whose prompt holds `marker`."""

    def __init__(self, claims, marker, **kw):
        super().__init__(claims, **kw)
        self.marker = marker

    def __call__(self, args, stdin_text, cwd):
        r = super().__call__(args, stdin_text, cwd)
        return dict(FAILED) if self.marker in stdin_text else r


def test_a_failed_extract_writes_no_notes_and_a_rerun_uses_the_cache(cfg, months, tmp_path):
    out = tmp_path / "out"
    model = FailsOn({"2025-07": JULY[:2]}, "# Chats from 2025-08")
    assert run(cfg, model, out, months=("2025-07", "2025-08")) == 1
    assert not out.exists() or list(out.iterdir()) == []
    rec = record(cfg)
    assert rec["error"].startswith("2025-08: ") and len(rec["claims"]) == 2     # the record is saved
    again = Model({"2025-07": JULY[:2]})
    assert run(cfg, again, out, months=("2025-07", "2025-08")) == 0
    assert [k for k, _, _ in again.calls] == ["claims", "questions"]   # July comes from the cache
    assert "# Chats from 2025-08" in again.calls[0][2]
    assert "Nothing in 2025-08." in (out / "spike diary 2025-08.md").read_text(encoding="utf-8")


def test_a_failed_reword_call_fails_the_run(cfg, months, tmp_path):
    hedged = [fact("The user probably waters the tomatoes every morning.",
                   ("C1", "I water the tomatoes every morning before work")),
              fact("The user wants answers in metric units.", ("C1", "please always answer me in metric units"),
                   kind="told")]
    model = FailsOn({"2025-07": hedged}, "Rewrite these")
    assert run(cfg, model, tmp_path / "out") == 1
    assert not (tmp_path / "out").exists()
    assert model.prompts("verdicts") == []
    rec = record(cfg)
    assert "the reword call failed" in rec["error"]
    assert rec["stats"].get("held by writing rules", 0) == 0
    assert all(not c["dropped"].startswith("writing rules") for c in rec["dropped"])


# --- bad month ranges are refused before any call ------------------------------------------

@pytest.mark.parametrize("first, last", [("2025-09", "2025-07"), ("2025-13", "2025-13"), ("2025-00", "2025-01"),
                                         ("2025-11", "2025-13"), ("2025-7", "2025-08"), ("2025-07", None)])
def test_main_refuses_bad_months_before_anything_runs(monkeypatch, capsys, first, last):
    monkeypatch.setattr(reflect, "run", lambda *a, **k: pytest.fail("reflect.run was reached"))
    monkeypatch.setattr(reflect.config, "load", lambda *a, **k: pytest.fail("the config was loaded"))
    argv = ["--from", first, "--cap-usd", "1"] + (["--to", last] if last else [])
    with pytest.raises(SystemExit) as e:
        reflect.main(argv)
    assert e.value.code == 2 and "--from" in capsys.readouterr().err


def test_main_takes_a_good_range(monkeypatch):
    seen = {}
    monkeypatch.setattr(reflect.config, "load", lambda *a, **k: object())
    monkeypatch.setattr(reflect, "run", lambda cfg, vault, months, *a, **k: seen.setdefault("months", months) and 0)
    assert reflect.main(["--from", "2025-11", "--to", "2026-01", "--cap-usd", "1"]) == 0
    assert seen["months"] == ["2025-11", "2025-12", "2026-01"]


@pytest.mark.parametrize("bad", [[], ["2025-13"], ["2025-00"], ["2025-7"]])
def test_run_refuses_an_empty_or_bad_month_list_with_no_calls(cfg, months, tmp_path, bad):
    model = Model({"2025-07": JULY})
    assert reflect.run(cfg, months=bad, out=tmp_path / "out", cap_usd=3.0, runner=model, now=NOW, log=QUIET) == 2
    assert model.calls == [] and not (tmp_path / "out").exists()


def test_no_months_still_means_pick(cfg, months, tmp_path):
    model = Model({"2025-07": JULY})
    assert reflect.run(cfg, months=None, pick=1, out=tmp_path / "out", cap_usd=3.0, runner=model, now=NOW,
                       log=QUIET) == 0
    assert [k for k, _, _ in model.calls][:1] == ["claims"]


# --- chats that changed after scoring are skipped --------------------------------------------

def _stale_run(cfg, tmp_path, name):
    model = Model({"2025-07": []})
    assert run(cfg, model, tmp_path / name) == 0
    _, readme = read_note(tmp_path / name / "README.md")
    return model.prompts("claims")[0], readme


def test_the_score_helper_stores_the_filters_own_hash(cfg, months):
    kept, _ = filt.kept_chats(cfg.vault, [], QUIET)
    scores = filt.load_scores(filt.profile_state(cfg.state, cfg.vault, cfg.vault))
    rels = {m.relative_to(cfg.vault).as_posix() for m in months.values()}
    assert rels <= {c.rel for c in kept}
    assert not rels & {c.rel for c in filt.to_score(kept, scores)}


def test_a_chat_edited_after_scoring_is_skipped_and_counted(cfg, months, tmp_path):
    p = months["a"]
    p.write_text(p.read_text(encoding="utf-8").replace("before work", "before work, rain or shine"),
                 encoding="utf-8")
    prompt, readme = _stale_run(cfg, tmp_path, "one")
    assert "I water the tomatoes" not in prompt and "I water the beans" in prompt
    assert "Skipped because they changed after the filter scored them: 1 " in readme
    assert record(cfg)["stale_chats"] == 1
    score(cfg, p)                                   # rescored: read again
    prompt, readme = _stale_run(cfg, tmp_path, "two")
    assert "rain or shine" in prompt
    assert "Skipped because they changed after the filter scored them: 0 " in readme


def test_a_chat_never_scored_is_not_counted_as_stale(cfg, months, tmp_path):
    chat(cfg, "c-n", "new one", [("I planted some garlic in the back bed today", "Nice.", None)], "2025-07-28")
    prompt, readme = _stale_run(cfg, tmp_path, "out")
    assert "garlic" not in prompt
    assert "Skipped because they changed after the filter scored them: 0 " in readme


def test_a_reword_keeps_a_link_whose_chat_name_holds_a_dash(cfg, months, tmp_path):
    p = chat(cfg, "c-d2", "Rust \u2014 the hard parts", [("how do lifetimes work when a struct holds a reference",
                                                         "Like this.", None)], "2025-07-25")
    score(cfg, p)
    link = "[[" + p.stem + "]]"
    assert "\u2014" in link
    quote = ("C3", "how do lifetimes work when a struct holds a reference")
    model = Rewords({"2025-07": [fact("The user probably asked about lifetimes in C3.", quote, kind="did")]},
                    [f"The user asked about Rust lifetimes in {link}."])
    assert run(cfg, model, tmp_path / "out") == 0
    assert [c["text"] for c in record(cfg)["claims"]] == [f"The user asked about Rust lifetimes in {link}."]


def test_links_are_left_out_of_the_rules_only_after_name_chats():
    c = Claim("m#01", "2025-07", "interpretation", "code", "The user asked about it in [[2025-07-02 I hate mondays (ab)]].")
    assert reflect.rule_drop(c, [], [], links=True) is None             # the script's link to a chat
    assert reflect.rule_drop(c, [], []).startswith("names a motive")    # first pass: the extractor's words
