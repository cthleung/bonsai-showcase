"""dream/filter.py and the profile runs: excluded topics, the sorted marker, scores loaded under the
lock, the spend ledger lock, and keys scrubbed before reflect's calls. Every chat, note and key here
is made up."""
import pytest

from common.notes import render
from dream import filter as filt
from dream import propose, reflect
from tests.dream.conftest import FakeClaude, chat_note, chat_rel, write
from tests.dream.test_reflect import Model, NOW, QUIET, chat, score


@pytest.fixture(autouse=True)
def no_real_claude(monkeypatch):
    monkeypatch.setattr(propose, "subprocess_runner", lambda *a, **k: pytest.fail("a test ran the real claude"))


def exclude(cfg, *topics):
    (cfg.state / filt.EXCLUDE_FILE).write_text("\n".join(topics) + "\n", encoding="utf-8")


def reflect_run(cfg, model, out, months=("2025-07",)):
    return reflect.run(cfg, months=list(months), out=out, cap_usd=3.0, runner=model, now=NOW, log=QUIET)


GARDEN = "I water the tomatoes every morning before work"
FACT = {"kind": "said", "theme": "taste", "text": "The user waters the tomatoes every morning before work.",
        "quotes": [{"chat": "C1", "quote": GARDEN}]}


# --- excluded topics -------------------------------------------------------------------

def test_topic_past_the_trim_or_in_an_answer_keeps_the_chat_out(cfg, tmp_path):
    exclude(cfg, "divorce")
    long_turn = GARDEN + ". " + ("blah " * 120) + "also my divorce is final"
    late = chat(cfg, "c-z", "garden", [(long_turn, "Ok.", None)], "2025-07-03")
    answer = chat(cfg, "c-ans", "garden two", [("I water the beans every morning too, same routine",
                                                "Sorry to hear about the divorce, gardening helps.", None)],
                  "2025-07-04")
    wrapped = chat(cfg, "c-w", "garden three", [("I water the peas every morning, my\nDivorce lawyer says hi",
                                                 "Ok.", None)], "2025-07-05")
    clean = chat(cfg, "c-ok", "garden four", [("I water the carrots every morning before the bus", "Ok.", None)],
                 "2025-07-06")
    for p in (late, answer, wrapped, clean):
        score(cfg, p)
    kept, skipped = filt.kept_chats(cfg.vault, ["divorce"], QUIET)
    rels = {c.rel for c in kept}
    assert skipped["excluded topic"] == 3
    assert all(p.relative_to(cfg.vault).as_posix() not in rels for p in (late, answer, wrapped))
    assert clean.relative_to(cfg.vault).as_posix() in rels
    model = Model({"2025-07": []})
    reflect_run(cfg, model, tmp_path / "out")
    prompt = model.prompts("claims")[0]
    assert "I water the carrots every morning" in prompt
    assert "divorce" not in prompt.lower()


def test_a_two_word_topic_wrapped_over_lines_still_counts(cfg):
    write(cfg.vault, chat_rel("wrapped"), chat_note("move", [("we talked about the\n  new   job offer today", "ok")]))
    _, skipped = filt.kept_chats(cfg.vault, ["new job"], QUIET)
    assert skipped["excluded topic"] == 1


def test_redact_topics_drops_the_bullet_paragraph_or_section():
    text = (
        "# Basics\n\n"
        "- Lives in Sapporo\n"
        "- Going through a divorce\n"
        "  - the lawyer is downtown\n"
        "  - court date in March\n"
        "- Likes ramen\n\n"
        "A paragraph that is fine.\n\n"
        "Another paragraph that names the\nDivorce over two lines.\n\n"
        "## Divorce\n\n"
        "Nothing here names it.\n\n"
        "### Lawyer\n\n"
        "- calls on Mondays\n\n"
        "```\n# a comment in code, not a heading\n```\n\n"
        "## Work\n\n"
        "- Writes Python\n"
    )
    out = filt.redact_topics(text, ["divorce"])
    assert "divorce" not in out.lower()
    for gone in ("lawyer is downtown", "court date", "Nothing here names it", "calls on Mondays", "a comment in code"):
        assert gone not in out
    for kept in ("# Basics", "Lives in Sapporo", "Likes ramen", "A paragraph that is fine.", "## Work", "Writes Python"):
        assert kept in out
    assert filt.redact_topics(text, []) == text
    assert filt.redact_topics("- nothing to see\n", ["divorce"]) == "- nothing to see\n"


def test_profile_notes_imports_and_rejected_are_redacted(cfg, tmp_path, monkeypatch):
    exclude(cfg, "divorce")
    score(cfg, chat(cfg, "c-a", "garden beds", [(GARDEN, "Watering in the morning is good.", None)], "2025-07-03"))
    write(cfg.vault, "2 Areas/Profile/basics.md",
          render({"source": "author"}, "- Lives in Sapporo\n- Going through a divorce\n"))
    write(cfg.vault, "2 Areas/Profile/divorce notes.md", render({"source": "author"}, "- the lawyer is downtown\n"))
    write(cfg.vault, "3 Resources/imports/chatgpt memory.md",
          "- User likes ramen\n- User is going through a divorce\n")
    write(cfg.vault, reflect.REJECTED_REL,
          render({"source": "author"}, "- The user hates mornings\n- The user is sad about the divorce\n"))
    seen = {}
    real_check = reflect.check_claims

    def spy(raw, month, chats, rejected, topics, *a, **k):
        seen["rejected"] = list(rejected)
        return real_check(raw, month, chats, rejected, topics, *a, **k)
    monkeypatch.setattr(reflect, "check_claims", spy)
    model = Model({"2025-07": [FACT]})
    assert reflect_run(cfg, model, tmp_path / "out") == 0
    ex, ck = model.prompts("claims")[0], model.prompts("verdicts")[0]
    for prompt in (ex, ck):
        assert "divorce" not in prompt.lower() and "lawyer" not in prompt
        assert "Lives in Sapporo" in prompt and "The user hates mornings" in prompt
    assert "User likes ramen" in ck
    # matches_rejected still gets the whole list
    assert "The user is sad about the divorce" in seen["rejected"]
    # and the filter's prompt
    run = FakeClaude({"chats": []})
    assert filt.run(cfg, cap_usd=1.0, runner=run, now=NOW) == 0
    assert run.calls and all("divorce" not in stdin.lower() for _, stdin, _ in run.calls)
    assert "Lives in Sapporo" in run.calls[0][1]


def test_redaction_comes_before_the_length_cut(cfg, monkeypatch):
    body = "".join(f"- filler line number {i}\n" for i in range(30)) + "- Going through a divorce\n"
    write(cfg.vault, "2 Areas/Profile/basics.md", render({"source": "author"}, body))
    full = reflect.profile_context(cfg.vault)
    # cut in the middle of the topic word, so cutting first would leave "Going through a divo"
    monkeypatch.setattr(reflect, "MAX_CONTEXT_CHARS", full.index("divorce") + 4)
    assert "Going through a divo" in reflect.profile_context(cfg.vault)
    out = reflect.profile_context(cfg.vault, ["divorce"])
    assert "Going through a" not in out and "filler line number 29" in out


def test_vault_context_without_topics_is_unchanged(cfg):
    write(cfg.vault, "2 Areas/Profile/basics.md",
          render({"source": "author"}, "- Lives in Sapporo\n- Going through a divorce\n"))
    plain = filt.vault_context(cfg.vault)
    assert plain == ("## 2 Areas/Profile/basics\n\n- Lives in Sapporo\n- Going through a divorce\n")
    assert filt.vault_context(cfg.vault, None) == plain
    assert "divorce" not in filt.vault_context(cfg.vault, ["divorce"])


# --- the sorted marker -----------------------------------------------------------------

def test_a_sorted_chat_is_never_called_checked_by_the_user(cfg):
    write(cfg.vault, chat_rel("unsure one"), chat_note("school essay", [("my teacher wants it friday", "ok")],
                                                       sorted="2026-11-19", chat_id="cid-unsure"))
    model = FakeClaude({"chats": []})
    assert filt.run(cfg, cap_usd=1.0, runner=model, now=NOW) == 0
    assert model.calls
    for _, stdin, _ in model.calls:
        assert "my teacher wants it friday" in stdin
        assert "checked by the user" not in stdin.lower() and "as theirs" not in stdin


# --- scores load under the lock -------------------------------------------------------

def test_scores_saved_by_another_run_during_the_scan_are_kept(cfg, monkeypatch):
    for n in range(3):
        write(cfg.vault, chat_rel(f"c{n}"), chat_note(f"chat {n}", [(f"I like thing number {n} a lot", "ok")],
                                                       chat_id=f"cid{n}"))
    pstate = filt.profile_state(cfg.state, cfg.vault, cfg.vault)
    real_kept = filt.kept_chats

    def kept_then_other_run_finishes(*a, **k):
        out = real_kept(*a, **k)
        sc = filt.load_scores(pstate)   # run B scores everything while this run scans
        for c in out[0]:
            sc[c.rel] = {"score": 9, "whose": "user", "reason": "B", "hash": c.hash, "month": c.date[:7]}
        filt.save_scores(pstate, sc)
        return out
    monkeypatch.setattr(filt, "kept_chats", kept_then_other_run_finishes)
    model = FakeClaude({"chats": [{"key": "C1", "score": 1, "whose": "user", "reason": "A"}]})
    assert filt.run(cfg, cap_usd=1.0, limit=1, runner=model, now=NOW) == 0
    assert model.calls == []                          # nothing left to pay for
    sc = filt.load_scores(pstate)
    kept, _ = real_kept(cfg.vault, [], QUIET)
    assert len(kept) == 4 and set(sc) == {c.rel for c in kept}
    assert {e["reason"] for e in sc.values()} == {"B"}


# --- the spend ledger lock ------------------------------------------------------------

def test_another_keys_add_between_load_and_save_survives(tmp_path, monkeypatch):
    import threading
    from dream import guards
    s = tmp_path / "s"
    night = guards.Budget(s, "2026-11-19", 5.0)
    prof = guards.Budget(s, filt.spend_key("2026-11-19"), 5.0)
    real_save = guards.save_json
    other = []

    def save_with_a_night_add_racing(path, data):
        if not other:   # the profile add has read the ledger; the night job adds now
            t = threading.Thread(target=night.add, args=(0.30,))
            other.append(t)
            t.start()
            t.join(timeout=0.3)   # without a lock it finishes here, and this save then writes over it
        real_save(path, data)
    monkeypatch.setattr(guards, "save_json", save_with_a_night_add_racing)
    prof.add(0.20)
    other[0].join(timeout=10)
    assert not other[0].is_alive()
    led = guards.load_json(s / "dream" / "spend.json", {})
    assert led == {"2026-11-19": 0.30, filt.spend_key("2026-11-19"): 0.20}
    assert not (s / "dream" / "spend.lock").exists()


def test_a_stale_lock_is_taken_over_and_a_fresh_one_waited_for(tmp_path, monkeypatch):
    import os
    import time
    from dream import guards
    s = tmp_path / "s"
    b = guards.Budget(s, "2026-11-19", 5.0)
    lock = s / "dream" / "spend.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("", encoding="utf-8")
    old = time.time() - guards.SPEND_LOCK_STALE - 5
    os.utime(lock, (old, old))
    b.add(0.10)                                       # the dead run's lock is taken over at once
    assert b.spent() == pytest.approx(0.10) and not lock.exists()
    assert not list(lock.parent.glob("spend.lock*"))
    lock.write_text("", encoding="utf-8")             # a live holder that never lets go
    monkeypatch.setattr(guards, "SPEND_LOCK_WAIT", 0.1)
    start = time.monotonic()
    b.add(0.10)                                       # waits, then records the paid call anyway
    assert time.monotonic() - start >= 0.1
    assert b.spent() == pytest.approx(0.20) and lock.exists()


# --- keys scrubbed before reflect's calls ---------------------------------------------

def test_a_planted_key_never_reaches_a_model_call(cfg, tmp_path):
    key = "sk-ant-" + "Zq7" * 10                      # made up, in the Anthropic key shape
    turn = f"{GARDEN}, and my key is {key} for the garden app"
    path = chat(cfg, "c-k", f"garden {key}", [(turn, f"Never share {key} with anyone.", None)], "2025-07-03")
    # write_chat scrubs, so put the key back by hand: an older note, or one edited since, may hold one
    path.write_text(path.read_text(encoding="utf-8").replace("[REDACTED:ANTHROPIC_KEY]", key), encoding="utf-8")
    assert path.read_text(encoding="utf-8").count(key) == 4   # title, heading, turn and answer
    score(cfg, path)
    write(cfg.vault, "2 Areas/Profile/basics.md", render({"source": "author"}, f"- Lives in Sapporo\n- key {key}\n"))
    write(cfg.vault, "3 Resources/imports/chatgpt memory.md", f"- User keeps {key} in a note\n")
    write(cfg.vault, reflect.REJECTED_REL, render({"source": "author"}, f"- The user shares {key}\n"))
    model = Model({"2025-07": [FACT]})
    assert reflect_run(cfg, model, tmp_path / "out") == 0
    assert {c[0] for c in model.calls} >= {"claims", "verdicts"}
    for _, _, stdin in model.calls:
        assert key not in stdin and "Zq7Zq7Zq7" not in stdin
    ex, ck = model.prompts("claims")[0], model.prompts("verdicts")[0]
    assert "[REDACTED:ANTHROPIC_KEY]" in ex and "Lives in Sapporo" in ex and GARDEN in ex
    assert "User keeps [REDACTED:ANTHROPIC_KEY] in a note" in ck
    # the quote check searches the same scrubbed text, so the claim built on that turn is kept
    out = (tmp_path / "out" / "spike facts digest.md").read_text(encoding="utf-8")
    assert "The user waters the tomatoes every morning before work." in out and key not in out


def test_the_filter_prompt_scrubs_the_profile_notes_too(cfg):
    key = "sk-ant-" + "Zq7" * 10
    write(cfg.vault, "2 Areas/Profile/basics.md", render({"source": "author"}, f"- Lives in Sapporo\n- key {key}\n"))
    model = FakeClaude({"chats": []})
    assert filt.run(cfg, cap_usd=1.0, runner=model, now=NOW) == 0
    assert model.calls and all(key not in stdin for _, stdin, _ in model.calls)
    assert "Lives in Sapporo" in model.calls[0][1]
