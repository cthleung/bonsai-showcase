"""The night job reads only new material (the inbox, and chats captured and
updated since the last finished run). Every note here is made up."""
import datetime as dt

from common.notes import render
from dream import collect, guards, night
from tests.dream.conftest import CHAT_REL, INBOX_REL, chat_body, chat_meta, chat_note, chat_rel, write

NOW = dt.datetime(2026, 11, 18, 3, 0, 0)
DAY = "2026-11-18"


# --- 1. new material only -------------------------------------------------------------------

def test_since_date_is_the_watermark_day_or_first_days_back():
    assert collect.since_date("2026-11-16T02:00:05", dt.date(2026, 11, 18)) == "2026-11-16"
    assert collect.since_date(None, dt.date(2026, 11, 18)) == "2026-11-15"
    assert collect.since_date(None, dt.date(2026, 11, 18), first_days=10) == "2026-11-08"


def test_window_reads_new_chats_and_leaves_the_backfill_and_sorted_out_chats(vault):
    write(vault, chat_rel("backfilled (Bb000001)"),
          chat_note("old trip", [("where to stay", "a hotel")], updated="2023-07-11", captured="2026-11-18"))
    write(vault, chat_rel("captured before (Cc000001)"),
          chat_note("last month", [("a question", "an answer")], updated="2026-11-01"))
    write(vault, chat_rel("continued lately (Dd000001)"),
          chat_note("an old chat the user came back to", [("one more thing", "sure")],
                    updated="2026-11-17", created="2024-02-01"))
    write(vault, chat_rel("junk (Ee000001)"),
          chat_note("junk", [("spam", "spam")], updated="2026-11-17", excluded="junk"))
    write(vault, "4 Archive/chats-excluded/junk/moved (Ff000001).md",
          chat_note("moved out", [("spam", "spam")], updated="2026-11-17"))
    history = []
    rels = {p.rel for p in collect.collect(vault, {}, since="2026-11-15", history=history)}
    assert rels == {CHAT_REL, INBOX_REL, chat_rel("continued lately (Dd000001)")}
    assert history == [chat_rel("backfilled (Bb000001)")]


def test_inbox_notes_are_read_whatever_their_date(vault):
    write(vault, "0 Inbox/mac/old idea.md", render({"date": "2024-01-01"}, "An idea from long ago.\n"))
    rels = {p.rel for p in collect.collect(vault, {}, since="2026-11-15")}
    assert {INBOX_REL, "0 Inbox/mac/old idea.md"} <= rels


def test_a_chat_updated_just_before_the_window_still_counts(vault):
    # capture runs hourly, so a chat updated late on the night before can be captured after the run
    write(vault, CHAT_REL, render(chat_meta(updated="2026-11-14"), chat_body(2)))
    meta = chat_meta(updated="2026-11-14")
    meta["captured"] = "2026-11-16"
    write(vault, CHAT_REL, render(meta, chat_body(2)))
    assert CHAT_REL in {p.rel for p in collect.collect(vault, {}, since="2026-11-16")}


def answer():
    return {"proposals": [{"action": "add", "target": "1 Projects/Garden/memory/tomatoes-by-the-fence.md",
                           "reason": "The user picked a spot", "sources": ["S1"], "description": "where the tomatoes go",
                           "type": "project", "date": "2026-11-12", "text": "Tomatoes go along the south fence."}]}


def test_night_reads_the_window_and_counts_the_history(cfg, fake):
    write(cfg.vault, chat_rel("backfilled (Bb000001)"),
          chat_note("old trip", [("where to stay", "a hotel")], updated="2023-07-11", captured="2026-11-18"))
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    claude = fake(answer())
    assert night.run(cfg, runner=claude, now=NOW) == 0
    stdin = claude.calls[0][1]
    assert "question 1" in stdin and "where to stay" not in stdin
    text = (cfg.vault / "0 Inbox" / f"digest {DAY}.md").read_text(encoding="utf-8")
    assert "1 older chat captured since 2026-11-15 was left for the profile spike." in text
    st = guards.load_json(guards.dream_root(cfg.state) / "state.json", {})
    assert st["watermark"] == "2026-11-18T03:00:00" and chat_rel("backfilled (Bb000001)") not in st["seen"]


def test_first_days_comes_from_the_environment(cfg, fake, monkeypatch):
    write(cfg.vault, CHAT_REL, render(chat_meta(updated="2026-11-09"), chat_body(2)))
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    claude = fake(answer())
    night.run(cfg, runner=claude, now=NOW)
    assert "question 1" not in claude.calls[0][1]
    monkeypatch.setenv("BONSAI_DREAM_FIRST_DAYS", "10")
    (guards.dream_root(cfg.state) / "state.json").unlink()
    claude = fake(answer())
    night.run(cfg, runner=claude, now=NOW + dt.timedelta(hours=1))
    assert "question 1" in claude.calls[0][1]


def test_after_a_night_only_chats_captured_since_then_are_read(cfg, fake):
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    night.run(cfg, runner=fake(answer()), now=NOW)
    later = NOW + dt.timedelta(days=5)
    write(cfg.vault, chat_rel("new one (Gg000001)"),
          chat_note("new one", [("a fresh question", "a fresh answer")], updated="2026-11-22"))
    write(cfg.vault, chat_rel("trickled in (Hh000001)"),
          chat_note("from 2023", [("an old question", "an old answer")], updated="2023-03-03", captured="2026-11-22"))
    claude = fake({"proposals": []})
    night.run(cfg, runner=claude, now=later)
    stdin = claude.calls[0][1]
    assert "a fresh question" in stdin and "an old question" not in stdin


def test_leftovers_keep_the_watermark_so_they_are_read_on_a_later_night(cfg, fake, monkeypatch):
    write(cfg.vault, CHAT_REL, render(chat_meta(), chat_body(6, size=400)))
    real_pack = collect.pack
    monkeypatch.setattr(night.collect, "pack", lambda ps: real_pack(ps, max_call=1_500, max_night=1_500))
    real_collect = collect.collect
    monkeypatch.setattr(night.collect, "collect", lambda *a, **kw: real_collect(*a, max_piece=1_000, **kw))
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    assert night.run(cfg, runner=fake({"proposals": []}), now=NOW) == 0
    st = guards.load_json(guards.dream_root(cfg.state) / "state.json", {})
    assert st["watermark"] == "2026-11-15", "a first run with leftovers pins its window"
    done = st["seen"][CHAT_REL]["turns_done"]
    assert done < 6
    # a week later, far past the first window, the rest of the chat is still new material
    claude = fake({"proposals": []})
    night.run(cfg, runner=claude, now=NOW + dt.timedelta(days=7))
    assert f"question {done + 1} " in claude.calls[0][1]
