import datetime as dt
import json
import os
import stat

import pytest

from common import config
from dream import digest, guards, night
from tests.dream.conftest import CHAT_REL, INBOX_REL

NOW = dt.datetime(2026, 11, 18, 3, 0, 0)
DAY = "2026-11-18"


def answer():
    return {"proposals": [{"action": "add", "target": "1 Projects/Garden/memory/tomatoes-by-the-fence.md",
                           "reason": "The user picked a spot", "sources": ["S1"], "description": "where the tomatoes go",
                           "type": "project", "date": "2026-11-12", "text": "Tomatoes go along the south fence."}]}


def test_copy_run_writes_the_digest_on_the_copy_only(cfg, fake):
    claude = fake(answer())
    assert night.run(cfg, on_copy=True, runner=claude, now=NOW) == 0
    copy = guards.dream_root(cfg.state) / "vault-copy"
    assert (copy / "0 Inbox" / f"digest {DAY}.md").exists()
    assert not (cfg.vault / "0 Inbox" / f"digest {DAY}.md").exists()
    assert guards.rehearsed(cfg.state)
    vstate = guards.vault_state_dir(cfg.state, copy, cfg.vault)
    assert vstate != guards.dream_root(cfg.state)
    assert not (guards.dream_root(cfg.state) / "state.json").exists()
    assert claude.calls[0][2] == guards.dream_root(cfg.state) / "work"
    claude2 = fake(answer())
    assert night.run(cfg, on_copy=True, runner=claude2, now=NOW) == 0
    assert len(claude2.calls) == 1, "a second copy run reads everything again"
    assert "^p01" in (copy / "0 Inbox" / f"digest {DAY}.md").read_text(encoding="utf-8")


def test_a_second_copy_run_replaces_read_only_folders_and_files(cfg, fake):
    # a sync client can mark vault folders read-only on Windows, and copytree keeps the mark
    ro_dir = cfg.vault / "1 Projects" / "Garden" / "memory"
    ro_file = ro_dir / "water-in-the-morning.md"
    os.chmod(ro_file, stat.S_IREAD)
    os.chmod(ro_dir, stat.S_IREAD | stat.S_IEXEC)
    try:
        assert night.run(cfg, on_copy=True, runner=fake(answer()), now=NOW) == 0
        assert night.run(cfg, on_copy=True, runner=fake(answer()), now=NOW) == 0
        copy = guards.dream_root(cfg.state) / "vault-copy"
        assert (copy / "1 Projects" / "Garden" / "memory" / "water-in-the-morning.md").exists()
    finally:
        os.chmod(ro_dir, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        os.chmod(ro_file, stat.S_IWRITE | stat.S_IREAD)


def test_real_vault_refused_before_a_copy_run(cfg, fake):
    claude = fake(answer())
    assert night.run(cfg, runner=claude, now=NOW) == 2
    assert claude.calls == []


def test_full_night_then_nothing_new(cfg, fake):
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    claude = fake(answer())
    assert night.run(cfg, runner=claude, now=NOW) == 0
    text = (cfg.vault / "0 Inbox" / f"digest {DAY}.md").read_text(encoding="utf-8")
    assert "- [ ] add | 1 Projects/Garden/memory/tomatoes-by-the-fence.md | The user picked a spot | source: [[garden idea]] ^p01" in text
    assert "Read 1 chat and 1 inbox note on 2026-11-18: 1 proposal." in text
    st = guards.load_json(guards.dream_root(cfg.state) / "state.json", {})
    assert set(st["seen"]) == {CHAT_REL, INBOX_REL} and st["watermark"] == "2026-11-18T03:00:00"
    assert night.run(cfg, runner=claude, now=NOW) == 0
    assert len(claude.calls) == 1   # nothing new the second time


def test_ticks_are_applied_at_the_start_of_the_next_night(cfg, fake):
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    night.run(cfg, runner=fake(answer()), now=NOW)
    path = cfg.vault / "0 Inbox" / f"digest {DAY}.md"
    path.write_text(path.read_text(encoding="utf-8").replace("- [ ] add", "- [x] add"), encoding="utf-8")
    night.run(cfg, runner=fake(), now=NOW + dt.timedelta(days=1))
    assert (cfg.vault / "1 Projects/Garden/memory/tomatoes-by-the-fence.md").exists()
    assert (cfg.state / "vault-history" / "HEAD").exists()


def test_dry_run_writes_nothing_and_moves_no_watermark(cfg, fake, capsys):
    claude = fake(answer())
    assert night.run(cfg, dry_run=True, runner=claude, now=NOW) == 0
    assert "- [ ] add" in capsys.readouterr().out
    assert not (cfg.vault / "0 Inbox" / f"digest {DAY}.md").exists()
    assert not (guards.dream_root(cfg.state) / "state.json").exists()


def test_stop_file_stops_it(cfg, fake):
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    (cfg.vault / "0 Inbox" / "STOP dream.md").write_text("", encoding="utf-8")
    claude = fake(answer())
    assert night.run(cfg, runner=claude, now=NOW) == 2 and claude.calls == []
    (cfg.vault / "0 Inbox" / "STOP dream.md").unlink()
    (guards.dream_root(cfg.state) / "STOP").write_text("", encoding="utf-8")
    assert night.run(cfg, runner=claude, now=NOW) == 2 and claude.calls == []


def test_lock_held_by_a_live_run_stops_a_second(cfg, fake):
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    lock = guards.dream_root(cfg.state) / "lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "started": NOW.isoformat()}), encoding="utf-8")
    claude = fake(answer())
    assert night.run(cfg, runner=claude, now=NOW) == 2 and claude.calls == []
    lock.write_text(json.dumps({"pid": os.getpid(), "started": (NOW - dt.timedelta(hours=7)).isoformat()}), encoding="utf-8")
    assert night.run(cfg, runner=claude, now=NOW) == 0 and not lock.exists()


def test_only_the_pc_runs_it(cfg, fake):
    mac = config.Config(vault=cfg.vault, state=cfg.state, machine="mac")
    assert night.run(mac, on_copy=True, runner=fake(answer()), now=NOW) == 2


def test_cap_reached_means_no_digest_and_notes_wait(cfg, fake):
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    guards.Budget(cfg.state, DAY, 1.0).add(1.0)
    claude = fake(answer())
    assert night.run(cfg, runner=claude, now=NOW, cap_usd=1.0) == 1
    assert claude.calls == []
    assert not (cfg.vault / "0 Inbox" / f"digest {DAY}.md").exists()
    assert not (guards.dream_root(cfg.state) / "state.json").exists()


def test_empty_answer_still_writes_a_short_digest(cfg, fake):
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    assert night.run(cfg, runner=fake({"proposals": []}), now=NOW) == 0
    text = (cfg.vault / "0 Inbox" / f"digest {DAY}.md").read_text(encoding="utf-8")
    assert "0 proposals" in text and "- [ ]" not in text and digest.check_digest(text) == []
