"""Undo with targets that differ in letter case, and ticked lines that don't match. All content is made up."""
import os

import pytest

from dream import apply
from tests.dream.conftest import INBOX_REL, MEMORY_REL
from tests.dream.test_apply import real_items, setup_digest, tick
from tests.dream.test_digest import DAY


def quiet(_m):
    pass


def names(folder):
    return sorted(os.listdir(folder))


# --- undo and letter case -------------------------------------------------------------

def test_undo_handles_targets_that_differ_only_in_letter_case(vault, state):
    if not (vault / MEMORY_REL.upper()).exists():
        pytest.skip("this file system tells letter case apart, so the bug can't happen here")
    items = real_items()
    items[0]["target"] = "1 Projects/garden/memory/tomatoes.md"            # the folder is Garden on disk
    items[2]["target"] = "1 Projects/Garden/memory/Water-In-the-morning.md"  # the note is water-in-the-morning.md
    items[3]["target"] = "0 Inbox/win/Garden Idea.md"                       # the note is garden idea.md
    path = setup_digest(vault, state, items)
    tick(path, "p01", "p03", "p04")
    memory_before = (vault / MEMORY_REL).read_bytes()
    memory_dir, inbox_dir, garden_dir = (vault / MEMORY_REL).parent, (vault / INBOX_REL).parent, vault / "1 Projects/Garden"
    memory_names, inbox_names, garden_names = names(memory_dir), names(inbox_dir), names(garden_dir)

    rep = apply.apply_ticked(vault, state, state / "vault-history", DAY, log=quiet)
    assert len(rep["done"]) == 3 and rep["failed"] == []
    assert (vault / MEMORY_REL).read_bytes() != memory_before

    res = apply.undo_last(vault, state, state / "vault-history", log=quiet)
    assert res["kept"] == []
    assert (vault / MEMORY_REL).read_bytes() == memory_before          # the merge is reverted
    assert names(memory_dir) == memory_names                           # the added note is gone
    assert names(inbox_dir) == inbox_names                             # the moved note is back, once
    assert names(garden_dir) == garden_names                           # and not left at the new place


# --- a tick on a line that doesn't match is logged and shown ---------------------------

def edit_line(path, pid, change):
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [change(l) if l.endswith(f"^{pid}") else l for l in lines]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_ticked_line_that_doesnt_match_is_logged_not_applied(vault, state):
    path = setup_digest(vault, state, real_items())
    tick(path, "p01", "p03", "p04")
    edit_line(path, "p01", lambda l: "  " + l)                                    # indented
    edit_line(path, "p03", lambda l: l.replace("adds a time", "adds the time"))   # an edited reason
    edit_line(path, "p04", lambda l: l.replace("- [x]", "- [X]") + "   ")        # harmless: still counts
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))   # CRLF, harmless too
    said = []
    rep = apply.apply_ticked(vault, state, state / "vault-history", DAY, log=said.append)
    assert rep["done"] == ["moved 0 Inbox/win/garden idea.md to 1 Projects/Garden/"]
    assert not (vault / "1 Projects/Garden/memory/tomatoes.md").exists()
    hits = [m for m in said if "ticked but the line doesn't match" in m]
    assert len(hits) == 2
    assert any("ticked but the line doesn't match: p01" in m and DAY in m for m in hits)
    assert any("ticked but the line doesn't match: p03" in m for m in hits)
    assert not any("The user picked a spot" in m for m in said)                   # the id only, never the line
    assert apply.mismatched(vault, state) == [(DAY, "p01"), (DAY, "p03")]


def test_the_next_digest_shows_the_mismatch_at_the_top(cfg, fake):
    import datetime as dt
    from dream import guards, night
    from tests.dream.conftest import write
    from tests.dream.test_night import NOW, answer
    guards.mark_rehearsed(cfg.state, cfg.vault, "x")
    assert night.run(cfg, runner=fake(answer()), now=NOW) == 0
    first = cfg.vault / "0 Inbox" / f"digest {DAY}.md"
    edit_line(first, "p01", lambda l: "    " + l.replace("- [ ]", "- [x]"))    # ticked, but indented
    write(cfg.vault, "0 Inbox/win/new idea.md", "Try a rain barrel by the shed.\n")
    later = NOW + dt.timedelta(days=1)
    assert night.run(cfg, runner=fake(), now=later) == 0
    assert not (cfg.vault / "1 Projects/Garden/memory/tomatoes-by-the-fence.md").exists()
    log = (guards.dream_root(cfg.state) / "night.log").read_text(encoding="utf-8")
    assert f"ticked but the line doesn't match: p01 (digest {DAY})" in log
    text = (cfg.vault / "0 Inbox" / f"digest {later.date().isoformat()}.md").read_text(encoding="utf-8")
    notice = text.index(f"ticked but the line doesn't match: p01 in [[digest {DAY}]]")
    assert notice < text.index("Read ") and "## Ticks that didn't count" in text
