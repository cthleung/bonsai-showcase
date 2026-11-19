import re

import pytest

from common.notes import read_note
from dream import digest, guards

DAY = "2026-11-18"
LINE_RX = re.compile(r"^- \[ \] (add|merge|move|flag|profile) \| [^|]+ \| [^|]+ \| source: (\[\[[^\n]+?\]\])(, \[\[[^\n]+?\]\])* \^p\d\d$")


def props():
    return [
        {"action": "add", "target": "1 Projects/Garden/memory/tomatoes.md", "reason": "The user picked a spot",
         "sources": ["0 Inbox/win/garden idea.md"], "links": ["garden idea"], "visibility": "private",
         "name": "tomatoes", "description": "where the tomatoes go", "type": "project", "date": "2026-11-12",
         "text": "Tomatoes go along the south fence.\n**Why:** sun.\n**How to apply:** plant there."},
        {"action": "profile", "target": "2 Areas/Profile/likes-ramen.md", "reason": "said it twice",
         "sources": ["c.md"], "links": ["2026-11-10 ramen (AbCdEfGh)", "garden idea"], "visibility": "private",
         "name": "likes-ramen", "description": "food", "type": "user", "date": "2026-11-10", "text": "The user likes ramen."},
        {"action": "merge", "target": "1 Projects/Garden/memory/water.md", "reason": "adds a time",
         "sources": ["c.md"], "links": ["x"], "visibility": "private", "text": "Water before 07:00 in summer."},
        {"action": "move", "target": "0 Inbox/win/garden idea.md", "reason": "belongs with the garden",
         "sources": ["c.md"], "links": ["garden idea"], "visibility": "private", "move_to": "1 Projects/Garden/"},
        {"action": "flag", "target": "1 Projects/Garden/memory/water.md", "reason": "times disagree",
         "sources": ["c.md"], "links": ["x"], "visibility": "private", "text": "The note says 08:00, the chat says 06:00."},
    ]


@pytest.mark.parametrize("text", [
    "a " + chr(0x2014) + " b", "a " + chr(0x2013) + " b", "half the time", "it adds weight", "the gravity of it",
    "a structural fix", "the load-bearing part", "some scaffolding", "a new framework", "the architecture",
    "said it yesterday", "due tomorrow", "set last week", "maybe later", "it might work", "probably fine",
    "it seems right",
])
def test_bans_are_caught(text):
    assert digest.check_text(text)


@pytest.mark.parametrize("text", ["The user likes ramen.", "Water before 07:00.", "set on 2026-11-17", "a full hour"])
def test_plain_text_passes(text):
    assert digest.check_text(text) == []


def test_digest_lines_follow_the_format_and_pass_the_ban_check(vault, state):
    path, text = digest.write(vault, state, DAY, "03:00", props(), [], [{"link": "garden idea", "turns": None}],
                              "Read 0 chats and 1 inbox note on 2026-11-18: 5 proposals. Cost $0.01.")
    assert path == vault / "0 Inbox" / f"digest {DAY}.md"
    meta, body = read_note(path)
    assert meta["visibility"] == "private" and meta["date"] == DAY
    boxes = [l for l in body.splitlines() if l.startswith("- [ ]")]
    assert len(boxes) == 5
    for line in boxes:
        assert LINE_RX.match(line), line
    assert "    > Tomatoes go along the south fence." in body
    assert "Moves to 1 Projects/Garden/" in body
    assert digest.check_digest(body) == []
    side = guards.load_json(digest.sidecar_path(state, DAY), {})
    assert [p["id"] for p in side["proposals"]] == ["p01", "p02", "p03", "p04", "p05"]


def test_note_names_in_links_and_paths_are_not_prose(vault, state):
    p = props()[0]
    p["links"] = ["maybe tomorrow"]
    p["target"] = "1 Projects/Garden/memory/half-moon.md"
    _, text = digest.write(vault, state, DAY, "03:00", [p], [], [], "Stats.", dry_run=True)
    assert digest.check_digest(text) == []


def test_bad_words_in_prose_hold_that_item_and_bad_static_text_stops_the_digest(vault, state):
    p = props()[0]
    p["text"] = "It might rain."
    path, text = digest.write(vault, state, DAY, "03:00", [p, props()[1]], [], [], "Stats.")
    assert "might" not in text and "- [ ] profile" in text and "- [ ] add" not in text
    assert "Held back for wording" in text
    side = guards.load_json(digest.sidecar_path(state, DAY), {})
    assert [h["id"] for h in side["held"]] == ["p01"]
    with pytest.raises(ValueError, match="writing rules"):
        digest.write(vault, state, "2026-11-19", "03:00", [props()[1]], [], [], "Read it today.")
    assert not (vault / "0 Inbox" / "digest 2026-11-19.md").exists()


def test_held_items_show_no_text_and_no_box(vault, state):
    h = props()[0]
    h["text"] = "It might rain."
    h["held"] = ['hedging: "might"']
    _, text = digest.write(vault, state, DAY, "03:00", [], [h], [], "Stats.")
    assert "might" not in text and "Held back for wording" in text
    assert "- [ ]" not in text


def test_dry_run_writes_nothing(vault, state):
    path, text = digest.write(vault, state, DAY, "03:00", props(), [], [], "Stats.", dry_run=True)
    assert path is None and "- [ ] add" in text
    assert not (vault / "0 Inbox" / f"digest {DAY}.md").exists()
    assert not digest.sidecar_path(state, DAY).exists()


def test_second_run_same_day_appends_and_keeps_ticks(vault, state):
    path, _ = digest.write(vault, state, DAY, "03:00", props()[:1], [], [], "First.")
    path.write_text(path.read_text(encoding="utf-8").replace("- [ ] add", "- [x] add"), encoding="utf-8")
    digest.write(vault, state, DAY, "05:00", props()[1:2], [], [], "Second.")
    body = path.read_text(encoding="utf-8")
    assert "- [x] add" in body and "# Run at 05:00" in body and "^p02" in body
    assert digest.pending_keys(state) == {("add", props()[0]["target"]), ("profile", props()[1]["target"])}


def test_banned_words_in_note_names_do_not_stop_the_digest(vault, state):
    named = props()[0]
    named["target"] = "1 Projects/Garden/memory/software-architecture.md"
    named["name"] = "software-architecture"
    mover = props()[3]
    mover["move_to"] = "2 Areas/Weight training/"
    held = props()[1]
    held["target"] = "2 Areas/Profile/half-marathon.md"
    held["held"] = ['hedging: "maybe"']
    path, text = digest.write(vault, state, DAY, "03:00", [named, mover], [held], [], "Stats.")
    assert path.exists() and digest.check_digest(text) == []
    assert "- [ ] add | 1 Projects/Garden/memory/software-architecture.md |" in text
    assert "Moves to 2 Areas/Weight training/" in text
    assert "- profile | 2 Areas/Profile/half-marathon.md | broke: hedging" in text
