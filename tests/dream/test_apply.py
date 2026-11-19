import pytest

from common.notes import read_note
from dream import apply, digest, guards
from tests.dream.conftest import INBOX_REL, MEMORY_REL
from tests.dream.test_digest import DAY, props


def setup_digest(vault, state, items):
    path, _ = digest.write(vault, state, DAY, "03:00", items, [], [], "Stats.")
    return path


def tick(path, *ids):
    text = path.read_text(encoding="utf-8")
    for i in ids:
        lines = [l.replace("- [ ]", "- [x]", 1) if l.endswith(f"^{i}") else l for l in text.splitlines()]
        text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def real_items():
    p = props()
    p[2]["target"] = MEMORY_REL       # merge into the fixture's memory note
    p[3]["target"] = INBOX_REL        # move the fixture's inbox note
    p[4]["target"] = MEMORY_REL       # flag
    return p


def test_ticked_in_reads_only_ticked_boxes_whose_line_matches():
    def p(pid):
        return {"id": pid, "line": f"- [ ] add | a{pid} | b | source: [[c]] ^{pid}"}
    props_ = [p("p01"), p("p02"), p("p03"), {"id": "p04"}]
    text = ("- [x] add | ap01 | b | source: [[c]] ^p01\n- [ ] add | ap02 | b | source: [[c]] ^p02\n"
            "- [X] add | ap03 | b | source: [[c]] ^p03  \n- [x] add | ap04 | b | source: [[c]] ^p04\n")
    assert [q["id"] for q in apply.ticked_in(text, props_)] == ["p01", "p03"]


def test_nothing_ticked_changes_nothing(vault, state):
    setup_digest(vault, state, real_items())
    rep = apply.apply_ticked(vault, state, state / "vault-history", DAY, log=lambda m: None)
    assert rep["done"] == [] and not (state / "vault-history").exists()


def test_apply_snapshots_first_and_does_each_action(vault, state):
    path = setup_digest(vault, state, real_items())
    tick(path, "p01", "p02", "p03", "p04", "p05")
    before_memory = (vault / MEMORY_REL).read_text(encoding="utf-8")
    rep = apply.apply_ticked(vault, state, state / "vault-history", DAY, log=lambda m: None)
    assert len(rep["done"]) == 5 and rep["failed"] == []

    meta, body = read_note(vault / "1 Projects/Garden/memory/tomatoes.md")
    assert meta == {"name": "tomatoes", "description": "where the tomatoes go", "type": "project",
                    "source": "author-approved", "date": "2026-11-12", "visibility": "private"}
    assert body.startswith("Tomatoes go along the south fence.") and "Source: [[garden idea]]" in body
    assert read_note(vault / "2 Areas/Profile/likes-ramen.md")[0]["type"] == "user"
    assert "Water before 07:00 in summer." in (vault / MEMORY_REL).read_text(encoding="utf-8")
    assert not (vault / INBOX_REL).exists() and (vault / "1 Projects/Garden/garden idea.md").exists()

    snap = apply.Snapshot(state / "vault-history", vault)
    assert snap.blob(rep["before"], MEMORY_REL).decode("utf-8") == before_memory
    side = guards.load_json(digest.sidecar_path(state, DAY), {})
    assert all(p.get("applied") == DAY for p in side["proposals"])

    again = apply.apply_ticked(vault, state, state / "vault-history", DAY, log=lambda m: None)
    assert again["done"] == []


def test_dry_run_changes_nothing(vault, state):
    path = setup_digest(vault, state, real_items())
    tick(path, "p01", "p04")
    rep = apply.apply_ticked(vault, state, state / "vault-history", DAY, dry_run=True, log=lambda m: None)
    assert len(rep["done"]) == 2
    assert (vault / INBOX_REL).exists() and not (vault / "1 Projects/Garden/memory/tomatoes.md").exists()
    assert not (state / "vault-history").exists()


def test_a_proposal_that_no_longer_fits_is_skipped_not_forced(vault, state):
    path = setup_digest(vault, state, real_items())
    tick(path, "p01", "p02")
    (vault / "1 Projects/Garden/memory/tomatoes.md").write_text("The user wrote this first.\n", encoding="utf-8")
    rep = apply.apply_ticked(vault, state, state / "vault-history", DAY, log=lambda m: None)
    assert len(rep["done"]) == 1 and "exists now" in rep["failed"][0]
    assert (vault / "1 Projects/Garden/memory/tomatoes.md").read_text(encoding="utf-8") == "The user wrote this first.\n"


def test_undo_puts_things_back_but_leaves_later_edits_alone(vault, state):
    path = setup_digest(vault, state, real_items())
    tick(path, "p01", "p03", "p04")
    memory_before = (vault / MEMORY_REL).read_bytes()
    apply.apply_ticked(vault, state, state / "vault-history", DAY, log=lambda m: None)
    (vault / "1 Projects/Garden/memory/tomatoes.md").write_text("edited by the user\n", encoding="utf-8")
    res = apply.undo_last(vault, state, state / "vault-history", log=lambda m: None)
    assert (vault / MEMORY_REL).read_bytes() == memory_before
    assert (vault / INBOX_REL).exists() and not (vault / "1 Projects/Garden/garden idea.md").exists()
    assert res["kept"] == ["1 Projects/Garden/memory/tomatoes.md"]
    assert (vault / "1 Projects/Garden/memory/tomatoes.md").read_text(encoding="utf-8") == "edited by the user\n"
    side = guards.load_json(digest.sidecar_path(state, DAY), {})
    assert not any(p.get("applied") for p in side["proposals"])
    again = apply.apply_ticked(vault, state, state / "vault-history", DAY, log=lambda m: None)
    assert again["done"] == [], "an undone tick is not applied again by the next night"
    assert digest.pending_keys(state) == {(p["action"], p["target"]) for p in real_items() if p["action"] in ("profile", "flag")}
    assert apply.undo_last(vault, state, state / "vault-history", log=lambda m: None) == {"restored": [], "kept": [], "record": None}


def test_snapshot_leaves_out_obsidian_config(vault, state):
    (vault / ".obsidian-win").mkdir()
    (vault / ".obsidian-win" / "workspace.json").write_text("{}", encoding="utf-8")
    snap = apply.Snapshot(state / "vault-history", vault)
    sha = snap.commit("test")
    files = snap.git("ls-tree", "-r", "--name-only", sha).splitlines()
    assert MEMORY_REL in files and not any(f.startswith(".obsidian") for f in files)


def test_apply_rebuilds_the_index_inside_the_snapshot(vault, state):
    path = setup_digest(vault, state, real_items())
    tick(path, "p01")
    rep = apply.apply_ticked(vault, state, state / "vault-history", DAY, log=lambda m: None)
    index = (vault / "Index.md").read_text(encoding="utf-8")
    assert "tomatoes" in index
    snap = apply.Snapshot(state / "vault-history", vault)
    assert snap.blob(rep["after"], "Index.md") is not None and snap.blob(rep["before"], "Index.md") is None
    apply.undo_last(vault, state, state / "vault-history", log=lambda m: None)
    assert not (vault / "Index.md").exists()


def test_links_are_built_in_one_place():
    assert digest.wikilink("2026-11-10 ramen (AbCdEfGh)") == "[[2026-11-10 ramen (AbCdEfGh)]]"
    assert apply._source_line({"links": ["a", "b"]}) == "Source: [[a]], [[b]]"


def test_apply_rechecks_moves_against_this_computers_inbox(vault, state):
    (vault / "0 Inbox/mac").mkdir(parents=True, exist_ok=True)
    (vault / "0 Inbox/mac/from the mac.md").write_text("A note from the Mac.\n", encoding="utf-8")
    items = real_items()
    items[3]["target"] = "0 Inbox/mac/from the mac.md"   # a sidecar entry that slipped past validate
    path = setup_digest(vault, state, items)
    tick(path, "p04")
    rep = apply.apply_ticked(vault, state, state / "vault-history", DAY, log=lambda m: None, inbox="win")
    assert rep["done"] == [] and "another computer's inbox" in rep["failed"][0]
    assert (vault / "0 Inbox/mac/from the mac.md").exists()
    assert not (vault / "1 Projects/Garden/from the mac.md").exists()


def test_apply_moves_from_its_own_inbox(vault, state):
    path = setup_digest(vault, state, real_items())   # p04 moves 0 Inbox/win/garden idea.md
    tick(path, "p04")
    rep = apply.apply_ticked(vault, state, state / "vault-history", DAY, log=lambda m: None, inbox="win")
    assert rep["failed"] == [] and (vault / "1 Projects/Garden/garden idea.md").exists()
    assert not (vault / INBOX_REL).exists()
