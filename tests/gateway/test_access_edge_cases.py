"""Gateway access edge cases on made-up temp vaults: odd frontmatter and encodings, one refusal
message, hard links, and index rows that went stale since the build."""
import json
import os
import time

import pytest

from common.notes import render
from gateway import access, searching
from index import build, chunking, store
from tests.gateway.conftest import CHAT_REL, PROFILE_META, PROFILE_REL, SECRET
from tests.index.conftest import FakeEmbedder, chat_body, chat_meta, write

QUIET = lambda *a, **k: None   # noqa: E731

# --- what counts as a diary or an excluded note ---------------------------------------------------

# Each of these must be refused by the gateway and skipped by the index.
HIDDEN_BY_FRONTMATTER = {
    "profile-diary.md": "---\ntype: profile-diary\n---\nzebrafish monthly diary\n",   # dream/reflect.py writes this
    "link-quoted.md": '---\ntype: "[[diary]]"\n---\nzebrafish diary via link\n',
    "link-bare.md": "---\ntype: [[diary]]\n---\nzebrafish diary via bare link\n",
    "list.md": "---\ntype: [note, Diary]\n---\nzebrafish\n",
    "capital-key.md": "---\nType: diary\n---\nzebrafish capital key\n",
    "types-key.md": "---\ntypes: diary\n---\nzebrafish\n",
    "excluded-empty.md": "---\nexcluded:\n---\nzebrafish excluded empty\n",
    "excluded-false.md": "---\nexcluded: false\n---\nzebrafish excluded false\n",
    "excluded-capital.md": "---\nExcluded: person\n---\nzebrafish\n",
    "bom.md": "\ufeff---\ntype: diary\n---\nzebrafish bom\n",
    "crlf.md": "---\r\ntype: diary\r\n---\r\nzebrafish crlf\r\n",
    "fence-trailing-space.md": "--- \ntype: diary\n---  \nzebrafish trailing space on the fences\n",
    "leading-blank.md": "\n\n---\ntype: diary\n---\nzebrafish leading blank lines\n",
    "dots-close.md": "---\ntype: diary\n...\nzebrafish yaml end marker\n",
    "yaml-tag.md": "---\ntype: !!str diary\n---\nzebrafish\n",
    "unclosed.md": "---\ntype: diary\nzebrafish with no closing fence\n",
    "not-a-mapping.md": "---\n- diary\n---\nzebrafish\n",
}
NOT_UTF8 = {
    "utf16.md": "\ufeff---\ntype: diary\n---\nzebrafish utf16 diary\n".encode("utf-16-le"),
    "latin1.md": "---\ntitle: caf\xe9\n---\nzebrafish\n".encode("latin-1"),
}
FOLDER = "2 Areas/t"


@pytest.fixture
def odd(vault):
    for name, text in HIDDEN_BY_FRONTMATTER.items():
        (vault / FOLDER).mkdir(parents=True, exist_ok=True)
        (vault / FOLDER / name).write_bytes(text.encode("utf-8"))
    for name, data in NOT_UTF8.items():
        (vault / FOLDER / name).write_bytes(data)
    return vault


@pytest.mark.parametrize("name", list(HIDDEN_BY_FRONTMATTER) + list(NOT_UTF8))
def test_odd_frontmatter_and_encodings_are_refused(odd, name):
    with pytest.raises(access.Refused) as e:
        access.read_note(odd, f"{FOLDER}/{name}")
    assert str(e.value) == access.REFUSED
    assert not access.readable(odd, f"{FOLDER}/{name}")


def test_odd_frontmatter_and_encodings_never_reach_the_index_or_recent(odd, settings):
    build.build(settings, FakeEmbedder(), log=QUIET)
    db = store.connect(settings.db_path)
    try:
        indexed = {r["path"] for r in db.execute("SELECT path FROM notes")}
    finally:
        db.close()
    assert not {p for p in indexed if p.startswith(FOLDER)}
    assert not {r.path for r in access.recent(odd, days=1, limit=100) if r.path.startswith(FOLDER)}
    assert searching.run(settings, SECRET, mode="bm25", log=QUIET)["hits"] == []


def test_ordinary_frontmatter_still_passes():
    for meta in ({"type": "user"}, {"type": ["project", "note"]}, {"title": "diary of a wimpy kid"},
                 {"type": None}, {}):
        assert not chunking.excluded_meta(meta)
    assert chunking.skip_reason("no frontmatter at all\n") is None
    assert chunking.skip_reason("---\ntitle: x\n---\nbody\n") is None


def test_a_note_indexed_under_the_old_rules_is_removed_on_the_next_build(settings, monkeypatch):
    rel = "2 Areas/Profile/monthly.md"
    write(settings.vault, rel, render({**PROFILE_META, "type": "profile-diary"}, "a month\n"))
    monkeypatch.setattr(chunking, "excluded_meta", lambda meta: False)
    monkeypatch.setattr(chunking, "RULES_VERSION", "1")
    build.build(settings, FakeEmbedder(), log=QUIET)
    monkeypatch.undo()
    build.build(settings, FakeEmbedder(), log=QUIET)       # the file never changed; the rules did
    db = store.connect(settings.db_path)
    try:
        assert rel not in {r["path"] for r in db.execute("SELECT path FROM notes")}
    finally:
        db.close()


# --- one refusal message, and trailing dots and spaces taken off first ----------------------------

def test_a_refusal_never_says_whether_the_note_exists(vault):
    msgs = set()
    for rel in ("0 Inbox/win/digest 2026-11-19.md", "0 Inbox/win/no such note.md", "2 Areas/Profile/rejected.md",
                "2 Areas/Profile/not here.md", "../x.md", "C:/x.md", "nowhere/at/all.md", "2 Areas/Profile"):
        with pytest.raises(access.Refused) as e:
            access.read_note(vault, rel)
        msgs.add(str(e.value))
    assert msgs == {access.REFUSED}


@pytest.mark.parametrize("variant", ["0 Inbox./win/digest 2026-11-19.md", "0 Inbox /win/digest 2026-11-19.md",
                                     "0 Inbox. . ./win/digest 2026-11-19.md", "2 Areas/Profile/rejected.md. ",
                                     "2 Areas/Profile/Long view.md...", "2 Areas/Profile/diary./2026-11-19.md"])
def test_trailing_dots_and_spaces_are_taken_off_before_the_first_check(vault, variant):
    with pytest.raises(access.Refused) as e:
        access.read_note(vault, variant)
    assert e.value.reason == "off limits as asked"


# --- hard links -----------------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["2 Areas/Profile/Long view.md", PROFILE_REL])
def test_a_file_with_a_second_hard_link_is_refused_by_both_names(vault, target):
    link = vault / "3 Resources" / "another name.md"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(vault / target, link)
    except OSError:
        pytest.skip("this file system can't make hard links")
    for rel in ("3 Resources/another name.md", target):
        with pytest.raises(access.Refused):
            access.read_note(vault, rel)


# --- stale index rows: no oracle, fresh snippets, and the log holds only what was returned --------

def _last_logged(s) -> dict:
    return json.loads(s.query_log.read_text(encoding="utf-8").splitlines()[-1])


def test_a_dropped_hit_is_never_counted_or_logged(built):
    write(built.vault, CHAT_REL, render(chat_meta("home server plans", "chatgpt", "ab12cd34ef", "2025-03-02",
                                                  excluded="person"),
                                        chat_body("home server plans", "ChatGPT", [("2025-03-02", "nas", "x")])))
    out = searching.run(built, "synology nas", mode="bm25", log=QUIET)
    assert out["hits"] == [] and out["notes"] == []
    assert _last_logged(built)["top"] == []


def test_a_changed_note_gets_its_snippet_and_title_from_the_file_as_it_is_now(built):
    """The chat lost a line after the build. The stale row still matches, but what the client sees
    comes from the current file."""
    old = (built.vault / CHAT_REL).read_text(encoding="utf-8")
    new = old.replace("Open Storage Manager and create a pool.", "Pick drives carefully.").replace(
        "title: home server plans", "title: nas notes")
    assert new != old
    (built.vault / CHAT_REL).write_text(new, encoding="utf-8", newline="\n")
    t = time.time() + 5
    os.utime(built.vault / CHAT_REL, (t, t))
    out = searching.run(built, "storage manager", mode="bm25", log=QUIET)
    assert [h["path"] for h in out["hits"]] == [CHAT_REL]
    hit = out["hits"][0]
    assert "storage manager" not in hit["snippet"].lower() and "pick drives" in hit["snippet"].lower()
    assert hit["title"] == "nas notes"
    assert [h["path"] for h in _last_logged(built)["top"]] == [CHAT_REL]


def test_a_changed_note_whose_turn_is_gone_is_dropped(built):
    text = render(chat_meta("home server plans", "chatgpt", "ab12cd34ef", "2025-03-02"),
                  chat_body("home server plans", "ChatGPT", [("2025-03-02", "only one turn now", "ok")]))
    (built.vault / CHAT_REL).write_text(text, encoding="utf-8", newline="\n")
    out = searching.run(built, "hyper backup cloud bucket", mode="bm25", log=QUIET)   # turn 2 in the index
    assert CHAT_REL not in [h["path"] for h in out["hits"]] and out["notes"] == []
