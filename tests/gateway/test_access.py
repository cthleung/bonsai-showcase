"""gateway/access.py: path escapes, every exclusion, read_note and recent, on made-up vaults."""
import os
import sys
import time
from pathlib import Path

import pytest

from common.notes import render
from gateway import access
from index import chunking
from tests.gateway.conftest import (BROKEN_REL, CHAT_REL, DIARY_TYPE_REL, EXCLUDED_META_REL,
                                    EXCLUDED_PATH_NOTES, HIDDEN_NOTES, MEMORY_REL, OFF_LIMITS,
                                    PROFILE_META, PROFILE_REL, RESEARCH_REL, SECRET)
from tests.index.conftest import write


def _reason(vault, rel) -> str:
    with pytest.raises(access.Refused) as e:
        access.read_note(vault, rel)
    assert str(e.value) == access.REFUSED          # the client always sees the same message
    return e.value.reason


def test_every_excluded_path_has_a_test_note():
    assert set(EXCLUDED_PATH_NOTES) == set(chunking.EXCLUDED_PATHS)


@pytest.mark.parametrize("rel", [CHAT_REL, MEMORY_REL, PROFILE_REL, RESEARCH_REL])
def test_ordinary_notes_are_served(vault, rel):
    out = access.read_note(vault, rel)
    assert out["path"] == rel and out["text"] == (vault / rel).read_text(encoding="utf-8")
    assert out["start"] == 0 and out["end"] == out["chars"] and out["more"] is False


@pytest.mark.parametrize("rel", OFF_LIMITS)
def test_every_off_limits_note_is_refused(vault, rel):
    assert (vault / rel).is_file()
    with pytest.raises(access.Refused):
        access.read_note(vault, rel)
    assert not access.readable(vault, rel)


def test_the_rules_are_the_index_rules_not_a_copy(vault, monkeypatch):
    """A folder added to the index's list is refused at once, with no change to the gateway."""
    assert access.read_note(vault, RESEARCH_REL)
    monkeypatch.setattr(chunking, "EXCLUDED_PATHS", chunking.EXCLUDED_PATHS + ("3 Resources/research/",))
    with pytest.raises(access.Refused):
        access.read_note(vault, RESEARCH_REL)
    monkeypatch.setattr(chunking, "EXCLUDED_TYPES", chunking.EXCLUDED_TYPES + ("user",))
    with pytest.raises(access.Refused):
        access.read_note(vault, PROFILE_REL)


@pytest.mark.parametrize("bad", [
    "", "   ",
    "../outside.md", "2 Areas/../../outside.md", "2 Areas\\..\\..\\outside.md",
    "/etc/passwd", "\\Windows\\win.ini", "C:/Windows/win.ini", "C:outside.md", "c:\\x.md",
    "//server/share/x.md", "\\\\server\\share\\x.md", "\\\\?\\C:\\x.md", "\\\\.\\C:\\x.md",
    "2 Areas/Profile/rejected.md::$DATA", "2 Areas/Profile/rejected.md:stream",
    "file:///C:/x.md", "x\x00.md",
])
def test_path_escapes_are_refused(vault, bad):
    (vault.parent / "outside.md").write_text("outside the vault\n", encoding="utf-8")
    with pytest.raises(access.Refused):
        access.read_note(vault, bad)


@pytest.mark.parametrize("variant", [
    "2 Areas/Profile/REJECTED.MD", "2 areas/profile/Rejected.md", "0 inbox/WIN/digest 2026-11-19.md",
    "2 Areas/Profile/rejected.md.", "2 Areas/Profile/rejected.md ", "2 Areas/Profile/Long view.md...",
    "0 Inbox./win/digest 2026-11-19.md", "0 Inbox /win/digest 2026-11-19.md",
    "./0 Inbox/win/digest 2026-11-19.md", "2 Areas/./Profile/diary/2026-11-19.md",
    "2 Areas//Profile//rejected.md", "2 Areas\\Profile\\rejected.md",
])
def test_spelling_variants_of_an_excluded_note_are_refused(vault, variant):
    with pytest.raises(access.Refused):
        access.read_note(vault, variant)


def _short(p: Path) -> str:
    import ctypes
    from ctypes import wintypes
    get = ctypes.windll.kernel32.GetShortPathNameW
    get.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    buf = ctypes.create_unicode_buffer(1024)
    return buf.value if get(str(p), buf, 1024) else ""


@pytest.mark.skipif(sys.platform != "win32", reason="8.3 short names are a Windows thing")
def test_a_short_name_for_an_excluded_folder_is_refused(vault):
    short_vault, short_note = _short(vault), _short(vault / "0 Inbox" / "win" / "digest 2026-11-19.md")
    rel = Path(short_note).relative_to(short_vault).as_posix() if short_vault and short_note else ""
    if "~" not in rel:
        pytest.skip("this volume doesn't make 8.3 short names")
    with pytest.raises(access.Refused):
        access.read_note(vault, rel)


def test_only_md_files_are_served(vault):
    write(vault, "2 Areas/Profile/photo.png", "not a note")
    write(vault, "2 Areas/Profile/settings.json", "{}")
    for rel in ("2 Areas/Profile/photo.png", "2 Areas/Profile/settings.json", "2 Areas/Profile", "missing.md"):
        with pytest.raises(access.Refused):
            access.read_note(vault, rel)


def _link(link: Path, target: Path) -> None:
    """A directory link without admin rights: a junction on Windows, a symlink elsewhere."""
    if sys.platform == "win32":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def test_a_link_out_of_the_vault_is_refused(vault, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.md").write_text(f"{SECRET}\n", encoding="utf-8")
    _link(vault / "2 Areas" / "escape", outside)
    assert (vault / "2 Areas" / "escape" / "secret.md").is_file()
    assert _reason(vault, "2 Areas/escape/secret.md") == "leads outside the vault"


def test_a_link_into_an_excluded_folder_is_refused(vault):
    _link(vault / "2 Areas" / "inbox link", vault / "0 Inbox")
    rel = "2 Areas/inbox link/win/digest 2026-11-19.md"
    assert (vault / rel).is_file()
    assert _reason(vault, rel) == "off limits once resolved"
    assert rel not in {r.path for r in access.recent(vault, days=1, limit=100)}


def test_a_file_symlink_out_of_the_vault_is_refused(vault, tmp_path):
    target = tmp_path / "secret.md"
    target.write_text(f"{SECRET}\n", encoding="utf-8")
    link = vault / "2 Areas" / "Profile" / "pointer.md"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("this account can't make file symlinks (Windows without developer mode)")
    assert _reason(vault, "2 Areas/Profile/pointer.md") == "leads outside the vault"


@pytest.mark.parametrize("line", ["line {i:05d} of a long english note\n", "第{i:05d}行，很長的中文筆記\n"])
def test_long_notes_come_in_pieces_under_the_token_cap(vault, line):
    body = "".join(line.format(i=i) for i in range(6_000))
    write(vault, "2 Areas/People/long.md", render(PROFILE_META, body))
    whole = (vault / "2 Areas/People/long.md").read_text(encoding="utf-8")
    pieces, start = [], 0
    while True:
        out = access.read_note(vault, "2 Areas/People/long.md", start=start)
        assert out["start"] == start and chunking.est_tokens(out["text"]) <= access.PAGE_TOKENS
        pieces.append(out["text"])
        if not out["more"]:
            break
        start = out["end"]
    assert len(pieces) > 1 and "".join(pieces) == whole


def _age(vault: Path, rel: str, days: float) -> None:
    t = time.time() - days * 86400
    os.utime(vault / rel, (t, t))


def test_recent_is_newest_first_and_skips_every_exclusion(vault):
    for rel in [CHAT_REL, MEMORY_REL, PROFILE_REL, *OFF_LIMITS]:
        _age(vault, rel, 0.5)
    _age(vault, MEMORY_REL, 0.1)                 # the newest
    _age(vault, PROFILE_REL, 5)                  # outside the window
    got = access.recent(vault, days=2, limit=20)
    assert [r.path for r in got] == [MEMORY_REL, CHAT_REL]
    assert got[0].kind == "memory" and got[0].source == "memory: Garden"
    assert got[1].kind == "chat" and got[1].title == "home server plans" and got[1].source == "chatgpt"
    assert [r.path for r in access.recent(vault, days=10, limit=20)] == [MEMORY_REL, CHAT_REL, PROFILE_REL]
    assert [r.path for r in access.recent(vault, days=10, limit=1)] == [MEMORY_REL]
    assert not {r.path for r in access.recent(vault, days=10, limit=100)} & set(OFF_LIMITS)


def test_recent_leaves_out_research_notes_the_index_doesnt_cover(vault):
    assert RESEARCH_REL not in {r.path for r in access.recent(vault, days=1, limit=100)}


def test_hidden_and_broken_names_used_in_the_fixture_are_really_there(vault):
    for rel in HIDDEN_NOTES + [DIARY_TYPE_REL, EXCLUDED_META_REL, BROKEN_REL]:
        assert (vault / rel).is_file()
