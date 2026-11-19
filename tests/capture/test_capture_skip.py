"""Chats moved out of the vault (to `4 Archive/chats-excluded/`, with their ids on the capture
skip list) never come back through capture: the skip list stops the export importer, and
write_chat leaves an archived note alone. Made-up chats, no network."""
import shutil

from capture import chatgpt_export
from capture.chatnote import Chat, Turn, write_chat
from common import capture_skip, notes
from tests.capture.test_chatgpt_export import ID_A, ID_B, export_conv, make_zip

JUNK = "conv-junkjunk-1"
KEEP = "conv-keepkeep-2"


def archive(cfg, source, cid, label="junk"):
    """Exclude one chat: move the note, set excluded, add the skip entry."""
    src = notes.find_chat_note(cfg.vault, source, cid)
    meta, body = notes.read_note(src)
    meta.update(excluded=label, sorted="2026-11-19")
    dest = cfg.vault / notes.EXCLUDED_DIR / label / src.name
    notes.write_note(dest, meta, body)
    src.unlink()
    capture_skip.add(cfg.state, source, cid, {"label": label, "when": "2026-11-19", "note": "x"})
    return dest


def chats_dir_files(cfg):
    return sorted((cfg.vault / notes.CHATS_DIR).rglob("*.md"))


def test_write_chat_leaves_an_archived_note_alone_without_the_skip_list(cfg):
    """BONSAI_STATE differs per computer: the Mac has no skip entry, but the note is in the archive."""
    chat = Chat("chatgpt", JUNK, "Junk", "https://x.invalid", "2026-11-10", "2026-11-10",
                [Turn("q", "a", "2026-11-10")], "chatgpt-export")
    write_chat(cfg, chat, {}, today="2026-11-10")
    dest = archive(cfg, "chatgpt", JUNK)
    capture_skip.remove(cfg.state, "chatgpt", JUNK)
    before = dest.read_bytes()
    chat.updated, chat.turns = "2026-11-15", chat.turns * 2
    path, wrote = write_chat(cfg, chat, {}, today="2026-11-15", force=True)
    assert path == dest and not wrote and dest.read_bytes() == before
    assert chats_dir_files(cfg) == []


def test_write_chat_on_the_skip_list_writes_nothing_even_with_no_note(cfg):
    capture_skip.add(cfg.state, "chatgpt", JUNK, {"label": "junk"})
    chat = Chat("chatgpt", JUNK, "Junk", "https://x.invalid", "2026-11-10", "2026-11-10",
                [Turn("q", "a", "2026-11-10")], "chatgpt-export")
    path, wrote = write_chat(cfg, chat, {}, today="2026-11-10", force=True)
    assert not wrote and not path.exists() and chats_dir_files(cfg) == []
    assert not (cfg.state / "raw").exists()


def test_a_rewrite_keeps_the_sorted_date(cfg):
    chat = Chat("chatgpt", KEEP, "Keep", "https://x.invalid", "2026-11-10", "2026-11-10",
                [Turn("q", "a", "2026-11-10")], "chatgpt-export")
    path, _ = write_chat(cfg, chat, {}, today="2026-11-10")
    meta, body = notes.read_note(path)
    notes.write_note(path, {**meta, "sorted": "2026-11-19"}, body)
    chat.updated, chat.turns = "2026-11-20", chat.turns * 2
    assert write_chat(cfg, chat, {}, today="2026-11-20")[1]
    assert notes.read_note(path)[0]["sorted"] == "2026-11-19"


def test_the_export_importer_skips_listed_chats(cfg, tmp_path):
    zip_path = make_zip(tmp_path / "export.zip", {"conversations.json": [
        export_conv(ID_A, "Junk", "2026-11-10"), export_conv(ID_B, "Keep", "2026-11-11")]})
    capture_skip.add(cfg.state, "chatgpt", ID_A, {"label": "junk"})
    report = chatgpt_export.import_export(cfg, zip_path, today="2026-11-19")
    assert report.counts["sorted-out"] == 1 and report.counts["new"] == 1
    assert notes.find_chat_note(cfg.vault, "chatgpt", ID_A) is None
    assert notes.find_chat_note(cfg.vault, "chatgpt", ID_B) is not None


def test_find_chat_note_finds_an_archived_note_and_checks_its_source(cfg):
    chat = Chat("chatgpt", JUNK, "Junk", "https://x.invalid", "2026-11-10", "2026-11-10",
                [Turn("q", "a", "2026-11-10")], "chatgpt-export")
    write_chat(cfg, chat, {}, today="2026-11-10")
    dest = archive(cfg, "chatgpt", JUNK)
    shutil.rmtree(cfg.vault / notes.CHATS_DIR)            # no chatgpt folder left at all
    assert notes.find_chat_note(cfg.vault, "chatgpt", JUNK) == dest
    assert notes.find_chat_note(cfg.vault, "claude", JUNK) is None
    assert notes.is_excluded(cfg.vault, dest)
