"""The ChatGPT export import, against small made-up exports built here (a ZIP or a folder).
No real export, no network."""
import json
import zipfile

import pytest

from capture import __main__ as cli
from capture import chatgpt_export
from capture.chatnote import Chat, Turn, write_chat
from capture.pull import Cursor
from common import notes
from tests.capture.test_chatgpt_parse import ts

ID_A = "aaaaaaaa-1111-4000-8000-000000000001"
ID_B = "bbbbbbbb-2222-4000-8000-000000000002"
ID_C = "cccccccc-3333-4000-8000-000000000003"
ID_BAD = "dddddddd-4444-4000-8000-000000000004"


def node(nid, parent, children, role=None, text=None, t=None, parts=None, metadata=None):
    n = {"id": nid, "parent": parent, "children": children, "message": None}
    if role:
        n["message"] = {"id": nid, "author": {"role": role}, "create_time": t,
                        "content": {"content_type": "text", "parts": parts or [text]},
                        "metadata": metadata or {}}
    return n


def export_conv(cid, title, day, turns=1, hour=12, **extra):
    """A conversation shaped like the export: the web backend's tree plus the export's own fields."""
    t = ts(day, hour)
    m = [node("root", None, ["u0"])]
    for i in range(turns):
        m.append(node(f"u{i}", "root" if i == 0 else f"a{i - 1}", [f"a{i}"], "user", f"question {i}", t))
        m.append(node(f"a{i}", f"u{i}", [f"u{i + 1}"] if i + 1 < turns else [], "assistant", f"answer {i}", t,
                      metadata={"model_slug": "gpt-test"}))
    conv = {"title": title, "create_time": ts(day, 9), "update_time": t,
            "mapping": {n["id"]: n for n in m}, "current_node": f"a{turns - 1}",
            "conversation_id": cid, "id": cid, "is_archived": False, "gizmo_id": None, "gizmo_type": None}
    conv.update(extra)
    return conv


def make_zip(path, files: dict, assets=()):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data if isinstance(data, str) else json.dumps(data))
        for name in assets:
            zf.writestr(name, b"fake bytes")
    return path


def split_export(tmp_path, a_day="2026-11-10", name="export.zip"):
    """Two shards and a manifest, the newer export layout."""
    a = export_conv(ID_A, "Ramen", a_day, turns=2)
    b = export_conv(ID_B, "Project chat", "2026-11-11", gizmo_id="g-p-123abc", gizmo_type="snorlax",
                    conversation_template_id="g-p-123abc")
    c = export_conv(ID_C, "Archived chat", "2026-11-09", is_archived=True)
    c["mapping"]["u0"]["message"]["metadata"]["attachments"] = [{"id": "file-x", "name": "menu.pdf"}]
    return make_zip(tmp_path / name, {
        "conversations-000.json": [a, b],
        "conversations-001.json": [c],
        "export_manifest.json": {"export_files": [{"path": "conversations-000.json"},
                                                  {"path": "conversations-001.json"}]},
        "chat.html": "<html></html>",
    }, assets=["file-AbC123-menu.pdf", "dalle-generations/pic.webp"])


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def note_of(cfg, cid):
    return notes.find_chat_note(cfg.vault, "chatgpt", cid)


# --- reading the export ----------------------------------------------------------------------

def test_split_files_are_all_read(cfg, tmp_path):
    report = chatgpt_export.import_export(cfg, split_export(tmp_path), today="2026-11-18")
    assert report.files == ["conversations-000.json", "conversations-001.json"]
    assert report.counts["new"] == 3 and report.total == 3
    for cid in (ID_A, ID_B, ID_C):
        meta, _ = notes.read_note(note_of(cfg, cid))
        assert meta["capture"] == "chatgpt-export" and meta["chat_id"] == cid
        assert meta["visibility"] == "private" and "project" not in meta
        assert (cfg.state / meta["raw"]).exists()
    assert not list(cfg.vault.rglob("*.zip")) and not list(cfg.vault.rglob("*.json"))


def test_shard_number_order_and_single_file_layout(tmp_path):
    names = ["x/conversations-010.json", "x/conversations-002.json", "x/conversations-1.json", "chat.html"]
    assert chatgpt_export.conversation_files(names) == [
        "x/conversations-1.json", "x/conversations-002.json", "x/conversations-010.json"]
    assert chatgpt_export.conversation_files(["conversations.json", "user.json"]) == ["conversations.json"]


def test_old_single_file_export_and_a_folder_both_work(cfg, tmp_path):
    folder = tmp_path / "unzipped"
    (folder / "nested").mkdir(parents=True)
    (folder / "nested" / "conversations.json").write_text(
        json.dumps([export_conv(ID_A, "Ramen", "2026-11-10")]), encoding="utf-8")
    report = chatgpt_export.import_export(cfg, folder, today="2026-11-18")
    assert report.counts["new"] == 1 and note_of(cfg, ID_A)


def test_note_body_matches_the_chat_note_format(cfg, tmp_path):
    chatgpt_export.import_export(cfg, split_export(tmp_path), today="2026-11-18")
    meta, body = notes.read_note(note_of(cfg, ID_A))
    assert meta["message_count"] == 2 and meta["model"] == "gpt-test" and meta["updated"] == "2026-11-10"
    assert meta["url"] == f"https://chatgpt.com/c/{ID_A}"
    assert "## 1. You (2026-11-10)\n\nquestion 0\n\n### ChatGPT\n\nanswer 0" in body
    _, archived = notes.read_note(note_of(cfg, ID_C))
    assert "[file: menu.pdf]" in archived


def test_keys_are_scrubbed_in_note_and_raw(cfg, tmp_path):
    fake = "sk-" + "y" * 30
    conv = export_conv(ID_A, "Keys", "2026-11-10")
    conv["mapping"]["u0"]["message"]["content"]["parts"] = [f"my key is {fake}"]
    chatgpt_export.import_export(cfg, make_zip(tmp_path / "e.zip", {"conversations.json": [conv]}), today="2026-11-18")
    path = note_of(cfg, ID_A)
    raw = cfg.state / notes.read_note(path)[0]["raw"]
    assert fake not in path.read_text(encoding="utf-8") and fake not in raw.read_text(encoding="utf-8")


# --- dedupe, reruns and never rolling back ---------------------------------------------------

def test_rerun_changes_nothing(cfg, tmp_path):
    zp = split_export(tmp_path)
    chatgpt_export.import_export(cfg, zp, today="2026-11-18")
    before = snapshot(cfg.vault), snapshot(cfg.state)
    report = chatgpt_export.import_export(cfg, zp, today="2026-11-19")
    assert report.counts["unchanged"] == 3 and report.counts["new"] == report.counts["updated"] == 0
    assert not report.cursor_changed
    assert (snapshot(cfg.vault), snapshot(cfg.state)) == before
    assert len(list((cfg.vault / "3 Resources/chats/chatgpt").rglob("*.md"))) == 3


def test_a_chat_in_two_shards_keeps_its_newest_copy(cfg, tmp_path):
    old = export_conv(ID_A, "Old title", "2026-11-10")
    new = export_conv(ID_A, "New title", "2026-11-12", turns=2)
    zp = make_zip(tmp_path / "e.zip", {"conversations-000.json": [new], "conversations-001.json": [old]})
    report = chatgpt_export.import_export(cfg, zp, today="2026-11-18")
    assert report.duplicates == 1 and report.counts["new"] == 1
    meta, _ = notes.read_note(note_of(cfg, ID_A))
    assert meta["title"] == "New title" and meta["message_count"] == 2


def test_older_export_never_rolls_back_a_newer_note(cfg, tmp_path):
    newer = make_zip(tmp_path / "new.zip", {"conversations.json": [export_conv(ID_A, "Ramen", "2026-11-12", turns=3)]})
    older = make_zip(tmp_path / "old.zip", {"conversations.json": [export_conv(ID_A, "Ramen", "2026-11-10", turns=1)]})
    chatgpt_export.import_export(cfg, newer, today="2026-11-18")
    path = note_of(cfg, ID_A)
    before = path.read_text(encoding="utf-8"), snapshot(cfg.state)
    report = chatgpt_export.import_export(cfg, older, today="2026-11-19")
    assert report.counts["skipped-older"] == 1 and report.counts["updated"] == 0
    assert (path.read_text(encoding="utf-8"), snapshot(cfg.state)) == before
    assert Cursor(cfg, "chatgpt").chats[ID_A] == ts("2026-11-12")


def test_older_export_never_rolls_back_a_newer_note(cfg, tmp_path):
    """A note written by an earlier capture on another computer: no cursor entry here, only the note."""
    live = Chat(source="chatgpt", chat_id=ID_A, title="Ramen live", url=f"https://chatgpt.com/c/{ID_A}",
                created="2026-11-10", updated="2026-11-14", capture="chatgpt-export",
                turns=[Turn(prompt=f"q{i}", answer=f"a{i}", date="2026-11-14") for i in range(4)])
    path, _ = write_chat(cfg, live, {"live": True}, today="2026-11-14")
    before = path.read_text(encoding="utf-8")
    report = chatgpt_export.import_export(
        cfg, make_zip(tmp_path / "e.zip", {"conversations.json": [export_conv(ID_A, "Ramen", "2026-11-12")]}),
        today="2026-11-18")
    assert report.counts["skipped-older"] == 1
    assert path.read_text(encoding="utf-8") == before
    assert ID_A not in Cursor(cfg, "chatgpt").chats


def test_same_day_with_no_cursor_entry_goes_by_turn_count(cfg, tmp_path):
    live = Chat(source="chatgpt", chat_id=ID_A, title="Ramen", url="u", created="2026-11-10",
                updated="2026-11-10", capture="chatgpt-export",
                turns=[Turn(prompt=f"q{i}", answer=f"a{i}", date="2026-11-10") for i in range(2)])
    write_chat(cfg, live, {}, today="2026-11-10")
    fewer = make_zip(tmp_path / "f.zip", {"conversations.json": [export_conv(ID_A, "Ramen", "2026-11-10", turns=1)]})
    more = make_zip(tmp_path / "m.zip", {"conversations.json": [export_conv(ID_A, "Ramen", "2026-11-10", turns=3)]})
    assert chatgpt_export.import_export(cfg, fewer, today="2026-11-18").counts["skipped-older"] == 1
    assert chatgpt_export.import_export(cfg, more, today="2026-11-18").counts["updated"] == 1
    assert notes.read_note(note_of(cfg, ID_A))[0]["message_count"] == 3


def test_newer_export_updates_and_keeps_the_existing_note_path_and_hand_edits(cfg, tmp_path):
    live = Chat(source="chatgpt", chat_id=ID_A, title="first title", url=f"https://chatgpt.com/c/{ID_A}",
                created="2026-11-10", updated="2026-11-10", capture="chatgpt-export",
                turns=[Turn(prompt="q", answer="a", date="2026-11-10")])
    path, _ = write_chat(cfg, live, {}, today="2026-11-10")
    meta, body = notes.read_note(path)
    meta["visibility"], meta["project"] = "career", "[[Food]]"
    notes.write_note(path, meta, body)
    zp = make_zip(tmp_path / "e.zip", {"conversations.json": [export_conv(ID_A, "renamed later", "2026-11-12", turns=2)]})
    report = chatgpt_export.import_export(cfg, zp, today="2026-11-18")
    assert report.counts["updated"] == 1
    assert note_of(cfg, ID_A) == path and len(list(path.parent.glob("*.md"))) == 1
    meta, _ = notes.read_note(path)
    assert meta["title"] == "renamed later" and meta["capture"] == "chatgpt-export"
    assert meta["visibility"] == "career" and meta["project"] == "[[Food]]" and meta["message_count"] == 2


# --- dry run ---------------------------------------------------------------------------------

def test_dry_run_writes_nothing_and_counts_like_the_real_run(cfg, tmp_path):
    zp = split_export(tmp_path)
    before = snapshot(cfg.vault), snapshot(cfg.state)
    dry = chatgpt_export.import_export(cfg, zp, dry_run=True, today="2026-11-18")
    assert (snapshot(cfg.vault), snapshot(cfg.state)) == before
    assert dry.lines()[0].startswith("Dry run, nothing written.")
    real = chatgpt_export.import_export(cfg, zp, today="2026-11-18")
    assert dry.counts == real.counts and dry.counts["new"] == 3


def test_cli_dry_run_writes_nothing(cfg, tmp_path, capsys):
    zp = split_export(tmp_path)
    before = snapshot(cfg.vault), snapshot(cfg.state)
    assert cli.main(["chatgpt-import", str(zp), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "3 new, 0 updated, 0 unchanged" in out and "Dry run" in out
    assert (snapshot(cfg.vault), snapshot(cfg.state)) == before


def test_cli_real_run_logs_and_releases_the_lock(cfg, tmp_path, capsys):
    assert cli.main(["chatgpt-import", str(split_export(tmp_path))]) == 0
    assert "chatgpt-export: 3 chats in the export, 3 new" in (cfg.state / "capture.log").read_text(encoding="utf-8")
    assert not (cfg.state / "capture.lock").exists()
    assert cli.main(["chatgpt-import", str(tmp_path / "missing.zip")]) == 1


# --- bad input -------------------------------------------------------------------------------

def test_malformed_conversations_are_skipped_and_reported(cfg, tmp_path):
    broken = export_conv(ID_BAD, "Broken", "2026-11-11")
    broken["mapping"] = "a-distinctive-payload-text"            # not a tree
    weird = export_conv(ID_C, "Weird", "2026-11-11")
    weird["mapping"]["a0"]["message"]["author"] = "not a dict"   # makes the parser raise
    zp = make_zip(tmp_path / "e.zip", {
        "conversations-000.json": [export_conv(ID_A, "Good", "2026-11-10"), broken, "not a conversation",
                                   {"title": "no id", "mapping": {}}],
        "conversations-001.json": "{ this is not json",
        "conversations-002.json": [weird, export_conv(ID_B, "Also good", "2026-11-12")],
    })
    report = chatgpt_export.import_export(cfg, zp, today="2026-11-18")
    assert report.counts["new"] == 2 and report.counts["failed"] == 4
    assert note_of(cfg, ID_A) and note_of(cfg, ID_B)
    assert not note_of(cfg, ID_BAD) and not note_of(cfg, ID_C)
    text = "\n".join(report.lines())
    assert "conversations-001.json (JSONDecodeError)" in text
    assert f"({ID_BAD}): no message tree" in text and "no conversation id" in text
    assert f"({ID_C}): AttributeError" in text
    assert "a-distinctive-payload-text" not in text

    # an unreadable file and an id-less chat can't be placed, so the watermark stays at None, and
    # with no watermark nothing is held either (the oldest hold would cut the listing short)
    c = Cursor(cfg, "chatgpt")
    assert set(c.chats) == {ID_A, ID_B}
    assert c.watermark is None and c.failures == {} and report.held == 0
    assert report.unplaced == 3 and "watermark stays where it was" in text


def test_failed_chats_with_id_and_time_are_held_and_the_watermark_moves(cfg, tmp_path):
    broken = export_conv(ID_BAD, "Broken", "2026-11-11")
    broken["mapping"] = "a-distinctive-payload-text"
    weird = export_conv(ID_C, "Weird", "2026-11-11")
    weird["mapping"]["a0"]["message"]["author"] = "not a dict"
    zp = make_zip(tmp_path / "e.zip", {"conversations.json": [
        export_conv(ID_A, "Good", "2026-11-10"), broken, weird, export_conv(ID_B, "Also good", "2026-11-12")]})
    report = chatgpt_export.import_export(cfg, zp, today="2026-11-18")
    assert report.counts["failed"] == 2 and report.unplaced == 0
    c = Cursor(cfg, "chatgpt")
    assert ID_BAD not in c.chats and ID_C not in c.chats
    assert set(c.failures) == {ID_BAD, ID_C} and report.held == 2
    assert c.floor() <= ts("2026-11-11") and c.watermark == ts("2026-11-12")


def test_a_failed_chat_with_no_time_keeps_the_watermark(cfg, tmp_path):
    timeless = export_conv(ID_BAD, "No time", "2026-11-11")
    timeless["update_time"] = timeless["create_time"] = None
    zp = make_zip(tmp_path / "e.zip", {"conversations.json": [export_conv(ID_A, "Good", "2026-11-10"), timeless]})
    report = chatgpt_export.import_export(cfg, zp, today="2026-11-18")
    assert report.counts["failed"] == 1 and report.unplaced == 1
    c = Cursor(cfg, "chatgpt")
    assert c.watermark is None and ID_A in c.chats and c.failures == {}


def test_an_unreadable_file_keeps_the_watermark(cfg, tmp_path):
    good = export_conv(ID_A, "Good", "2026-11-12")
    zp = make_zip(tmp_path / "e.zip", {"conversations-000.json": [good],
                                      "conversations-001.json": "[{ broken"})
    report = chatgpt_export.import_export(cfg, zp, today="2026-11-18")
    c = Cursor(cfg, "chatgpt")
    assert c.watermark is None and ID_A in c.chats and report.unplaced == 1
    assert note_of(cfg, ID_A)


def test_full_ids_sharing_the_first_8_characters_get_two_notes(cfg, tmp_path):
    one, two = "samepref-aaaa-0001", "samepref-bbbb-0002"
    zp = make_zip(tmp_path / "e.zip", {"conversations.json": [
        export_conv(one, "First chat", "2026-11-10"), export_conv(two, "Second chat", "2026-11-11")]})
    report = chatgpt_export.import_export(cfg, zp, today="2026-11-18")
    assert report.counts["new"] == 2
    assert note_of(cfg, one) != note_of(cfg, two)
    assert notes.read_note(note_of(cfg, two))[0]["title"] == "Second chat"
    assert chatgpt_export.import_export(cfg, zp, today="2026-11-19").counts["unchanged"] == 2


def test_a_bad_path_is_an_export_error(cfg, tmp_path):
    with pytest.raises(chatgpt_export.ExportError):
        chatgpt_export.import_export(cfg, tmp_path / "nope.zip")
    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(chatgpt_export.ExportError):
        chatgpt_export.import_export(cfg, tmp_path / "x.txt")
    with pytest.raises(chatgpt_export.ExportError):
        chatgpt_export.import_export(cfg, make_zip(tmp_path / "e.zip", {"user.json": {}}))


# --- coverage and the cursor afterwards ------------------------------------------------------

def test_coverage_report(cfg, tmp_path):
    report = chatgpt_export.import_export(cfg, split_export(tmp_path), today="2026-11-18")
    assert (report.projects, report.archived, report.gpts) == (1, 1, 0)
    assert report.attachments == 1 and report.asset_files == 2
    text = "\n".join(report.lines())
    assert "1 project chats" in text and "1 archived chats" in text and "temporary chats" in text
    for bad in (chr(0x2014), chr(0x2013), "ha" + "lf"):     # the vault writing rules
        assert bad not in text.lower()


def test_coverage_says_when_a_field_is_missing(cfg, tmp_path):
    conv = export_conv(ID_A, "Plain", "2026-11-10")
    for key in ("is_archived", "gizmo_id", "gizmo_type"):
        conv.pop(key)
    report = chatgpt_export.import_export(cfg, make_zip(tmp_path / "e.zip", {"conversations.json": [conv]}),
                                          today="2026-11-18")
    assert report.archived is None and report.projects is None
    assert "archived chats: field not in this export" in "\n".join(report.lines())


def test_the_cursor_records_what_the_export_covered(cfg, tmp_path):
    convs = [export_conv(ID_A, "Ramen", "2026-11-10"), export_conv(ID_B, "Other", "2026-11-11")]
    zp = make_zip(tmp_path / "e.zip", {"conversations.json": convs})
    chatgpt_export.import_export(cfg, zp, today="2026-11-18")
    c = Cursor(cfg, "chatgpt")
    assert c.watermark == ts("2026-11-11") and not c.first_run
    assert set(c.chats) == {ID_A, ID_B}
    before = c.path.read_bytes()
    chatgpt_export.import_export(cfg, zp, today="2026-11-19")        # a rerun changes nothing
    assert c.path.read_bytes() == before
