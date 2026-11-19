from common.notes import render
from dream import collect
from tests.dream.conftest import CHAT_REL, INBOX_REL, MEMORY_REL, chat_body, chat_meta, write


def test_collect_reads_chats_and_both_inboxes_not_the_digest(vault):
    write(vault, "0 Inbox/mac/from the mac.md", "A note from the Mac.\n")
    write(vault, "0 Inbox/digest 2026-11-17.md", "- [ ] add | x | y | source: [[z]] ^p01\n")
    write(vault, "3 Resources/chats/google-ai-mode/2026/raw.json", "{}")
    rels = {p.rel for p in collect.collect(vault, {})}
    assert rels == {CHAT_REL, INBOX_REL, "0 Inbox/mac/from the mac.md"}


def test_chat_piece_has_turns_and_visibility_from_the_note(vault):
    write(vault, CHAT_REL, render(chat_meta(visibility="career"), chat_body(3)))
    [chat] = [p for p in collect.collect(vault, {}) if p.kind == "chat"]
    assert chat.turns == (1, 3) and chat.turns_total == 3
    assert chat.visibility == "career"
    assert "question 1" in chat.text and "answer 3" in chat.text
    assert chat.link == "2026-11-10 ramen in sapporo (AbCdEfGh)"


def test_missing_or_odd_visibility_is_private(vault):
    write(vault, CHAT_REL, render(chat_meta(visibility="everyone"), chat_body(1)))
    pieces = collect.collect(vault, {})
    assert {p.visibility for p in pieces} == {"private"}


def test_seen_notes_are_skipped_and_new_turns_only_are_sent(vault):
    seen = collect.mark_seen({}, collect.collect(vault, {}))
    assert collect.collect(vault, seen) == []
    write(vault, CHAT_REL, render(chat_meta(updated="2026-11-12"), chat_body(5)))
    [p] = collect.collect(vault, seen)
    assert p.turns == (4, 5)
    assert "question 3" not in p.text and "question 4" in p.text


def test_captured_date_alone_does_not_count_as_a_change(vault):
    seen = collect.mark_seen({}, collect.collect(vault, {}))
    meta = chat_meta()
    meta["captured"] = "2026-11-17"
    write(vault, CHAT_REL, render(meta, chat_body(3)))
    assert collect.collect(vault, seen) == []


def test_edited_inbox_note_is_read_again(vault):
    seen = collect.mark_seen({}, collect.collect(vault, {}))
    write(vault, INBOX_REL, "Plant tomatoes along the north fence instead.\n")
    [p] = collect.collect(vault, seen)
    assert p.rel == INBOX_REL and "north fence" in p.text


def test_long_chat_splits_by_turn_and_the_cap_rolls_the_rest_over(vault):
    write(vault, CHAT_REL, render(chat_meta(), chat_body(10, size=400)))
    pieces = [p for p in collect.collect(vault, {}, max_piece=2_000) if p.kind == "chat"]
    assert len(pieces) > 1
    assert pieces[0].turns[0] == 1 and pieces[-1].turns[1] == 10
    batches, left = collect.pack(pieces, max_call=2_100, max_night=2 * 2_100)
    assert left, "the night cap leaves some pieces for later"
    read = [p for b in batches for p in b.pieces]
    seen = collect.mark_seen({}, read)
    nxt = [p for p in collect.collect(vault, seen, max_piece=2_000) if p.kind == "chat"]
    assert nxt[0].turns[0] == read[-1].turns[1] + 1


def test_pack_keys_sources_per_call(vault):
    batches, left = collect.pack(collect.collect(vault, {}))
    assert not left and len(batches) == 1
    assert [p.key for p in batches[0].pieces] == ["S1", "S2"]


def test_keys_in_notes_are_scrubbed_before_the_model_sees_them(vault):
    fake_key = "sk-ant-" + "Zq7" * 10
    write(vault, INBOX_REL, f"my key is {fake_key}\n")
    [p] = [p for p in collect.collect(vault, {}) if p.kind == "inbox"]
    assert fake_key not in p.text and "[REDACTED" in p.text


def test_existing_notes_and_homes(vault):
    (vault / "2 Areas" / "Health").mkdir()
    notes = collect.existing_notes(vault)
    assert [n["rel"] for n in notes] == [MEMORY_REL]
    assert notes[0]["description"] == "garden watering time"
    assert collect.homes(vault) == ["1 Projects/Garden/memory/", "2 Areas/Health/", "2 Areas/Profile/"]


def test_drafts_and_capture_notices_in_the_inbox_are_skipped(vault):
    write(vault, "0 Inbox/win/North Star.md", render({"title": "North Star", "status": "draft"}, "Draft words.\n"))
    write(vault, "0 Inbox/win/capture login needed - chatgpt 2026-11-18.md",
          render({"source": "capture", "tags": ["capture"]}, "Log in again.\n"))
    write(vault, "0 Inbox/mac/tagged notice.md", render({"tags": ["capture"]}, "Log in again.\n"))
    write(vault, "0 Inbox/win/finished.md", render({"title": "Finished", "status": "done"}, "Ready.\n"))
    rels = {p.rel for p in collect.collect(vault, {})}
    assert rels == {CHAT_REL, INBOX_REL, "0 Inbox/win/finished.md"}
    write(vault, "0 Inbox/win/North Star.md", render({"title": "North Star", "status": "ready"}, "Final words.\n"))
    assert "0 Inbox/win/North Star.md" in {p.rel for p in collect.collect(vault, {})}


def test_a_draft_chat_note_is_skipped_too(vault):
    meta = chat_meta()
    meta["status"] = "draft"
    write(vault, CHAT_REL, render(meta, chat_body(3)))
    assert CHAT_REL not in {p.rel for p in collect.collect(vault, {})}
