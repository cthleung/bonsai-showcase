import json

import pytest

from capture import chatnote
from capture.chatnote import Chat, Turn, push_headings, render_body, write_chat
from common import notes


def make_chat(**kw):
    base = dict(source="chatgpt", chat_id="AbCdEfGh1234567", title="best ramen in sapporo",
                url="https://chatgpt.com/c/AbCdEfGh1234567", created="2026-11-10", updated="2026-11-12",
                turns=[Turn(prompt="where's good ramen?", answer="## Picks\n- Kumo", date="2026-11-10"),
                       Turn(prompt="open late?", answer="Yes.", date="2026-11-12", quote="Kumo",
                            sources=[{"url": "https://example.test/kumo", "title": "Kumo [menu]"}])],
                capture="chatgpt-export", model="gpt-test")
    base.update(kw)
    return Chat(**base)


def test_push_headings_keeps_order_and_skips_code():
    text = "# Top\ntext\n## Sub\n```\n# a comment in code\n```\n### Deeper"
    out = push_headings(text)
    assert out.splitlines() == ["#### Top", "text", "##### Sub", "```", "# a comment in code", "```", "###### Deeper"]
    assert push_headings("#### already\n##### fine") == "#### already\n##### fine"
    assert push_headings("no headings") == "no headings"


def test_render_body_shape():
    body = render_body(make_chat())
    assert body.startswith("# best ramen in sapporo\n")
    assert "## 1. You (2026-11-10)\n\nwhere's good ramen?\n\n### ChatGPT\n\n#### Picks\n- Kumo" in body
    assert "## 2. You (2026-11-12)\n\n> Kumo\n\nopen late?\n\n### ChatGPT\n\nYes." in body
    assert "- [Kumo (menu)](https://example.test/kumo)" in body


def test_write_chat_new_note(cfg):
    chat = make_chat()
    path, wrote = write_chat(cfg, chat, {"x": 1}, today="2026-11-12")
    assert wrote
    assert path == cfg.vault / "3 Resources/chats/chatgpt/2026/2026-11-10 best ramen in sapporo (AbCdEfGh).md"
    meta, body = notes.read_note(path)
    assert list(meta) == ["title", "source", "chat_id", "url", "created", "updated", "captured", "model",
                          "message_count", "capture", "raw", "visibility", "tags"]
    assert meta["visibility"] == "private" and meta["tags"] == ["chat"] and meta["message_count"] == 2
    assert meta["raw"] == "raw/chatgpt/AbCdEfGh1234567.json"
    assert (cfg.state / meta["raw"]).exists()
    assert not (cfg.vault / "raw").exists()


def test_raw_and_note_are_scrubbed(cfg):
    fake = "sk-" + "x" * 30
    chat = make_chat(turns=[Turn(prompt=f"my key is {fake}", answer="don't share it", date="2026-11-10")])
    path, _ = write_chat(cfg, chat, {"prompt": f"my key is {fake}"}, today="2026-11-12")
    assert fake not in path.read_text(encoding="utf-8")
    raw = (cfg.state / chatnote.raw_rel_path("chatgpt", chat.chat_id)).read_text(encoding="utf-8")
    assert fake not in raw and "REDACTED" in raw
    json.loads(raw)


def test_unchanged_chat_is_left_alone(cfg):
    path, _ = write_chat(cfg, make_chat(), {}, today="2026-11-12")
    before = path.read_text(encoding="utf-8")
    _, wrote = write_chat(cfg, make_chat(), {}, today="2026-11-13")
    assert not wrote and path.read_text(encoding="utf-8") == before


def test_force_rewrites_an_unchanged_note(cfg):
    write_chat(cfg, make_chat(), {}, today="2026-11-12")
    _, wrote = write_chat(cfg, make_chat(), {}, today="2026-11-13", force=True)
    assert wrote


def test_rewrite_keeps_file_name_and_hand_edits(cfg):
    path, _ = write_chat(cfg, make_chat(), {}, today="2026-11-12")
    meta, body = notes.read_note(path)
    meta["visibility"], meta["project"] = "career", "[[Food]]"
    notes.write_note(path, meta, body)
    more = make_chat(title="renamed chat", updated="2026-11-14",
                     turns=make_chat().turns + [Turn(prompt="thanks", answer="Any time.", date="2026-11-14")])
    path2, wrote = write_chat(cfg, more, {}, today="2026-11-14")
    assert wrote and path2 == path
    meta2, _ = notes.read_note(path2)
    assert meta2["visibility"] == "career" and meta2["project"] == "[[Food]]"
    assert meta2["message_count"] == 3 and meta2["updated"] == "2026-11-14" and meta2["title"] == "renamed chat"


def test_vault_text_follows_writing_rules():
    banned = [chr(0x2014), chr(0x2013), "half", "weight", "gravity", "structural", "load-bearing", "scaffolding",
              "framework", "architecture"]
    for text in chatnote.INTROS.values():
        for word in banned:
            assert word not in text.lower()


SECRET_BODY = "a-distinctive-payload-text-0451"


def test_an_unparseable_existing_note_is_never_overwritten(cfg):
    path, _ = write_chat(cfg, make_chat(), {}, today="2026-11-12")
    broken = f"---\ntitle: [unclosed {SECRET_BODY}\nchat_id: AbCdEfGh1234567\n---\nhand-written body\n".encode("utf-8")
    path.write_bytes(broken)
    raw = cfg.state / chatnote.raw_rel_path("chatgpt", "AbCdEfGh1234567")
    raw_before = raw.read_bytes()
    with pytest.raises(chatnote.NoteConflict):
        write_chat(cfg, make_chat(updated="2026-11-14"), {"new": 1}, today="2026-11-14")
    assert path.read_bytes() == broken and raw.read_bytes() == raw_before


def test_another_chats_note_at_the_path_is_never_overwritten(cfg):
    chat = Chat(source="chatgpt", chat_id="sameid8-AAAA", title="t", url="u", created="2026-11-18",
                updated="2026-11-18", turns=[Turn(prompt="p", answer="a")], capture="chatgpt-export")
    path, _ = write_chat(cfg, chat, {}, today="2026-11-18")
    before = path.read_bytes()
    other = Chat(**{**chat.__dict__, "chat_id": "sameid8-BBBB"})   # same first 8, same name
    with pytest.raises(chatnote.NoteConflict):
        write_chat(cfg, other, {}, today="2026-11-18")
    assert path.read_bytes() == before
    assert not (cfg.state / "raw" / "chatgpt" / "sameid8-BBBB.json").exists()


def test_raw_path_cleans_the_chat_id(cfg):
    assert chatnote.raw_rel_path("chatgpt", "../../evil:id/x") == "raw/chatgpt/evilidx.json"
    assert chatnote.raw_rel_path("chatgpt", "AbCd-ef_1234567890abcdef") == "raw/chatgpt/AbCd-ef_1234567890abcdef.json"
    assert chatnote.raw_rel_path("chatgpt", "::") == "raw/chatgpt/noid.json"
    chat = Chat(source="chatgpt", chat_id="../x:y", title="odd id", url="u", created="2026-11-18",
                updated="2026-11-18", turns=[Turn(prompt="p", answer="a")], capture="chatgpt-export")
    path, _ = write_chat(cfg, chat, {"k": 1}, today="2026-11-18")
    raw = cfg.state / "raw" / "chatgpt" / "xy.json"
    assert raw.exists() and json.loads(raw.read_text(encoding="utf-8")) == {"k": 1}
    assert notes.read_note(path)[0]["raw"] == "raw/chatgpt/xy.json"
    assert not list(cfg.state.parent.glob("*.json")) and not (cfg.state / "x").exists()
