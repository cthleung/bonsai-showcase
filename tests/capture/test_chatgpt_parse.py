"""ChatGPT conversation parsing against a made-up conversation tree. No network."""
import datetime as dt

from capture import chatgpt_parse


def ts(day: str, hour: int = 12) -> float:
    return dt.datetime.fromisoformat(f"{day}T{hour:02d}:00:00").timestamp()


def node(nid, parent, children, role=None, text=None, t=None, **extra):
    n = {"id": nid, "parent": parent, "children": children}
    if role:
        meta = extra.pop("metadata", {})
        n["message"] = {"author": {"role": role}, "content": {"content_type": extra.pop("ctype", "text"),
                                                              "parts": [text]},
                        "create_time": t, "metadata": meta, **extra}
    return n


def conversation():
    # root -> system -> user1 -> asst1 -> user2 -> (asst2a old branch | asst2b current)
    m = [node("root", None, ["sys"]),
         node("sys", "root", ["u1"], "system", "you are helpful", ts("2026-11-10")),
         node("u1", "sys", ["a1"], "user", "best ramen in sapporo?", ts("2026-11-10")),
         node("a1", "u1", ["think", "u2"], "assistant", "## Picks\nKumo citeturn0search0",
              ts("2026-11-10"), metadata={"model_slug": "gpt-test",
                                          "content_references": [{"items": [{"url": "https://example.test/k",
                                                                             "title": "Kumo"}]}]}),
         node("think", "a1", [], "assistant", "secret thoughts", ts("2026-11-10"), ctype="thoughts"),
         node("u2", "a1", ["a2a", "a2b"], "user", "open late?", ts("2026-11-12")),
         node("a2a", "u2", [], "assistant", "old answer", ts("2026-11-12")),
         node("a2b", "u2", [], "assistant", "Yes, till 2.", ts("2026-11-12"), metadata={"model_slug": "gpt-test-2"})]
    return {"title": "Ramen", "create_time": ts("2026-11-10"), "update_time": ts("2026-11-12"),
            "mapping": {n["id"]: n for n in m}, "current_node": "a2b"}


def test_current_path_follows_current_node():
    ids = [n["id"] for n in chatgpt_parse.current_path(conversation())]
    assert ids == ["root", "sys", "u1", "a1", "u2", "a2b"]


def test_turns_skip_hidden_and_keep_sources():
    turns, model = chatgpt_parse.turns_from_conversation(conversation())
    assert [t.prompt for t in turns] == ["best ramen in sapporo?", "open late?"]
    assert turns[0].answer == "## Picks\nKumo"
    assert turns[1].answer == "Yes, till 2."
    assert turns[0].sources == [{"url": "https://example.test/k", "title": "Kumo"}]
    assert turns[0].date == "2026-11-10" and model == "gpt-test-2"


def test_is_hidden():
    assert chatgpt_parse.is_hidden(node("x", None, [], "system", "s"))
    assert chatgpt_parse.is_hidden(node("x", None, [], "assistant", "code", ctype="code"))
    assert chatgpt_parse.is_hidden(node("x", None, [], "assistant", "t", metadata={"is_visually_hidden_from_conversation": True}))
    assert not chatgpt_parse.is_hidden(node("x", None, [], "assistant", "hi"))


def test_part_text_placeholders():
    assert chatgpt_parse.part_text({"content_type": "image_asset_pointer", "asset_pointer": "file-service://x"}) == "[image]"
    assert chatgpt_parse.part_text({"content_type": "file_asset_pointer", "file_name": "a.pdf"}) == "[file: a.pdf]"


def test_chat_from_conversation_builds_the_note_fields():
    chat = chatgpt_parse.chat_from_conversation("AbCdEfGh1234567", conversation(), "2026-11-19")
    assert chat.source == "chatgpt" and chat.capture == "chatgpt-export"
    assert chat.url == "https://chatgpt.com/c/AbCdEfGh1234567" and chat.title == "Ramen"
    assert (chat.created, chat.updated) == ("2026-11-10", "2026-11-12") and len(chat.turns) == 2
    assert not chat.archived


def test_citation_marks_and_private_use_characters_are_dropped():
    msg = {"content": {"parts": ["Kumo \ue200cite\ue202turn0search0\ue201 is good\ue123"]}}
    assert chatgpt_parse.message_text(msg) == "Kumo  is good"
