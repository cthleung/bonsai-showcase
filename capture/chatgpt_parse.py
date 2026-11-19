"""ChatGPT conversation JSON -> a Chat for capture/chatnote.py. Pure functions, no network.

The official data export (capture/chatgpt_export.py) holds each conversation as a `mapping` tree of
message nodes plus a `current_node`. These helpers walk that tree from `current_node` back to the
root (the branch last seen), drop the hidden nodes (system prompts, tool calls, reasoning), and
turn what's left into turns: a turn opens at each user message, and the visible assistant messages
after it are its answer.

Borrowed from pavel-voronin/chatgpt-sync (MIT License, Copyright (c) 2026 Pavel Voronin): walking a
conversation from `current_node` back to the root, and which nodes are hidden (is_hidden below is a
port of its isHiddenConversationNode). The note format is ours.
"""
from __future__ import annotations

import re

from capture.chatnote import Chat, Turn
from capture.pull import local_date

SOURCE = "chatgpt"

# ChatGPT marks citations and other inline widgets with private-use characters: drop them
_PRIVATE_USE = re.compile("[\ue000-\uf8ff]")
_CITE = re.compile("\ue200(?:cite|navlist|filecite|image_group|entity)\ue202.*?\ue201", re.S)


def current_path(conv: dict) -> list[dict]:
    """The nodes from the root to current_node: the branch last seen."""
    mapping = conv.get("mapping") or {}
    node, path, seen = mapping.get(conv.get("current_node") or ""), [], set()
    while isinstance(node, dict) and node.get("id") not in seen:
        seen.add(node.get("id"))
        path.append(node)
        node = mapping.get(node.get("parent") or "")
    if path:
        return path[::-1]
    roots = [n for n in mapping.values() if not n.get("parent")]   # no current_node: follow first children
    node, path = (roots[0] if roots else None), []
    while isinstance(node, dict) and node.get("id") not in seen:
        seen.add(node.get("id"))
        path.append(node)
        kids = node.get("children") or []
        node = mapping.get(kids[-1]) if kids else None
    return path


def is_hidden(node: dict) -> bool:
    msg = node.get("message") or {}
    role = (msg.get("author") or {}).get("role") or ""
    name = (msg.get("author") or {}).get("name") or ""
    recipient = msg.get("recipient") or ""
    ctype = (msg.get("content") or {}).get("content_type") or ""
    meta = msg.get("metadata") or {}
    if meta.get("is_visually_hidden_from_conversation"):
        return True
    if recipient == "api_tool.call_tool" or recipient in ("canmore.create_textdoc", "canmore.update_textdoc"):
        return True
    if role == "tool" and name in ("api_tool.call_tool", "canmore.create_textdoc", "canmore.update_textdoc"):
        return True
    if role in ("system", "user_editable_context"):
        return True
    if ctype in ("thoughts", "reasoning_recap", "model_editable_context", "code"):
        return True
    return False


def part_text(part) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    kind = part.get("content_type") or ""
    if kind == "image_asset_pointer":
        return "[image]"
    if kind == "file_asset_pointer" or part.get("file_name"):
        return f"[file: {part.get('file_name') or 'file'}]"
    if isinstance(part.get("text"), str):
        return part["text"]
    return ""


def message_text(msg: dict) -> str:
    content = msg.get("content") or {}
    parts = content.get("parts")
    if isinstance(parts, list):
        text = "\n\n".join(t for t in (part_text(p) for p in parts) if t and t.strip())
    else:
        text = content.get("text") or ""
    text = _CITE.sub("", text)
    text = _PRIVATE_USE.sub("", text)
    for att in (msg.get("metadata") or {}).get("attachments") or []:
        if att.get("name") and att["name"] not in text:
            text = f"{text}\n\n[file: {att['name']}]".strip()
    return text.strip()


def message_sources(msg: dict) -> list[dict]:
    out, seen = [], set()
    for ref in (msg.get("metadata") or {}).get("content_references") or []:
        items = (ref.get("items") or []) + ([ref] if ref.get("url") else [])
        for it in items:
            url = it.get("url")
            if url and url.startswith("http") and url not in seen:
                seen.add(url)
                out.append({"url": url, "title": it.get("title") or ""})
    return out


def turns_from_conversation(conv: dict) -> tuple[list[Turn], str]:
    """(turns, model). A turn opens at each user message; the visible assistant messages after it
    are its answer."""
    turns: list[Turn] = []
    model = "unknown"
    for node in current_path(conv):
        msg = node.get("message")
        if not msg or is_hidden(node):
            continue
        role = (msg.get("author") or {}).get("role")
        text = message_text(msg)
        if role == "user":
            turns.append(Turn(prompt=text, answer="", date=local_date(msg.get("create_time"))))
        elif role == "assistant":     # tool output (search results, code output) stays in the raw JSON only
            if not turns:
                turns.append(Turn(prompt="", answer="", date=local_date(msg.get("create_time"))))
            t = turns[-1]
            t.answer = f"{t.answer}\n\n{text}".strip() if text else t.answer
            t.sources += [s for s in message_sources(msg) if s not in t.sources]
            model = (msg.get("metadata") or {}).get("model_slug") or model
    return turns, model


def chat_from_conversation(chat_id: str, conv: dict, today: str, capture: str = "chatgpt-export",
                           archived: bool = False) -> Chat:
    turns, model = turns_from_conversation(conv)
    first = next((t.prompt.split("\n")[0] for t in turns if t.prompt), "")
    return Chat(source=SOURCE, chat_id=chat_id, url=f"https://chatgpt.com/c/{chat_id}",
                title=(conv.get("title") or first or chat_id).strip()[:200],
                created=local_date(conv.get("create_time")) or today,
                updated=local_date(conv.get("update_time")) or today,
                turns=turns, capture=capture, model=model,
                archived=bool(archived or conv.get("is_archived")))


