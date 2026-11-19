"""One captured chat -> one vault note (docs/note-format.md) plus its raw JSON under BONSAI_STATE.

    chat = Chat(source="chatgpt", chat_id="...", title="...", url="...", created="2026-11-10",
                updated="2026-11-12", turns=[Turn(prompt="...", answer="...", date="2026-11-10")],
                capture="chatgpt-export")
    path, wrote = write_chat(cfg, chat, raw_obj)

Rules this follows:
- The file name is fixed by chat_id: an existing note is found with notes.find_chat_note and kept,
  even if the chat was renamed. A file at the new note's path that find_chat_note didn't confirm
  (frontmatter that won't parse, or another chat's id) is never written over: NoteConflict.
- A note is rewritten only when `updated` or `message_count` changed. `captured` changes on every
  run, so without this check every hourly run would rewrite every note and the sync client would
  upload them all again.
- visibility is set here, never by a model: `private` for a new note. On a rewrite, the note's own
  `visibility`, `project` and `tags` are kept, so an edit the user made by hand survives.
- The raw JSON is scrubbed and goes to BONSAI_STATE/raw/<source>/<chat_id>.json; `raw:` holds that
  path relative to BONSAI_STATE (the state folder differs between the PC and the Mac).
- A chat moved out of the vault (to `4 Archive/chats-excluded/`) is never written again: its id is on the
  capture skip list (common/capture_skip.py), or its note sits in `4 Archive/chats-excluded/` or
  has `excluded:` set. The skip list lives in BONSAI_STATE, which differs per computer, so the
  note check covers the other computer. A kept chat's `sorted:` date survives a rewrite.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from common import capture_skip, notes
from common.scrub import scrub_obj

SOURCE_NAMES = {"google-ai-mode": "AI Mode", "chatgpt": "ChatGPT", "claude": "Claude"}
INTROS = {
    "google-ai-mode": "Google AI Mode chat, captured automatically. The answers are Google's text, kept as written.",
    "chatgpt": "ChatGPT chat, captured automatically. The answers are ChatGPT's text, kept as written.",
    "claude": "Claude chat, captured automatically. The answers are Claude's text, kept as written.",
}
DEFAULT_VISIBILITY = "private"


@dataclass
class Turn:
    prompt: str
    answer: str
    date: str | None = None                 # "YYYY-MM-DD"
    quote: str | None = None                # the bit of an answer the user quoted back, if any
    sources: list[dict] = field(default_factory=list)   # [{"url": ..., "title": ...}]


@dataclass
class Chat:
    source: str
    chat_id: str
    title: str
    url: str
    created: str
    updated: str
    turns: list[Turn]
    capture: str                            # how it was captured: chatgpt-export | specstory | codex-rollout
    model: str = "unknown"
    project: str | None = None              # e.g. "[[Food]]"
    archived: bool = False                  # ChatGPT: the chat is archived there (the key is written only when true)


_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def raw_rel_path(source: str, chat_id: str) -> str:
    """`raw/<source>/<chat_id>.json`, the id cleaned to [A-Za-z0-9_-] the way common/notes.py cleans
    it for file names (full length here), so an id like `../x` or `a:b` can't leave the raw folder."""
    return f"raw/{source}/{_ID_CHARS.sub('', str(chat_id)) or 'noid'}.json"


# --- markdown -------------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})(\s)", re.M)
_FENCE = re.compile(r"^\s*(```|~~~)")


def push_headings(text: str, top: int = 4) -> str:
    """Shift the answer's headings so its biggest one is `top` (####), keeping their order.
    Lines inside code fences are left alone."""
    lines = text.split("\n")
    fenced, levels = False, []
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and (m := _HEADING.match(line)):
            levels.append(len(m.group(1)))
    if not levels or min(levels) >= top:
        return text
    shift = top - min(levels)
    out, fenced = [], False
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and (m := _HEADING.match(line)):
            line = "#" * min(6, len(m.group(1)) + shift) + line[len(m.group(1)):]
        out.append(line)
    return "\n".join(out)


def render_body(chat: Chat) -> str:
    who = SOURCE_NAMES.get(chat.source, chat.source)
    md = [f"# {chat.title}", "", INTROS.get(chat.source, f"{who} chat, captured automatically."), ""]
    for n, t in enumerate(chat.turns, 1):
        md += [f"## {n}. You ({t.date or chat.updated})", ""]
        if t.quote:
            md += [f"> {line}" if line else ">" for line in t.quote.split("\n")] + [""]
        if t.prompt:
            md += [t.prompt.strip(), ""]
        md += [f"### {who}", "", push_headings(t.answer.strip()) or "(no answer text)", ""]
        if t.sources:
            md += ["#### Sources", ""]
            for s in t.sources:
                label = (s.get("title") or "").replace("[", "(").replace("]", ")").strip() or s["url"]
                md.append(f"- [{label}]({s['url']})")
            md.append("")
    return "\n".join(md).rstrip() + "\n"


def build_meta(chat: Chat, captured: str, existing: dict | None = None) -> dict:
    existing = existing or {}
    meta = {
        "title": chat.title,
        "source": chat.source,
        "chat_id": chat.chat_id,
        "url": chat.url,
        "created": chat.created,
        "updated": chat.updated,
        "captured": captured,
        "model": chat.model or "unknown",
    }
    project = existing.get("project") or chat.project
    if project:
        meta["project"] = project
    if chat.archived:
        meta["archived"] = True
    meta.update({
        "message_count": len(chat.turns),
        "capture": chat.capture,
        "raw": raw_rel_path(chat.source, chat.chat_id),
        "visibility": existing.get("visibility") or DEFAULT_VISIBILITY,
        "tags": existing.get("tags") or ["chat"],
    })
    if existing.get("sorted"):   # the date the user checked the chat; a rewrite keeps it
        meta["sorted"] = existing["sorted"]
    return meta


def needs_rewrite(existing: dict, chat: Chat) -> bool:
    return (str(existing.get("updated")) != chat.updated
            or str(existing.get("message_count")) != str(len(chat.turns))
            or bool(existing.get("archived")) != chat.archived)


class NoteConflict(RuntimeError):
    """A file already sits where this chat's note would go, and it isn't confirmed as this chat's
    note (its frontmatter won't parse, or its chat_id is another chat's). Capture never writes over
    it; the caller counts a failed try for the chat."""


def write_chat(cfg, chat: Chat, raw_obj, today: str | None = None, force: bool = False) -> tuple[Path, bool]:
    """Write (or leave alone) the note for this chat. Returns (note path, whether it was written).
    force=True rewrites an unchanged note too (e.g. to bring an older note into the current shape).
    Raises NoteConflict, and writes nothing (not even the raw JSON), when the target file exists but
    find_chat_note didn't confirm it as this chat's note. A chat that was sorted out (on the skip
    list, or its note archived or marked `excluded:`) is left alone, force or not, and nothing is
    written."""
    captured = today or dt.date.today().isoformat()
    path = notes.find_chat_note(cfg.vault, chat.source, chat.chat_id)
    if capture_skip.skipped(cfg.state, chat.source, chat.chat_id):
        return path or notes.chat_note_path(cfg.vault, chat.source, chat.created, chat.title, chat.chat_id), False
    existing: dict = {}
    if path:
        existing, _ = notes.read_note(path)
        if notes.is_excluded(cfg.vault, path, existing):
            return path, False
        if not force and not needs_rewrite(existing, chat):
            return path, False
    else:
        path = notes.chat_note_path(cfg.vault, chat.source, chat.created, chat.title, chat.chat_id)
        if path.exists():
            raise NoteConflict("a file that isn't this chat's note is already at the note's path")
    raw_path = Path(cfg.state) / raw_rel_path(chat.source, chat.chat_id)
    notes.write_text_atomic(raw_path, json.dumps(scrub_obj(raw_obj), ensure_ascii=False, indent=1))
    notes.write_note(path, build_meta(chat, captured, existing), render_body(chat))
    return path, True
