"""The capture skip list: chats moved out of the vault (a person's or junk), which capture must never
bring back, even when the chat changes later.

    BONSAI_STATE/capture-skip.json
    {"chatgpt": {"<chat_id>": {"label": "junk", "when": "2026-11-19", "note": "4 Archive/..."}}, ...}

Whatever moves a chat out (by hand, or a sorting step) adds its id with add(), and remove() takes it
off again. The export importer skips a listed id before writing, and chatnote.write_chat refuses it
as a last guard.

A run on a copy of the vault keeps its own list in that copy's state folder (see dream/guards.py),
so a rehearsal never stops real capture. Capture only ever reads BONSAI_STATE/capture-skip.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from common.notes import write_text_atomic

FILE = "capture-skip.json"


def path(folder: Path) -> Path:
    return Path(folder) / FILE


def load(folder: Path) -> dict:
    """{source: {chat_id: info}}. A missing or unreadable file is an empty list."""
    try:
        data = json.loads(path(folder).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def ids(folder: Path, source: str) -> set[str]:
    return {str(k) for k in (load(folder).get(source) or {})}


def skipped(folder: Path, source: str, chat_id: str) -> bool:
    return str(chat_id) in ids(folder, source)


def save(folder: Path, data: dict) -> None:
    write_text_atomic(path(folder), json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n")


def add(folder: Path, source: str, chat_id: str, info: dict) -> bool:
    """Add one id. Returns False (and changes nothing) when it was already there."""
    data = load(folder)
    by_source = data.setdefault(source, {})
    if str(chat_id) in by_source:
        return False
    by_source[str(chat_id)] = info
    save(folder, data)
    return True


def remove(folder: Path, source: str, chat_id: str) -> bool:
    data = load(folder)
    if str(chat_id) not in (data.get(source) or {}):
        return False
    del data[source][str(chat_id)]
    if not data[source]:
        del data[source]
    save(folder, data)
    return True
