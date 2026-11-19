"""What the capture sources share: a cursor per source, API times, and how an error may be logged.

The cursor is BONSAI_STATE/cursors/<source>.json: {"watermark": <epoch seconds or null>,
"chats": {chat_id: <update time captured>}, "failures": {chat_id: {"tries": n, "updated": t}},
"gave_up": {...}, "last_run": "..."}. It is saved after every change, so a run that stops part way
keeps what it did.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from common import notes

MAX_TRIES = 3              # runs in a row a single chat may fail before it's given up on


class Cursor:
    def __init__(self, cfg, source: str):
        self.path = Path(cfg.state) / "cursors" / f"{source}.json"
        data = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        self.watermark: float | None = data.get("watermark")
        self.chats: dict = data.get("chats", {})
        self.failures: dict = data.get("failures", {})
        self.gave_up: dict = data.get("gave_up", {})

    @property
    def first_run(self) -> bool:
        return self.watermark is None and not self.chats

    def is_new(self, chat_id: str, updated: float | None) -> bool:
        seen = self.chats.get(chat_id)
        return seen is None or (updated is not None and updated > float(seen))

    def floor(self, skip=()) -> float | None:
        """Where listing may stop: the watermark, or the update time of the oldest chat that failed
        and hasn't been given up on, whichever is older. Failed chats in `skip` don't count."""
        times = [self.watermark] + [f.get("updated") for cid, f in self.failures.items() if cid not in skip]
        times = [t for t in times if t is not None]
        return min(times) if times else None

    def mark(self, chat_id: str, updated: float | None) -> None:
        self.chats[chat_id] = updated
        self.failures.pop(chat_id, None)
        if updated is not None and (self.watermark is None or updated > self.watermark):
            self.watermark = updated
        self.save()

    def fail(self, chat_id: str, updated: float | None) -> tuple[int, bool]:
        """Count a failed try. Returns (tries so far, given up). Giving up marks the chat done."""
        tries = int((self.failures.get(chat_id) or {}).get("tries", 0)) + 1
        if tries >= MAX_TRIES:
            self.gave_up[chat_id] = {"tries": tries, "updated": updated,
                                     "when": dt.datetime.now().isoformat(timespec="seconds")}
            self.mark(chat_id, updated)
            return tries, True
        self.failures[chat_id] = {"tries": tries, "updated": updated}
        self.save()
        return tries, False

    def hold(self, chat_id: str, updated: float | None) -> None:
        """Keep a failed chat in failures, so the floor stays at it, without counting a try."""
        tries = int((self.failures.get(chat_id) or {}).get("tries", 0))
        self.failures[chat_id] = {"tries": tries, "updated": updated}
        self.save()

    def save(self) -> None:
        notes.write_text_atomic(self.path, json.dumps({
            "watermark": self.watermark, "chats": self.chats, "failures": self.failures,
            "gave_up": self.gave_up, "last_run": dt.datetime.now().isoformat(timespec="seconds")}, indent=1))


def epoch(value) -> float | None:
    """An API time (epoch seconds, or an ISO string with or without a zone) -> epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.timestamp()


def local_date(value) -> str | None:
    e = epoch(value)
    return dt.datetime.fromtimestamp(e).date().isoformat() if e is not None else None


def error_kind(e: BaseException) -> str:
    """What a log line may say about an error: its class and HTTP status, never its text (a YAML or
    JSON error quotes the note or payload it choked on)."""
    status = getattr(e, "status", None)
    return type(e).__name__ + (f" HTTP {status}" if status else "")
