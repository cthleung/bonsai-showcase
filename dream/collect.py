"""Gather what the night job reads: inbox notes and chat notes that are new or changed since the
last run, plus the existing memory notes and PARA homes the model needs to spot duplicates.

The watermark is per note, in <vault state>/state.json:

    {"watermark": "2026-11-18T02:00:05", "seen": {"<vault-relative path>": {"hash": ..., "turns_done": 12,
     "turns_total": 12}}}

A note is read again when its hash changes (frontmatter minus `captured`, plus body). For a chat note
only the turns after `turns_done` are sent, so a chat that grew by two turns costs two turns. A long
chat is split into pieces by turn; pieces the night's cap leaves out roll to the next night (the
watermark only moves after a night that read everything it found, so they stay inside the window).

Only new material is read. Inbox notes are always looked at. A chat note is looked at
only when it's new since the last finished night run (`since`, the watermark's date):

    captured on or after `since`              capture wrote it after the last run
    updated on or after `since` - 2 days      and the chat itself changed lately

The second rule is what keeps the backfill out. Capture's ChatGPT trickle writes old chats every
hour with a fresh `captured` date, so `captured` alone can't tell a 2023 chat from a new one. The
2 days cover capture's lag (it runs hourly) and a short capture outage. A chat captured since
`since` whose last update is older is history: the profile spike reads it, not the night job.
The first run (no watermark) looks back BONSAI_DREAM_FIRST_DAYS days, 3 by default.

Chats moved out of the vault are never read: anything with `excluded:` set,
and anything under `4 Archive/chats-excluded/` (that folder isn't in the read list at all).

`visibility` comes from the source note's frontmatter (capture sets it for chats), else "private".
The model never sees or sets it.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from common.notes import CHATS_DIR, EXCLUDED_DIR, parse
from common.scrub import scrub_text

INBOX_DIRS = ("0 Inbox/win", "0 Inbox/mac")
VISIBILITIES = ("private", "creative", "career", "public")
MAX_PIECE_CHARS = 100_000     # one piece of a note (about 25k tokens)
MAX_CALL_CHARS = 120_000      # sources in one claude call
MAX_NIGHT_CHARS = 400_000     # sources in one night, across calls
MAX_INDEX_NOTES = 500
FIRST_DAYS = 3                # how far back the first night run looks, without a watermark
UPDATED_LAG_DAYS = 2          # how long before `since` a chat's last update can be and still count
TURN_RX = re.compile(r"^## (\d+)\. You\b.*$", re.M)


@dataclass
class Piece:
    """One chunk of one source note, as the model will read it."""
    rel: str                     # vault-relative path, forward slashes
    kind: str                    # "chat" | "inbox"
    title: str
    link: str                    # file stem, for [[link]]
    visibility: str
    date: str                    # the note's updated/created date, or file date
    hash: str
    text: str
    turns: tuple[int, int] | None = None     # first and last turn in this piece
    turns_total: int | None = None
    key: str = ""                # S1, S2... set when packed into a call


@dataclass
class Batch:
    pieces: list[Piece] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(p.text) for p in self.pieces)


def rel_path(vault: Path, path: Path) -> str:
    return Path(path).relative_to(vault).as_posix()


def visibility_of(meta: dict) -> str:
    value = str(meta.get("visibility", "")).strip().lower()
    return value if value in VISIBILITIES else "private"


HASH_SKIP = ("captured", "sorted")   # set by scripts, not a change in what the note says


def note_hash(meta: dict, body: str) -> str:
    meta = {k: v for k, v in meta.items() if k not in HASH_SKIP}
    blob = yaml.safe_dump(meta, sort_keys=True, allow_unicode=True) + "\n---\n" + body
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _file_date(path: Path) -> str:
    return _dt.date.fromtimestamp(path.stat().st_mtime).isoformat()


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[cut here: {len(text) - limit} more characters not shown]\n"


def split_turns(body: str) -> list[tuple[int, str]]:
    """[(turn number, text of that turn)] from a chat note body. [] when it has no turn headings."""
    marks = list(TURN_RX.finditer(body))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out.append((int(m.group(1)), body[m.start():end].rstrip() + "\n"))
    return out


def skip_note(meta: dict) -> bool:
    """Notes dream leaves alone: any draft the user is still editing (`status: draft`, inbox or chat),
    capture's own notices (`source: capture` or a `capture` tag), and chats sorted out of the vault
    (`excluded:` set)."""
    tags = meta.get("tags") or []
    tags = [tags] if isinstance(tags, str) else tags
    return (bool(meta.get("excluded"))
            or str(meta.get("status", "")).strip().lower() == "draft"
            or str(meta.get("source", "")).strip().lower() == "capture"
            or "capture" in [str(t).lower() for t in tags])


def day_of(value) -> str:
    """'YYYY-MM-DD' from a frontmatter date (YAML may give a date object), or '' when there's none."""
    text = str(value or "").strip()[:10]
    return text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else ""


def since_date(watermark: str | None, today: _dt.date, first_days: int = FIRST_DAYS) -> str:
    """The first day a chat counts as new: the watermark's date, or `first_days` back on a first run."""
    day = day_of(watermark)
    return day or (today - _dt.timedelta(days=max(0, int(first_days)))).isoformat()


def chat_window(meta: dict, since: str) -> str:
    """'new' when the chat counts as new material since `since`, 'history' when capture wrote it
    since then but the chat itself is older (the backfill), 'old' when capture wrote it before.
    A note with no dates counts as new, so nothing is dropped for a missing field."""
    captured = day_of(meta.get("captured"))
    if captured and captured < since:
        return "old"
    updated = day_of(meta.get("updated")) or day_of(meta.get("created"))
    lag = (_dt.date.fromisoformat(since) - _dt.timedelta(days=UPDATED_LAG_DAYS)).isoformat()
    if updated and updated < lag:
        return "history"
    return "new"


def source_files(vault: Path) -> list[Path]:
    """Inbox notes (both machines) and chat notes. Only .md; the digest itself sits in 0 Inbox/,
    not in win/ or mac/, so it's never read back."""
    vault = Path(vault)
    files: list[Path] = []
    for d in INBOX_DIRS:
        root = vault / d
        if root.is_dir():
            files += sorted(p for p in root.rglob("*.md") if p.is_file())
    chats = vault / CHATS_DIR
    if chats.is_dir():
        files += sorted(p for p in chats.rglob("*.md") if p.is_file())
    excluded = (vault / EXCLUDED_DIR).as_posix() + "/"
    return [p for p in files if not p.name.startswith(".") and not p.as_posix().startswith(excluded)]


def _chat_pieces(rel, path, meta, body, h, seen_entry, max_piece) -> list[Piece]:
    turns = split_turns(body)
    title = str(meta.get("title") or path.stem)
    vis = visibility_of(meta)
    date = str(meta.get("updated") or meta.get("created") or _file_date(path))
    if not turns:   # no turn headings: treat it like a plain note
        return [Piece(rel, "chat", title, path.stem, vis, date, h, _cut(body, max_piece))]
    total = max(n for n, _ in turns)
    done = int(seen_entry.get("turns_done", 0)) if seen_entry else 0
    if done > total:   # the note was rewritten shorter: read it all again
        done = 0
    todo = [(n, t) for n, t in turns if n > done]
    head = (f"Chat: {title}\nsource: {meta.get('source', '')}, created {meta.get('created', '')}, "
            f"updated {meta.get('updated', '')}, {total} turns in all\n\n")
    pieces: list[Piece] = []
    cur: list[tuple[int, str]] = []
    size = len(head)

    def flush():
        if cur:
            text = head + "".join(t for _, t in cur)
            pieces.append(Piece(rel, "chat", title, path.stem, vis, date, h, text,
                                (cur[0][0], cur[-1][0]), total))

    for n, t in todo:
        t = _cut(t, max_piece - len(head))
        if cur and size + len(t) > max_piece:
            flush()
            cur, size = [], len(head)
        cur.append((n, t))
        size += len(t)
    flush()
    return pieces


def collect(vault: Path, seen: dict, max_piece: int = MAX_PIECE_CHARS, since: str | None = None,
            history: list[str] | None = None) -> list[Piece]:
    """Every piece of every source note that is new or changed since `seen`. With `since` (a
    'YYYY-MM-DD'), chat notes outside the window (chat_window) are left out, and the ones that are
    history (captured lately, but old) go on `history` when a list is given, for the log."""
    vault = Path(vault)
    out: list[Piece] = []
    for path in source_files(vault):
        rel = rel_path(vault, path)
        try:
            meta, body = parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
            continue
        h = note_hash(meta, body)
        if skip_note(meta):
            continue
        entry = seen.get(rel)
        is_chat = rel.startswith(CHATS_DIR.as_posix() + "/")
        if is_chat and since:
            window = chat_window(meta, since)
            if window != "new":
                if window == "history" and history is not None:
                    history.append(rel)
                continue
        if is_chat:
            if entry and entry.get("hash") == h and entry.get("turns_done", 0) >= (entry.get("turns_total") or 0):
                continue
            out += _chat_pieces(rel, path, meta, body, h, entry, max_piece)
        else:
            if entry and entry.get("hash") == h:
                continue
            title = str(meta.get("title") or path.stem)
            date = str(meta.get("date") or meta.get("updated") or _file_date(path))
            text = f"Inbox note: {rel}\n\n" + path.read_text(encoding="utf-8")
            out.append(Piece(rel, "inbox", title, path.stem, visibility_of(meta), date, h,
                             _cut(text, max_piece)))
    for p in out:
        p.text = scrub_text(p.text)   # keys never go to the model either
    return out


def pack(pieces: list[Piece], max_call: int = MAX_CALL_CHARS, max_night: int = MAX_NIGHT_CHARS) -> tuple[list[Batch], list[Piece]]:
    """Group pieces into calls. Returns (batches, left over for another night). Pieces of one chat
    stay in order: once one piece of a chat is left over, the later ones are too."""
    batches: list[Batch] = []
    left: list[Piece] = []
    blocked: set[str] = set()
    total = 0
    cur = Batch()
    for p in pieces:
        if p.rel in blocked or total + len(p.text) > max_night:
            left.append(p)
            blocked.add(p.rel)
            continue
        if cur.pieces and cur.chars + len(p.text) > max_call:
            batches.append(cur)
            cur = Batch()
        cur.pieces.append(p)
        total += len(p.text)
    if cur.pieces:
        batches.append(cur)
    for b in batches:
        for i, p in enumerate(b.pieces, 1):
            p.key = f"S{i}"
    return batches, left


def mark_seen(seen: dict, pieces: list[Piece]) -> dict:
    """Record pieces as read. Call only for pieces whose call succeeded."""
    for p in pieces:
        if p.turns:   # pieces of one chat arrive in turn order, so the last one read wins
            seen[p.rel] = {"hash": p.hash, "turns_done": p.turns[1], "turns_total": p.turns_total}
        else:
            seen[p.rel] = {"hash": p.hash, "turns_done": 0, "turns_total": 0}
    return seen


# --- what's already in the vault ------------------------------------------------------------

def memory_files(vault: Path) -> list[Path]:
    vault = Path(vault)
    files: list[Path] = []
    projects = vault / "1 Projects"
    if projects.is_dir():
        for mem in sorted(projects.glob("*/memory")):
            files += sorted(p for p in mem.rglob("*.md") if p.is_file())
    areas = vault / "2 Areas"
    if areas.is_dir():
        files += sorted(p for p in areas.rglob("*.md") if p.is_file())
    return [p for p in files if not p.name.startswith(".")]


def existing_notes(vault: Path, limit: int = MAX_INDEX_NOTES) -> list[dict]:
    """[{rel, name, type, description, visibility, snippet}] for memory notes, for duplicate checks."""
    out = []
    for path in memory_files(vault)[:limit]:
        try:
            meta, body = parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
            continue
        snippet = " ".join(body.split())[:200]
        out.append({
            "rel": rel_path(vault, path),
            "name": str(meta.get("name") or path.stem),
            "type": str(meta.get("type") or ""),
            "description": scrub_text(str(meta.get("description") or "")),
            "visibility": visibility_of(meta),
            "snippet": scrub_text(snippet),
        })
    return out


def homes(vault: Path) -> list[str]:
    """Folders a new memory note can go in: each project's memory folder, each area, and Profile."""
    vault = Path(vault)
    out = []
    if (vault / "1 Projects").is_dir():
        out += [f"1 Projects/{d.name}/memory/" for d in sorted((vault / "1 Projects").iterdir())
                if d.is_dir() and not d.name.startswith(".")]
    if (vault / "2 Areas").is_dir():
        out += [f"2 Areas/{d.name}/" for d in sorted((vault / "2 Areas").iterdir())
                if d.is_dir() and not d.name.startswith(".")]
    if "2 Areas/Profile/" not in out:
        out.append("2 Areas/Profile/")
    return out
