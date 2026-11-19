"""Import ChatGPT history from the official data export (chatgpt.com, Settings > Data controls > Export).

    python -m capture chatgpt-import PATH [--dry-run]     PATH: the export ZIP, or the folder it unzipped to

The export brings in the whole ChatGPT history in one go, read from a file on this computer: no
network, no login.

Each conversation (a `mapping` tree plus `current_node`) goes through
chatgpt_parse.chat_from_conversation and chatnote.write_chat, like every other source: same note
path, same key scrubbing, same raw JSON file under BONSAI_STATE, with `capture: chatgpt-export`.

The export: older ones have one `conversations.json`; newer ones split it into `conversations-000.json`,
`conversations-001.json` and so on (some tools see `conversations-0.json` too), plus an
`export_manifest.json`, the attachment files and a `chat.html` viewer. Every file named like that,
at any depth, is read. A ZIP is read in place with zipfile: nothing is extracted or copied, so the
export never lands in the vault or in git.

What it does with each chat:
- Dedupes by full conversation id (a chat in two files keeps its newest copy), and finds an existing
  note with notes.find_chat_note, so a chat imported before keeps its note and file name.
- Never rolls a note back. The note's `updated` day is compared first (the vault is shared between
  computers, so the note is the truth); on the same day, this computer's cursor gives the exact time;
  with no cursor entry, the export rewrites the note only when it has more turns. See classify().
- A chat that won't parse is skipped and reported (by its id and the error's kind, never its text),
  and the rest go on.

The cursor (BONSAI_STATE/cursors/chatgpt.json, the shape in capture/pull.py) is updated once, at
the end, from what was actually written or confirmed: each such chat's time goes into `chats`, and
the watermark moves up to the newest of them, so anything that reads the cursor later (another
import, or a puller that lists chats by update time) knows where the vault stands. A chat that
failed here and that a chat list would show (not archived, not in a project) is held in
`failures`, so the cursor's floor stays at it. Chats the import skipped as older are left out of
the cursor, so their newer time there stays. When something in the export can't be placed (a file
that won't read, a failed chat with no id or no time), the watermark doesn't move. A rerun of the
same export changes nothing, the cursor file included.

--dry-run reads the export and the vault and reports the same counts, and writes nothing at all.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from capture import chatgpt_parse
from capture.chatnote import write_chat
from capture.pull import Cursor, epoch, error_kind, local_date
from common import capture_skip, notes

SOURCE = chatgpt_parse.SOURCE
CAPTURE = "chatgpt-export"
SAME = 1.0          # seconds: update times this close are the same capture (list ISO times vs float times)
OUTCOMES = ("new", "updated", "unchanged", "skipped-older", "sorted-out", "empty", "failed")
_CONV_FILE = re.compile(r"^conversations(?:-(\d+))?\.json$")

NOT_IN_EXPORT = [
    "temporary chats and deleted chats (the export doesn't have them)",
    "project names and project files (a project chat carries only the project's id, so notes get no `project:`)",
    "attachment and image files (notes show a [file: name] or [image] line; the files stay in the export)",
    "other branches of a chat (the note follows the branch last seen; the raw JSON keeps the whole tree)",
]


# --- reading the export ----------------------------------------------------------------------

class ExportError(RuntimeError):
    """The path isn't an export we can read (missing, not a ZIP or folder, no conversation files)."""


@contextlib.contextmanager
def open_export(path: Path | str):
    """Yields (names, read): every file in the export as a relative posix path, and read(name) ->
    bytes. A ZIP is read in place, never extracted."""
    path = Path(path)
    if path.is_dir():
        names = sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())
        yield names, lambda name: (path / name).read_bytes()
    elif path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = sorted(i.filename for i in zf.infolist() if not i.is_dir())
            yield names, zf.read
    elif path.exists():
        raise ExportError(f"not a ZIP file or a folder: {path}")
    else:
        raise ExportError(f"no such file or folder: {path}")


def conversation_files(names: list[str]) -> list[str]:
    """`conversations.json` and `conversations-NNN.json` at any depth, in number order."""
    found = []
    for name in names:
        m = _CONV_FILE.match(PurePosixPath(name).name)
        if m:
            found.append((str(PurePosixPath(name).parent), int(m.group(1)) if m.group(1) else -1, name))
    return [name for _, _, name in sorted(found)]


def is_asset(name: str) -> bool:
    """A file in the export that isn't data or the viewer: an attachment, an image, a project file."""
    return not name.lower().endswith((".json", ".html"))


@dataclass
class Item:
    chat_id: str | None
    updated: float | None
    conv: dict | None
    where: str
    problem: str | None = None


def conversation_id(conv: dict) -> str | None:
    cid = conv.get("conversation_id") or conv.get("id")
    return cid.strip() if isinstance(cid, str) and cid.strip() else None


def check_conversation(conv, where: str) -> Item:
    if not isinstance(conv, dict):
        return Item(None, None, None, where, "not a conversation object")
    cid = conversation_id(conv)
    t = epoch(conv.get("update_time")) if not isinstance(conv.get("update_time"), bool) else None
    if t is None:
        t = epoch(conv.get("create_time")) if not isinstance(conv.get("create_time"), bool) else None
    if cid is None:
        return Item(None, t, None, where, "no conversation id")
    where = f"{where} ({cid})"
    if not isinstance(conv.get("mapping"), dict) or not conv.get("mapping"):
        return Item(cid, t, conv, where, "no message tree")
    if t is None:
        return Item(cid, None, conv, where, "no update_time or create_time")
    return Item(cid, t, conv, where)


def read_items(names: list[str], read, report: "Report") -> list[Item]:
    """Every conversation in the export, deduped by id (newest copy kept). Bad items are kept, with
    their problem, so they get reported and held; a file that won't parse is reported and skipped."""
    best: dict[str, Item] = {}
    bad: list[Item] = []
    for name in conversation_files(names):
        report.files.append(name)
        try:
            data = json.loads(read(name).decode("utf-8-sig"))
        except Exception as e:   # noqa: BLE001  (one bad file must not stop the others)
            report.bad_files.append(f"{name} ({error_kind(e)})")
            continue
        if not isinstance(data, list):
            report.bad_files.append(f"{name} (not a list of conversations)")
            continue
        for n, conv in enumerate(data):
            item = check_conversation(conv, f"{PurePosixPath(name).name} item {n}")
            if item.chat_id is None:
                bad.append(item)
                continue
            old = best.get(item.chat_id)
            if old is not None:
                report.duplicates += 1
                if old.problem is None and (item.problem or (item.updated or 0) <= (old.updated or 0)):
                    continue
            best[item.chat_id] = item
    return bad + sorted(best.values(), key=lambda i: (i.updated or 0, i.chat_id))


# --- coverage --------------------------------------------------------------------------------

def is_project_chat(conv: dict) -> bool:
    return (conv.get("gizmo_type") == "snorlax"
            or str(conv.get("conversation_template_id") or "").startswith("g-p-")
            or str(conv.get("gizmo_id") or "").startswith("g-p-"))


def is_gpt_chat(conv: dict) -> bool:
    gizmo = str(conv.get("gizmo_id") or "")
    return gizmo.startswith("g-") and not is_project_chat(conv)


def attachment_refs(conv: dict) -> int:
    """Attachments and images referenced anywhere in the chat's tree (every branch)."""
    n = 0
    for node in (conv.get("mapping") or {}).values():
        msg = node.get("message") if isinstance(node, dict) else None
        if not isinstance(msg, dict):
            continue
        n += len((msg.get("metadata") or {}).get("attachments") or [])
        for part in ((msg.get("content") or {}).get("parts") or []):
            if isinstance(part, dict) and (part.get("asset_pointer") or part.get("content_type") in
                                           ("image_asset_pointer", "file_asset_pointer")):
                n += 1
    return n


@dataclass
class Report:
    dry_run: bool = False
    files: list[str] = field(default_factory=list)
    bad_files: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=lambda: dict.fromkeys(OUTCOMES, 0))
    problems: list[str] = field(default_factory=list)       # "<where>: <what>", never chat text
    duplicates: int = 0
    total: int = 0
    archived: int | None = None             # None: the field never appears in this export
    projects: int | None = None
    gpts: int | None = None
    attachments: int = 0
    asset_files: int = 0
    held: int = 0
    no_id: int = 0                          # failed chats with no id: the cursor can't hold them
    unplaced: int = 0                       # see update_cursor
    cursor_changed: bool = False
    watermark_moved: bool = False

    def count(self, conv: dict) -> None:
        self.total += 1
        if "is_archived" in conv:
            self.archived = (self.archived or 0) + bool(conv.get("is_archived"))
        if any(k in conv for k in ("gizmo_type", "gizmo_id", "conversation_template_id")):
            self.projects = (self.projects or 0) + is_project_chat(conv)
            self.gpts = (self.gpts or 0) + is_gpt_chat(conv)
        try:
            self.attachments += attachment_refs(conv)
        except Exception:   # noqa: BLE001  (coverage is a count; a strange tree doesn't stop the import)
            pass

    def summary(self) -> str:
        c = self.counts
        return (f"{c['new']} new, {c['updated']} updated, {c['unchanged']} unchanged, "
                f"{c['skipped-older']} skipped (older than the note), {c['sorted-out']} sorted out (on the skip list), "
                f"{c['empty']} empty, {c['failed']} failed")

    def lines(self) -> list[str]:
        def field_count(n, what):
            return f"{n} {what}" if n is not None else f"{what}: field not in this export"
        out = [("Dry run, nothing written. " if self.dry_run else "") +
               f"ChatGPT export: {len(self.files)} conversation file(s), {self.total} chats "
               f"({self.duplicates} duplicate copies dropped)",
               f"  {self.summary()}",
               "Coverage:",
               f"  {field_count(self.projects, 'project chats')} (imported like any other chat)",
               f"  {field_count(self.gpts, 'custom GPT chats')} (imported like any other chat)",
               f"  {field_count(self.archived, 'archived chats')} (imported like any other chat)",
               f"  {self.attachments} attachment or image references in the chats, "
               f"{self.asset_files} attachment files in the export (not copied)",
               "Not in the notes:"] + [f"  {line}" for line in NOT_IN_EXPORT]
        if self.bad_files:
            out += ["Files that couldn't be read:"] + [f"  {f}" for f in self.bad_files]
        if self.problems:
            out += ["Chats skipped:"] + [f"  {p}" for p in self.problems]
        if not self.dry_run:
            out.append(f"Cursor: {'updated' if self.cursor_changed else 'unchanged'}"
                       + (f", {self.held} failed chat(s) held in it for a later run" if self.held else ""))
        if self.unplaced:
            out.append(f"Cursor watermark {'would stay' if self.dry_run else 'stays'} where it was: "
                       f"{self.unplaced} file(s) or chat(s) couldn't be placed (an unreadable file, or a "
                       f"failed chat with no id or time)")
        return out


# --- deciding and writing --------------------------------------------------------------------

def classify(cfg, cursor: Cursor, chat_id: str, t: float, n_turns: int) -> str:
    """new | updated | unchanged | skipped-older, from the note and this computer's cursor.

    The note is checked first: the vault is shared between computers, and its `updated` day is the
    truth about what's there. Export day before the note's day: older. After: updated. On the same
    day, the cursor's exact time decides when this computer has one. With no cursor entry, the
    export rewrites the note only when it has more turns (the day alone can't tell two captures on
    the same day apart, and fewer turns would roll the note back)."""
    path = notes.find_chat_note(cfg.vault, SOURCE, chat_id)
    known = cursor.chats.get(chat_id)
    known = float(known) if known is not None else None
    if path is None:
        if known is not None and t < known - SAME:
            return "skipped-older"
        return "new"
    meta, _ = notes.read_note(path)
    note_day, export_day = str(meta.get("updated") or ""), local_date(t) or ""
    if note_day and export_day < note_day:
        return "skipped-older"
    if note_day and export_day > note_day:
        return "updated"
    if known is not None:
        if t > known + SAME:
            return "updated"
        return "skipped-older" if t < known - SAME else "unchanged"
    try:
        count = int(meta.get("message_count"))
    except (TypeError, ValueError):
        count = 0
    if n_turns > count:
        return "updated"
    return "skipped-older" if n_turns < count else "unchanged"


def can_be_listed(conv: dict | None) -> bool:
    """Whether ChatGPT's main chat list would show this chat (not archived, not a project chat).
    Only those are worth holding the cursor's floor at: a puller that lists by update time reaches
    them, and archived and project chats are listed some other way."""
    return isinstance(conv, dict) and not conv.get("is_archived") and not is_project_chat(conv)


def import_export(cfg, path: Path | str, dry_run: bool = False, today: str | None = None) -> Report:
    """Import every chat in the export. Raises ExportError when there's nothing to import."""
    today = today or dt.date.today().isoformat()
    report = Report(dry_run=dry_run)
    cursor = Cursor(cfg, SOURCE)
    skip = capture_skip.ids(cfg.state, SOURCE)   # sorted out of the vault: never written again
    with open_export(path) as (names, read):
        if not conversation_files(names):
            raise ExportError(f"no conversations.json or conversations-NNN.json in {path}")
        report.asset_files = sum(1 for n in names if is_asset(n))
        items = read_items(names, read, report)

    done: dict[str, float] = {}             # written or confirmed in the vault at this time
    held: dict[str, float | None] = {}      # failed here; held in the cursor for a later run
    for item in items:
        if isinstance(item.conv, dict):
            report.count(item.conv)
        else:
            report.total += 1
        if item.problem:
            report.counts["failed"] += 1
            report.problems.append(f"{item.where}: {item.problem}")
            if not item.chat_id:
                report.no_id += 1
            elif can_be_listed(item.conv):
                held[item.chat_id] = item.updated
            continue
        try:
            chat = chatgpt_parse.chat_from_conversation(item.chat_id, item.conv, today, capture=CAPTURE)
            if item.chat_id in skip:
                outcome = "sorted-out"
            elif not chat.turns:
                outcome = "empty"
            else:
                outcome = classify(cfg, cursor, item.chat_id, item.updated, len(chat.turns))
                if outcome in ("new", "updated") and not dry_run:
                    write_chat(cfg, chat, item.conv, today=today, force=True)
        except Exception as e:   # noqa: BLE001  (one bad chat must not stop the rest)
            report.counts["failed"] += 1
            report.problems.append(f"{item.where}: {error_kind(e)}")
            if can_be_listed(item.conv):
                held[item.chat_id] = item.updated
            continue
        report.counts[outcome] += 1
        if outcome != "skipped-older":
            done[item.chat_id] = item.updated

    report.unplaced = len(report.bad_files) + report.no_id + sum(
        1 for cid, t in held.items() if t is None and cid not in cursor.chats)
    if not dry_run:
        report.held, report.cursor_changed, report.watermark_moved = update_cursor(
            cfg, cursor, done, held, report.unplaced)
    return report


def update_cursor(cfg, cursor: Cursor, done: dict[str, float], held: dict[str, float | None],
                  unplaced: int = 0) -> tuple[int, bool, bool]:
    """Record what the import wrote or confirmed, once, at the end. Returns (chats held, changed,
    watermark moved).

    - `chats[id]` goes up to the export's time for every chat in `done` (never down).
    - A chat that failed here and that a chat list would show is held in `failures` with its
      time, so the floor (Cursor.floor) never moves past it. A failed project or archived chat is
      only reported (see can_be_listed). A failed chat already in `chats` isn't held: the vault has
      it, and a hold would keep the floor down for good.
    - The watermark goes up to the newest time in `done` (never down), but only when every chat in
      the export was placed. `unplaced` counts what can't be: a conversation file that wouldn't
      read, a failed chat with no id, a failed chat a chat list would show but with no time to hold
      it at. Then the watermark stays where it was. When it stays at None, no chat is held either
      (the floor would be the oldest hold and cut any listing there).
    - Nothing is saved when nothing changed, so a rerun leaves the cursor file as it was."""
    changed, n_held, moved = False, 0, False
    for cid, t in done.items():
        known = cursor.chats.get(cid)
        if known is None or t > float(known) + SAME:
            cursor.chats[cid] = t
            changed = True
        if cid in cursor.failures:
            cursor.failures.pop(cid)
            changed = True
    if done and not unplaced:
        newest = max(done.values())
        if cursor.watermark is None or newest > cursor.watermark + SAME:
            cursor.watermark = newest
            changed = moved = True
    if cursor.watermark is not None:
        for cid, t in held.items():
            if cid in cursor.chats or cid in done or t is None:
                continue
            n_held += 1
            if cid not in cursor.failures:
                cursor.failures[cid] = {"tries": 0, "updated": t}
                changed = True
    if changed:
        cursor.save()
    return n_held, changed, moved
