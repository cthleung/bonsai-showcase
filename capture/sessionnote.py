"""What the local agent-session sources share (capture/claude_code.py, capture/codex.py): the note,
the per-file cursor, the note finder, scrubbing and the run report.

These sources read files on this computer (SpecStory's Markdown, Codex's rollout JSONL) and write one
note per session to `3 Resources/chats/<source>/<folder>/<YYYY-MM-DD> <title> (<id8>).md`, where the
folder is the repo for claude-code and the year for codex. The note follows docs/note-format.md, with
three changes for local sessions:
- `raw:` points at the source file (or a list of files) on this computer. Nothing is copied: the
  SpecStory history alone is well over 100 MB, and the transcripts stay where the agent keeps them.
- `agent:` names the agent that ran it, and for codex `host:` the computer the thread ran on.
  claude-code leaves `host` out: SpecStory history syncs through git, so the computer reading it
  isn't where the session ran.
- There's no `url:`, since a local session has none.

A note makes a session searchable and readable by agents on every device. It doesn't make the session
resumable in the agent on another computer.

Rewrites: a session's note is rendered again whenever one of its files changed (the cursor below), and
written only when the text differs from the file, `captured`, `raw` and `host` aside (the PC and the
Mac read the same SpecStory history at different paths, so they'd take turns rewriting it). So a run with no new records
writes nothing, and a turn that grew mid-answer still gets filled in. The note's own `visibility`,
`project`, `tags` and `sorted` survive a rewrite, as in capture/chatnote.py. A session sorted out of
the vault (the skip list, `4 Archive/chats-excluded/`, or `excluded:`) is never written again.

The cursor is BONSAI_STATE/cursors/<source>-<host>.json:
    {"host": "...", "files": {"<path>": {"mtime": ..., "size": ..., "id": "<session id>"}},
     "sessions": {"<session id>": "<note path in the vault>"}, "last_run": "..."}
A session's files are marked only after its note was written (or found unchanged), so a session that
failed is read again on the next run, and a source that stops part way keeps what it did.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from capture.chatnote import NoteConflict, Turn, push_headings
from common import capture_skip, notes
from common.scrub import load_secrets, scrub_obj, scrub_text

# Credentials with no fixed prefix, which common/scrub.py can't catch by shape: an AWS secret key, a
# bearer token, and `password = ...` style pairs (a value counts only when it has a digit, so
# `password = os.environ` stays). Agent sessions print config and commands, so whatever these match
# is redacted before every write (redact_loose) and counted in the run's log line. Group 1 is kept,
# group 2 is the value.
LOOSE = [
    ("AWS_SECRET_KEY", re.compile(r"(?i)(aws_secret_access_key[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9/+=]{40})")),
    ("BEARER_TOKEN", re.compile(r"(?i)(\bbearer\s+)(?!\[REDACTED)([A-Za-z0-9_\-.=+/~]{20,})")),
    ("SECRET_VALUE", re.compile(
        r"(?i)(\b(?:api[_-]?key|secret(?:[_-]?key)?|client[_-]?secret|access[_-]?token|auth[_-]?token|"
        r"refresh[_-]?token|session[_-]?token|password|passwd|pwd)\b[\"']?\s*[:=]\s*[\"']?)"
        r"(?!\[REDACTED)(?=[^\s\"'`,;)\]}]*\d)([^\s\"'`,;)\]}]{8,})")),
]


def redact_loose(text: str) -> tuple[str, int]:
    """(text with every LOOSE match's value redacted, how many)."""
    total = 0
    for name, rx in LOOSE:
        text, n = rx.subn(lambda m, name=name: f"{m.group(1)}[REDACTED:{name}]", text)
        total += n
    return text, total


def clean(text: str, secrets: dict) -> tuple[str, int]:
    """Scrub by value and shape (common/scrub.py), then redact the loose shapes. (text, loose hits)."""
    return redact_loose(scrub_text(text, secrets))


def clean_obj(obj, secrets: dict, hits: list):
    if isinstance(obj, str):
        text, n = clean(obj, secrets)
        hits.append(n)
        return text
    if isinstance(obj, dict):
        return {k: clean_obj(v, secrets, hits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_obj(v, secrets, hits) for v in obj]
    return obj


UNTITLED = "untitled session"


def clean_title(title: str, secrets: dict) -> str:
    """The title, or UNTITLED when it holds anything key-shaped: it becomes the file name and the
    cursor's note path, which nothing scrubs later. A SpecStory slug has its hyphens turned into
    spaces, so a key there only shows its shape once the spaces are hyphens again."""
    for form in (title, title.replace(" ", "-")):
        text, _ = clean(form, secrets)
        if text != form:
            return UNTITLED
    return title


def host_name() -> str:
    """This computer's name, lowercased, safe for a file name."""
    name = platform.node() or os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown"
    return re.sub(r"[^a-z0-9_-]", "-", name.lower()).strip("-") or "unknown"


def posix(path: Path | str) -> str:
    return Path(path).as_posix()


def local_date(when: dt.datetime | None) -> str | None:
    """A UTC (or zoned) time -> this computer's local date, YYYY-MM-DD."""
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return when.astimezone().date().isoformat()


def short_title(text: str, max_len: int = 60) -> str:
    """The first line of a message, cut at a word to max_len, as a note title."""
    for line in (text or "").splitlines():
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)        # [label](url) -> label
        line = re.sub(r"[#*_`>]+", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            if len(line) > max_len:
                cut = line[:max_len]
                line = cut.rsplit(" ", 1)[0] if " " in cut else cut
            return line.rstrip(" .,;:")
    return ""


# --- the answer: text, reasoning and collapsed tool calls ------------------------------------

def tool_marker(names: list[str]) -> str:
    """One line for a run of tool calls: `*[3 tool calls: Bash x2, Read]*`."""
    counts = Counter(names)
    listed = ", ".join(f"{n} x{c}" if c > 1 else n for n, c in counts.items())
    what = "tool call" if len(names) == 1 else f"{len(names)} tool calls"
    return f"*[{what}: {listed}]*"


def quote_block(text: str, label: str | None = None) -> str:
    lines = text.strip().split("\n")
    if label:
        lines[0] = f"*{label}:* {lines[0]}"
    return "\n".join(f"> {line}" if line.strip() else ">" for line in lines)


SIDE_MARK = "*[a subagent worked:"


def side_marker(steps: int) -> str:
    return f"{SIDE_MARK} {steps} step{'s' if steps != 1 else ''}]*"


def render_parts(parts: list[tuple[str, str]]) -> str:
    """parts: ("text", s) the agent's reply, ("think", s) a reasoning summary, ("tool", name) a tool
    call, ("side", "") a subagent's step, ("mark", s) a one-line marker (a file change, a compaction).
    A run of tool calls becomes one marker line, a run of subagent steps another (the two interleave,
    so a run is broken only by text), and the same marker twice in a row shows once."""
    out: list[str] = []
    tools: list[str] = []
    side = [0]

    def flush():
        if tools:
            out.append(tool_marker(tools))
            tools.clear()
        if side[0]:
            out.append(side_marker(side[0]))
            side[0] = 0
    for kind, value in parts:
        if kind == "tool":
            tools.append(value or "tool")
            continue
        if kind == "side":
            side[0] += 1
            continue
        value = (value or "").strip()
        if not value:              # an empty reply between two tool calls doesn't split their run
            continue
        flush()
        if kind == "think":
            value = quote_block(value, "thinking")
        if out and out[-1] == value and kind == "mark":
            continue
        out.append(value)
    flush()
    return "\n\n".join(out)


# --- one session -> one note ------------------------------------------------------------------

@dataclass
class Session:
    source: str                 # claude-code | codex (the folder under 3 Resources/chats/)
    chat_id: str                # the session id
    title: str
    created: str                # YYYY-MM-DD, local
    updated: str
    turns: list[Turn]
    agent: str                  # claude-code | copilot | copilot-cli | codex-cli | codex
    agent_name: str             # the answer heading: "Claude Code", "Copilot", "Codex"
    folder: str                 # under the source folder: the repo (claude-code) or the year (codex)
    raw: list[str]              # the source files on this computer, never copied
    visibility: str             # set by the script for a new note: by repo, never by a model
    host: str | None            # the computer the session ran on, when the source says (codex); None leaves it out
    capture: str                # specstory | codex-rollout
    intro: str
    model: str = "unknown"
    extra: dict = field(default_factory=dict)   # more frontmatter (repo, cwd), after host


def render_body(s: Session) -> str:
    md = [f"# {s.title}", "", s.intro, ""]
    for n, t in enumerate(s.turns, 1):
        md += [f"## {n}. You ({t.date or s.updated})", ""]
        if t.quote:
            md += [quote_block(t.quote), ""]
        if t.prompt:
            md += [t.prompt.strip(), ""]
        md += [f"### {s.agent_name}", "", push_headings(t.answer.strip()) or "(no answer text)", ""]
    return "\n".join(md).rstrip() + "\n"


def build_meta(s: Session, captured: str, existing: dict | None = None) -> dict:
    existing = existing or {}
    meta = {
        "title": s.title,
        "source": s.source,
        "chat_id": s.chat_id,
        "created": s.created,
        "updated": s.updated,
        "captured": captured,
        "model": s.model or "unknown",
        "agent": s.agent,
    }
    if s.host:
        meta["host"] = s.host
    meta.update({k: v for k, v in s.extra.items() if v not in (None, "")})
    if existing.get("project"):
        meta["project"] = existing["project"]
    meta.update({
        "message_count": len(s.turns),
        "capture": s.capture,
        "raw": s.raw[0] if len(s.raw) == 1 else list(s.raw),
        "visibility": existing.get("visibility") or s.visibility,
        "tags": existing.get("tags") or ["chat"],
    })
    if existing.get("sorted"):
        meta["sorted"] = existing["sorted"]
    return meta


class NoteIndex:
    """Finds a session's existing note by its ` (<id8>).md` suffix, confirmed by chat_id, like
    notes.find_chat_note, but listing the folders once per run instead of once per session (the
    vault sits in a synced folder, and a SpecStory history can hold hundreds of sessions)."""

    _SUFFIX = re.compile(r"\(([A-Za-z0-9_-]{1,8})\)\.md$")

    def __init__(self, vault: Path, source: str):
        self.vault, self.source = Path(vault), source
        self.by_id8: dict[str, list[tuple[Path, bool]]] = defaultdict(list)
        self.planned: set[str] = set()
        for root, archived in ((self.vault / notes.CHATS_DIR / source, False), (self.vault / notes.EXCLUDED_DIR, True)):
            if root.is_dir():
                for path in sorted(root.glob("*/*.md")):
                    m = self._SUFFIX.search(path.name)
                    if m:
                        self.by_id8[m.group(1)].append((path, archived))

    def find(self, chat_id: str) -> Path | None:
        for path, archived in self.by_id8.get(notes.chat_id8(chat_id), []):
            try:
                meta, _ = notes.read_note(path)
            except (OSError, ValueError, UnicodeDecodeError, yaml.YAMLError):
                continue
            if str(meta.get("chat_id")) != str(chat_id):
                continue
            if archived and str(meta.get("source")) != self.source:
                continue
            return path
        return None

    def add(self, path: Path, chat_id: str) -> None:
        self.by_id8[notes.chat_id8(chat_id)].append((path, False))

    # A dry run writes nothing, so it keeps the paths it would have written here, to find the same
    # name clashes a real run would.
    def plan(self, path: Path) -> None:
        self.planned.add(str(path).lower())

    def taken(self, path: Path) -> bool:
        return str(path).lower() in self.planned


def note_path(vault: Path, s: Session, tell_apart: bool = False) -> Path:
    """The new note's path. tell_apart=True adds the end of the id after the title: a Codex id is a
    UUID v7, whose first 8 characters are its start time, so sessions started within the same
    minute (parallel subagents, say) can share date, title and id8. The ` (<id8>).md` ending
    stays, so the note is still found by its id."""
    folder = notes.safe_filename(s.folder, 60)
    title = s.title
    if tell_apart:
        title = f"{s.title} {notes._ID_CHARS.sub('', s.chat_id)[-6:]}"
    return Path(vault) / notes.CHATS_DIR / s.source / folder / notes.chat_note_name(s.created, title, s.chat_id)


def write_session(cfg, s: Session, index: NoteIndex, secrets: dict, today: str,
                  dry_run: bool = False) -> tuple[str, Path, str, int]:
    """(outcome, note path, note text, loose redactions). outcome is "new", "updated", "unchanged" or
    "sorted-out". A dry run returns the same and writes nothing. Raises NoteConflict when a file that
    isn't this session's note sits at the new note's path."""
    s = replace(s, title=clean_title(s.title, secrets))
    path = index.find(s.chat_id)
    if capture_skip.skipped(cfg.state, s.source, s.chat_id):
        return "sorted-out", path or note_path(cfg.vault, s), "", 0
    existing: dict = {}
    old_text = None
    if path:
        old_text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        existing, _ = notes.parse(old_text)
        if notes.is_excluded(cfg.vault, path, existing):
            return "sorted-out", path, "", 0
    body, hits = clean(render_body(s), secrets)

    def text_for(captured: str, keep_origin: bool = False) -> tuple[str, int]:
        meta = build_meta(s, captured, existing)
        if keep_origin:          # compare with the note's own raw and host: a change of reader alone isn't news
            for key in ("raw", "host"):
                if key in existing and key in meta:
                    meta[key] = existing[key]
        counted: list = []
        meta = clean_obj(meta, secrets, counted)
        return notes.render(meta, body), sum(counted)
    if path and old_text == text_for(str(existing.get("captured") or today), keep_origin=True)[0]:
        return "unchanged", path, old_text, 0
    text, meta_hits = text_for(today)
    outcome = "updated" if path else "new"
    if not path:
        path = note_path(cfg.vault, s)
        if path.exists() or (dry_run and index.taken(path)):
            path = note_path(cfg.vault, s, tell_apart=True)
            if path.exists() or (dry_run and index.taken(path)):
                raise NoteConflict("a file that isn't this session's note is already at the note's path")
        if dry_run:
            index.plan(path)
    if not dry_run:
        meta, body_again = notes.parse(text)
        notes.write_note(path, meta, body_again)       # scrubs again with the default secrets, writes safely
        index.add(path, s.chat_id)
    return outcome, path, text, hits + meta_hits


SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".specstory", ".mypy_cache", ".pytest_cache"}


def env_files(repo: Path, max_depth: int = 5) -> list[Path]:
    """A repo's `.env` files at any depth up to max_depth (a tool in a subfolder can keep its own),
    plus `.claude/settings.local.json`. Skips .git, node_modules and the like."""
    repo = Path(repo)
    found = [repo / ".claude" / "settings.local.json"]
    if not repo.is_dir():
        return []
    base = len(repo.parts)
    for root, dirs, files in os.walk(repo):
        depth = len(Path(root).parts) - base
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and depth < max_depth]
        if ".env" in files:
            found.append(Path(root) / ".env")
    return [f for f in found if f.is_file()]


def secrets_for(repos: list[Path]) -> dict:
    """{value: NAME}: Bonsai's own secrets plus the ones in each configured repo's `.env` files and
    `.claude/settings.local.json`, so a value pasted or printed in a session is caught by value, not
    only by shape. Only repos from Bonsai's own settings are read, never a path a transcript names."""
    found = load_secrets()
    files = []
    for repo in repos:
        files += env_files(repo)
    found.update(load_secrets(files))
    return found


def count_suspects(text: str) -> int:
    """LOOSE shapes still in a note after redaction (0 unless redact_loose missed one)."""
    return sum(len(rx.findall(text)) for _, rx in LOOSE)


# --- cursor -----------------------------------------------------------------------------------

class FileCursor:
    def __init__(self, cfg, source: str, host: str | None = None):
        self.host = host or host_name()
        self.path = Path(cfg.state) / "cursors" / f"{source}-{self.host}.json"
        data = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            except ValueError:
                data = {}
        self.files: dict = data.get("files") or {}
        self.sessions: dict = data.get("sessions") or {}

    @staticmethod
    def stamp(path: Path) -> dict:
        st = path.stat()
        return {"mtime": st.st_mtime, "size": st.st_size}

    def changed(self, key: str, stamp: dict) -> bool:
        seen = self.files.get(key)
        return not seen or seen.get("mtime") != stamp["mtime"] or seen.get("size") != stamp["size"]

    def known_id(self, key: str) -> str | None:
        return (self.files.get(key) or {}).get("id")

    def mark(self, files: dict[str, dict], session_id: str, note: str | None) -> None:
        for key, stamp in files.items():
            self.files[key] = dict(stamp, id=session_id)
        if note:
            self.sessions[session_id] = note

    def save(self) -> None:
        notes.write_text_atomic(self.path, json.dumps({
            "host": self.host, "files": self.files, "sessions": self.sessions,
            "last_run": dt.datetime.now().isoformat(timespec="seconds")}, indent=1))


# --- the run report ---------------------------------------------------------------------------

@dataclass
class Report:
    source: str
    dry_run: bool = False
    files: int = 0              # source files found
    changed_files: int = 0      # new or changed since the cursor
    sessions: int = 0           # sessions found across all files
    read: int = 0               # sessions read this run (they had a new or changed file)
    counts: Counter = field(default_factory=Counter)      # new, updated, unchanged, sorted-out, empty, failed, ...
    kinds: Counter = field(default_factory=Counter)       # sessions by agent
    note_bytes: int = 0         # the notes this run would write (new and updated)
    biggest: tuple = (0, "")
    source_bytes: int = 0       # the source files read
    redacted: int = 0           # loose key-shaped strings (LOOSE) redacted in the notes this run wrote
    suspects: int = 0           # LOOSE shapes still there after redaction (should stay 0)
    malformed: int = 0          # unreadable records (codex: bad JSON lines)
    notes_: list = field(default_factory=list)            # extra lines for the report

    def add_note(self, outcome: str, path: Path, text: str, redacted: int = 0) -> None:
        self.counts[outcome] += 1
        if outcome in ("new", "updated"):
            self.redacted += redacted
            size = len(text.encode("utf-8"))
            self.note_bytes += size
            self.suspects += count_suspects(text)
            if size > self.biggest[0]:
                self.biggest = (size, path.name)

    def summary(self) -> str:
        c = self.counts
        return (f"{self.files} files ({self.changed_files} new or changed), {self.sessions} sessions, "
                f"{self.read} read: {c['new']} new, {c['updated']} updated, {c['unchanged']} unchanged, "
                f"{c['sorted-out']} sorted out, {c['empty']} empty, {c['skipped']} skipped, {c['failed']} failed; "
                f"{self.redacted} key-shaped strings redacted")

    def lines(self) -> list[str]:
        mb = lambda n: f"{n / 1_000_000:.1f} MB"   # noqa: E731
        out = [("Dry run, nothing written. " if self.dry_run else "") + f"{self.source}: {self.summary()}"]
        if self.kinds:
            out.append("  sessions by agent: " + ", ".join(f"{k} {n}" for k, n in self.kinds.most_common()))
        out.append(f"  read {mb(self.source_bytes)} of source files; the notes "
                   f"{'that would be written' if self.dry_run else 'written'} come to {mb(self.note_bytes)}"
                   + (f", the biggest {self.biggest[0] // 1000} KB ({self.biggest[1]})" if self.biggest[0] else ""))
        if c_side := self.counts.get("sidechain markers"):
            out.append(f"  subagent (sidechain) runs cut to markers: {c_side}")
        out.append(f"  key-shaped strings left after scrubbing and redaction: {self.suspects}"
                   + (" (a real run must wait until the scrubber catches them)" if self.suspects else ""))
        if self.malformed:
            out.append(f"  {self.malformed} unreadable records skipped")
        out += [f"  {line}" for line in self.notes_]
        return out
