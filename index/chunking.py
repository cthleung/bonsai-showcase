"""Vault notes -> chunks for the index.

- **Chat notes** (`3 Resources/chats/`): one chunk per turn, the user's prompt plus the answer, with
  the answer cleaned (source lists and link URLs dropped). A turn longer than MAX_TOKENS is cut into
  windows of WINDOW_TOKENS that overlap by OVERLAP_TOKENS, and every window after the first repeats
  the start of the user's prompt so it still says what was asked.
- **Memory notes** (`1 Projects/<P>/memory/`, everything in `2 Areas/`, so the profile and people
  notes too): one chunk per note, windowed the same way if one ever runs long.
- Every chunk starts with `[<source> | <date> | <title>]`, the free version of contextual retrieval.

Skipped: everything on EXCLUDED_PATHS (the inbox, the archive, the profile diary, the long view
and rejected claims), notes with an `excluded` key or a `type` containing a word on EXCLUDED_TYPES
(excluded_meta), notes whose frontmatter won't parse or that aren't UTF-8, and files whose names
start with a dot. Token counts are estimates (a CJK or Hangul character is about one token, other text about
3.5 characters a token), good enough to keep each chunk well inside the server's context.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from capture.chatnote import SOURCE_NAMES
from common.turns import parse_turns, split_turn as turns_split
from common import notes

MAX_TOKENS = 1000          # a turn up to this size stays one chunk
WINDOW_TOKENS = 800        # longer turns are cut into windows this big
OVERLAP_TOKENS = 100       # that overlap by this much
PROMPT_HEAD_TOKENS = 150   # how much of the prompt each later window repeats

CHATS_DIR = notes.CHATS_DIR
MEMORY_GLOBS = ("1 Projects/*/memory/**/*.md", "2 Areas/**/*.md")

# Never indexed: agents never load the diary
# or the long view, rejected claims must never come back, and the inbox holds unreviewed agent
# output and digests. A path ending in "/" is a folder, anything else is one file. Matching
# ignores case. A note already in the index that matches is removed on the next build.
EXCLUDED_PATHS = (
    "0 Inbox/",
    "4 Archive/",
    "2 Areas/Profile/diary/",
    "2 Areas/Profile/Long view.md",
    "2 Areas/Profile/rejected.md",
)
EXCLUDED_TYPES = ("diary",)        # a `type:` containing one of these is never indexed (excluded_meta)


def excluded_path(rel: str) -> bool:
    rel = rel.lower()
    for p in EXCLUDED_PATHS:
        p = p.lower()
        if rel == p or (p.endswith("/") and rel.startswith(p)):
            return True
    return False


def hidden_path(rel: str) -> bool:
    """True when any part of the path starts with a dot (`.obsidian-win/`, `.trash/`, `.draft.md`)."""
    return any(part.startswith(".") for part in Path(rel).parts)


def off_limits(rel: str) -> bool:
    """The path rules in one place: on EXCLUDED_PATHS, or hidden. note_files() and the gateway
    (gateway/access.py) both use this, so what agents can read never drifts from what's indexed."""
    return excluded_path(rel) or hidden_path(rel)


FLAT_MAX_DEPTH = 20     # nesting deeper than this in one value is refused as unreadable
FLAT_MAX_ITEMS = 1000   # and so is a value that flattens to more scalars than this (YAML aliases)


def _flat(value, depth: int = 0, out: list | None = None) -> list:
    """Every scalar in a frontmatter value, lists and mappings flattened (`type: [[diary]]` is a
    list inside a list in YAML). Raises ValueError past FLAT_MAX_DEPTH or FLAT_MAX_ITEMS, so a
    self-referencing alias (`type: &a [*a]`) or an alias bomb can't hang or crash the rules."""
    out = [] if out is None else out
    if depth > FLAT_MAX_DEPTH:
        raise ValueError("frontmatter value nested too deep")
    if isinstance(value, dict):
        for k, v in value.items():
            _flat(k, depth + 1, out)
            _flat(v, depth + 1, out)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            _flat(v, depth + 1, out)
    else:
        out.append(value)
        if len(out) > FLAT_MAX_ITEMS:
            raise ValueError("frontmatter value too big")
    return out


def excluded_meta(meta: dict) -> bool:
    """Keys are compared without case, and every key that folds to the same name counts (`Type`
    and `type` both). Excluded when:
    - an `excluded` key is there at all, whatever its value (a chat moved out);
    - any `type` (or `types`) value contains a word on EXCLUDED_TYPES once `[[ ]]` link brackets
      are stripped and lists flattened: `diary`, `profile-diary`, `"[[diary]]"`, `[Diary]` all count.
    Raises ValueError for a value _flat() won't flatten; skip_reason() treats that as unreadable."""
    if not isinstance(meta, dict):
        return False
    kinds: list = []
    for key, value in meta.items():
        name = str(key).strip().lower()
        if name == "excluded":
            return True
        if name in ("type", "types"):
            kinds += _flat(value)
    words = [str(k).replace("[", "").replace("]", "").strip().lower() for k in kinds if k is not None]
    return any(t in w for w in words for t in EXCLUDED_TYPES)


# Bump CHUNKER_VERSION whenever a change here alters chunk text: the next build then re-chunks
# every note (vectors of chunks whose text came out the same are reused). rules_id() covers the
# exclusion lists and the folders scanned: when it changes, the next build re-reads every note's
# frontmatter, so a note indexed before a rule existed is removed even if the file never changed.
# Bump RULES_VERSION when how the lists are applied changes (2: type matched as a substring,
# keys without case, any `excluded` key, lenient frontmatter, files that aren't UTF-8 refused;
# 3: duplicate keys, an ambiguous `...` closer and frontmatter the strict parser refuses are all
# unreadable, every key that folds to `type` counts, and runaway values are refused).
CHUNKER_VERSION = "1"
RULES_VERSION = "3"


def rules_id() -> str:
    blob = repr((sorted(EXCLUDED_PATHS), sorted(EXCLUDED_TYPES), sorted(MEMORY_GLOBS), str(CHATS_DIR),
                 RULES_VERSION))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def decode(data: bytes) -> str | None:
    """A note's bytes as text, or None when they aren't valid UTF-8 (a UTF-16 file, say), which
    the rules can't read and so never pass."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def skip_reason(text: str | None) -> str | None:
    """Why a note's text keeps it out of the index and away from agents ("excluded",
    "unreadable"), or None. None for text means the file wasn't valid UTF-8 (decode()).
    Never raises. Both readings must work: what the strict parser (the one notes are written
    with) refuses stays refused, and the lenient one sees frontmatter the strict one would miss.
    Either reading being excluded excludes."""
    if text is None:
        return "unreadable"
    try:
        strict, _ = notes.parse(text)
        lenient, _ = notes.parse_lenient(text)
        return "excluded" if excluded_meta(strict) or excluded_meta(lenient) else None
    except Exception:            # RecursionError and MemoryError included: refused, never a crash
        return "unreadable"

CJK_RX = re.compile(r"[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-䶿一-鿿가-힯豈-﫿]")
LINK_RX = re.compile(r"!?\[([^\]]*)\]\((?:[^)\s]+)\)")
BARE_URL_RX = re.compile(r"https?://\S{60,}")


@dataclass
class Chunk:
    path: str                  # vault-relative, forward slashes
    kind: str                  # chat | memory
    source: str                # chatgpt, google-ai-mode, claude, memory, profile, people...
    title: str
    date: str                  # YYYY-MM-DD
    turn: int | None           # chat turn number, None for a memory note
    part: int                  # window number within the turn, from 0
    text: str                  # what's embedded and searched: prefix plus content
    visibility: str = "private"

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:24]


@dataclass
class NoteChunks:
    path: str
    kind: str
    title: str
    date: str
    source: str
    visibility: str
    chunks: list[Chunk] = field(default_factory=list)


def est_tokens(text: str) -> int:
    cjk = len(CJK_RX.findall(text))
    return cjk + int((len(text) - cjk) / 3.5) + 1


def has_cjk(text: str) -> bool:
    return bool(CJK_RX.search(text))


# --- finding the notes ----------------------------------------------------------------------

def note_files(vault: Path) -> list[tuple[Path, str]]:
    """[(file, kind)] for every note the index covers, sorted, with `4 Archive/` and dot files left out."""
    vault = Path(vault)
    found: dict[Path, str] = {}
    chats = vault / CHATS_DIR
    if chats.is_dir():
        for p in chats.rglob("*.md"):
            found[p] = "chat"
    for pattern in MEMORY_GLOBS:
        for p in vault.glob(pattern):
            found.setdefault(p, "memory")
    out = []
    for p, kind in found.items():
        rel = p.relative_to(vault).as_posix()
        if not p.is_file() or off_limits(rel):
            continue
        out.append((p, kind))
    return sorted(out, key=lambda t: t[0].as_posix())


# --- cleaning and windowing -----------------------------------------------------------------

def clean_answer(text: str) -> str:
    """Drop the `#### Sources` lists, keep link labels without their URLs, squeeze blank lines."""
    text = re.split(r"^#{2,6} Sources\s*$", text, maxsplit=1, flags=re.M)[0]
    text = LINK_RX.sub(lambda m: m.group(1), text)
    text = BARE_URL_RX.sub("[link]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pieces(text: str, budget: int) -> list[str]:
    """Split text into lines, and lines into words or character runs, each under `budget` tokens."""
    out: list[str] = []
    for line in text.split("\n"):
        line = line + "\n"
        if est_tokens(line) <= budget:
            out.append(line)
            continue
        for word in re.split(r"(?<=\s)", line):
            if est_tokens(word) <= budget:
                out.append(word)
                continue
            # A run with no spaces (Chinese, or a long URL or blob): small pieces, so windows can
            # still overlap inside it.
            small = max(1, min(budget, 50))
            step = small if has_cjk(word) else int(small * 3.5)
            out += [word[i:i + step] for i in range(0, len(word), step)]
    return out


def windows(text: str, size: int = WINDOW_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Greedy windows of whole lines (or words) up to `size` tokens, each repeating the last
    `overlap` tokens of the one before."""
    pieces = _pieces(text, size)
    cost = [est_tokens(p) for p in pieces]
    out, start = [], 0
    while start < len(pieces):
        end, total = start, 0
        while end < len(pieces) and (end == start or total + cost[end] <= size):
            total += cost[end]
            end += 1
        out.append("".join(pieces[start:end]).strip())
        if end >= len(pieces):
            break
        back, kept = end, 0
        while back - 1 > start and kept + cost[back - 1] <= overlap:
            back -= 1
            kept += cost[back]
        start = back
    return [w for w in out if w]


def _head(text: str, tokens: int) -> str:
    if est_tokens(text) <= tokens:
        return text.strip()
    return windows(text, tokens, 0)[0].strip() + " ..."


# --- chat notes -----------------------------------------------------------------------------

def split_turns(body: str, message_count=None) -> list[tuple[int, str | None, str]]:
    """[(turn number, date or None, the turn's text without its heading)], from
    common.turns.parse_turns, so a turn heading inside a code block doesn't start a turn."""
    return [(t.n, t.date, t.text[len(t.heading):]) for t in parse_turns(body, message_count).turns]


def split_turn(text: str, source: str) -> tuple[str, str]:
    """(the user's prompt, the answer) from one turn's text. Every `>` line of the prompt (the quoted
    bit of an answer) is left out: it's the model's words. The answer heading is found by
    common.turns; `source` stays for callers."""
    you, answer, _ = turns_split(text)
    return you, answer


def _turn_chunks(rel: str, meta: dict, title: str, source: str, visibility: str,
                 n: int | None, date: str, prompt: str, answer: str) -> list[Chunk]:
    who = SOURCE_NAMES.get(source, source)
    prefix = f"[{who} | {date} | {title}]"
    body = f"You: {prompt}\n\n{who}: {answer}" if prompt or answer else ""
    if not body:
        return []
    if est_tokens(body) <= MAX_TOKENS:
        parts = [f"{prefix}\n{body}"]
    else:
        wins = windows(body)
        head = f"You: {_head(prompt, PROMPT_HEAD_TOKENS)}" if prompt else ""
        parts = [f"{prefix}\n{wins[0]}"] + [
            f"{prefix}\n{head}\n[...]\n{w}" if head else f"{prefix}\n{w}" for w in wins[1:]]
    return [Chunk(rel, "chat", source, title, date, n, i, t, visibility) for i, t in enumerate(parts)]


def chat_chunks(rel: str, meta: dict, body: str) -> list[Chunk]:
    source = str(meta.get("source") or "chat")
    title = str(meta.get("title") or Path(rel).stem)
    visibility = str(meta.get("visibility") or "private")
    fallback = str(meta.get("created") or meta.get("updated") or "")
    turns = parse_turns(body, meta.get("message_count")).turns
    if not turns:   # no turn headings: index the whole body in windows
        return _turn_chunks(rel, meta, title, source, visibility, None, fallback, "", clean_answer(body))
    out: list[Chunk] = []
    for t in turns:
        out += _turn_chunks(rel, meta, title, source, visibility, t.n, t.date or fallback,
                            t.you, clean_answer(t.answer))
    return out


# --- memory notes ---------------------------------------------------------------------------

def memory_source(rel: str) -> str:
    parts = rel.split("/")
    if rel.startswith("2 Areas/Profile/"):
        return "profile"
    if rel.startswith("2 Areas/People/"):
        return "people"
    if len(parts) > 2 and parts[0] == "1 Projects":
        return f"memory: {parts[1]}"
    if len(parts) > 2 and parts[0] == "2 Areas":
        return f"area: {parts[1]}"
    return "memory"


def memory_chunks(rel: str, meta: dict, body: str, file_date: str) -> list[Chunk]:
    title = str(meta.get("name") or meta.get("title") or Path(rel).stem)
    date = str(meta.get("date") or meta.get("updated") or meta.get("created") or file_date)
    source = memory_source(rel)
    visibility = str(meta.get("visibility") or "private")
    desc = str(meta.get("description") or "").strip()
    content = (desc + "\n\n" if desc else "") + clean_answer(body)
    prefix = f"[{source} | {date} | {title}]"
    parts = [content] if est_tokens(content) <= MAX_TOKENS else windows(content)
    return [Chunk(rel, "memory", source, title, date, None, i, f"{prefix}\n{p}", visibility)
            for i, p in enumerate(parts) if p.strip()]


# --- one note -------------------------------------------------------------------------------

def note_chunks(vault: Path, path: Path, kind: str, text: str | None = None) -> NoteChunks | None:
    """The chunks for one note, or None when the note is skipped (`excluded:` set, or frontmatter
    that won't parse)."""
    rel = path.relative_to(vault).as_posix()
    if text is None:
        text = path.read_text(encoding="utf-8", errors="replace")
    if skip_reason(text):
        return None
    try:
        meta, body = notes.parse(text)
    except Exception:
        return None
    file_date = dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
    if kind == "chat":
        chunks = chat_chunks(rel, meta, body)
        title = str(meta.get("title") or path.stem)
        date = str(meta.get("created") or meta.get("updated") or file_date)
        source = str(meta.get("source") or "chat")
    else:
        chunks = memory_chunks(rel, meta, body, file_date)
        title = str(meta.get("name") or meta.get("title") or path.stem)
        date = str(meta.get("date") or file_date)
        source = memory_source(rel)
    return NoteChunks(rel, kind, title, date, source, str(meta.get("visibility") or "private"), chunks)
