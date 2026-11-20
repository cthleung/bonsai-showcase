"""Read and write vault notes: YAML frontmatter + markdown body, written safely for a synced folder.

    from common import notes
    meta, body = notes.read_note(path)
    notes.write_note(path, meta, body)           # scrubs keys, writes a temp file, then renames it
    name = notes.chat_note_name("2026-11-10", "best ramen in sapporo", "AbCdEfGh1234567")
    existing = notes.find_chat_note(vault, "google-ai-mode", "AbCdEfGh1234567")

Dates: frontmatter dates are read and written as plain "YYYY-MM-DD" strings, never datetime.date,
so `meta["updated"] > cursor` compares strings and a date round-trips unchanged. Pass dates in as
strings too (date.isoformat()); a date object is converted on write.

Quoting is left to YAML: `project: "[[Food]]"` comes back quoted (bare [[Food]] would be a list),
`tags: [chat]` stays a flow list, plain titles come out unquoted. Key order is kept as given.

write_note skips the write when the file already holds the same text, so a sync client doesn't
upload unchanged notes. It returns True when it wrote.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import time
from pathlib import Path

import yaml

from common.scrub import scrub_obj, scrub_text

CHATS_DIR = Path("3 Resources") / "chats"
EXCLUDED_DIR = Path("4 Archive") / "chats-excluded"   # chats moved out as a person's or junk


# --- YAML that keeps dates as strings -------------------------------------------------------

def _without_timestamps(cls):
    cls.yaml_implicit_resolvers = {
        ch: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:timestamp"]
        for ch, resolvers in cls.yaml_implicit_resolvers.copy().items()
    }
    return cls


@_without_timestamps
class _Loader(yaml.SafeLoader):
    pass


@_without_timestamps
class _Dumper(yaml.SafeDumper):
    pass


def _date_repr(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value.isoformat())


def _list_repr(dumper, value):
    # Lists of plain values stay inline (`tags: [chat]`); anything nested goes block style.
    flat = all(not isinstance(v, (list, dict)) for v in value)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=flat)


_Dumper.add_representer(_dt.date, _date_repr)
_Dumper.add_representer(_dt.datetime, _date_repr)
_Dumper.add_representer(list, _list_repr)


def dump_frontmatter(meta: dict) -> str:
    """Block-style mapping, one `key: value` line per field, so `grep "^visibility: private"` works."""
    if not meta:
        return ""
    return yaml.dump(meta, Dumper=_Dumper, sort_keys=False, allow_unicode=True,
                     default_flow_style=False, width=10_000)


def parse(text: str) -> tuple[dict, str]:
    """(frontmatter dict, body). A note without frontmatter gives ({}, text)."""
    text = text.lstrip("﻿").replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = re.search(r"^---[ \t]*$", text[4:], re.M)
    if not end:
        return {}, text
    meta = yaml.load(text[4:4 + end.start()], Loader=_Loader) or {}
    if not isinstance(meta, dict):
        raise ValueError("frontmatter is not a mapping")
    body = text[4 + end.end():]
    return meta, body[1:] if body.startswith("\n") else body


@_without_timestamps
class _RulesLoader(yaml.SafeLoader):
    """For the off-limits rules: a key that appears twice in one mapping is an error, since which
    one counts would be a guess (`type: diary` then `type: note`)."""

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
            seen = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=True)
                try:
                    marker = ("k", key) if not isinstance(key, (list, dict)) else ("r", repr(key))
                except TypeError:
                    marker = ("r", repr(key))
                if marker in seen:
                    raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}",
                                                            key_node.start_mark)
                seen.add(marker)
        return super().construct_mapping(node, deep=deep)


def parse_lenient(text: str) -> tuple[dict, str]:
    """Like parse(), for deciding what's off limits rather than for writing. It also sees
    frontmatter whose fences carry trailing spaces or that sits after blank lines, and it raises
    (so the caller refuses the note as unreadable) where the reading would be a guess:
    - an opening fence with no closing one;
    - a `...` closing line with a `---` line anywhere after it (which of the two ends the
      frontmatter is ambiguous; `...` alone is the YAML end marker and is accepted);
    - a key that appears twice in one mapping;
    - frontmatter that isn't a mapping."""
    lines = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or lines[i].rstrip() != "---":
        return {}, "\n".join(lines)
    for j in range(i + 1, len(lines)):
        closer = lines[j].rstrip()
        if closer in ("---", "..."):
            if closer == "..." and any(line.rstrip() == "---" for line in lines[j + 1:]):
                raise ValueError("frontmatter ends with `...` and a `---` line follows: ambiguous")
            meta = yaml.load("\n".join(lines[i + 1:j]), Loader=_RulesLoader) or {}
            if not isinstance(meta, dict):
                raise ValueError("frontmatter is not a mapping")
            return meta, "\n".join(lines[j + 1:])
    raise ValueError("frontmatter has no closing fence")


def render(meta: dict, body: str) -> str:
    body = body if body.endswith("\n") or not body else body + "\n"
    if not meta:
        return body
    return f"---\n{dump_frontmatter(meta)}---\n{body}"


def read_note(path: Path | str) -> tuple[dict, str]:
    return parse(Path(path).read_text(encoding="utf-8"))


# --- safe writes ----------------------------------------------------------------------------

def write_text_atomic(path: Path | str, text: str, retries: int = 5, delay: float = 0.2) -> None:
    """Write to a temp file beside the target, then os.replace it in, so a sync client never
    uploads a partly written file. Retries the rename a few times: a sync client can briefly lock a
    file it's uploading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(delay)


def write_note(path: Path | str, meta: dict, body: str, scrub: bool = True) -> bool:
    """Write a note. Keys are scrubbed from the frontmatter and body unless scrub=False.
    Returns False (and leaves the file alone) when the file already holds this exact text."""
    if scrub:
        meta, body = scrub_obj(meta), scrub_text(body)
    text = render(meta, body)
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8").replace("\r\n", "\n") == text:
        return False
    write_text_atomic(path, text)
    return True


# --- file names -----------------------------------------------------------------------------

# Windows-illegal, what breaks Obsidian links, and every character str.splitlines() breaks a line on.
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f\x85  #^\[\]]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_SOURCE_RX = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_reserved(name: str) -> bool:
    """True for Windows device names (con, nul, aux, com1 ...), with or without an extension."""
    return name.split(".")[0].strip().upper() in _RESERVED


def is_safe_part(part: str) -> bool:
    """True when `part` is fine as one file or folder name on Windows, macOS and in an Obsidian
    link: not empty, not `.` or `..`, no control characters, none of `<>:"/\\|?*#^[]`, no leading
    or trailing space, no trailing dot, and not a Windows reserved name."""
    return (bool(part) and part not in (".", "..") and not _BAD_CHARS.search(part)
            and part == part.strip(" ") and not part.endswith(".") and not is_reserved(part))


def safe_filename(name: str, max_len: int = 100) -> str:
    """A title made safe to use in a file name on Windows, macOS and in Obsidian links.
    No extension is added. Never returns an empty string."""
    name = _BAD_CHARS.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rsplit(" ", 1)[0] if " " in name[:max_len] else name[:max_len]
    name = name.rstrip(" .")
    if not name:
        name = "untitled"
    if is_reserved(name):
        name = f"_{name}"
    return name


def chat_id8(chat_id: str) -> str:
    """The id part of a chat file name: the first 8 characters of chat_id once anything but
    letters, digits, `_` and `-` is dropped. Never empty."""
    return _ID_CHARS.sub("", str(chat_id))[:8] or "noid"


def chat_note_name(created: str, title: str, chat_id: str) -> str:
    """`<YYYY-MM-DD> <title> (<id8>).md` (docs/note-format.md). Round brackets, because Obsidian
    ends a `[[...]]` link at the first `]]`, so a name ending in `]` can't be linked.
    The title is scrubbed first: write_note scrubs the body and frontmatter, but nothing scrubs a
    path, so a key pasted into a chat title would otherwise end up in the vault as a file name."""
    if not _DATE_RX.fullmatch(str(created)):
        raise ValueError(f"created must be YYYY-MM-DD, got {created!r}")
    return f"{created} {safe_filename(scrub_text(str(title)), 80)} ({chat_id8(chat_id)}).md"


def _check_source(source: str) -> None:
    if not _SOURCE_RX.fullmatch(str(source)):
        raise ValueError(f"bad chat source {source!r}")


def chat_note_path(vault: Path, source: str, created: str, title: str, chat_id: str) -> Path:
    """Where a chat note lives. Use find_chat_note first: once written, a chat's file name stays put."""
    _check_source(source)
    return Path(vault) / CHATS_DIR / source / created[:4] / chat_note_name(created, title, chat_id)


def find_chat_note(vault: Path, source: str, chat_id: str) -> Path | None:
    """The existing note for this chat, found by the ` (<id8>).md` suffix and confirmed by chat_id.
    It also finds a note still named the old way (` [<id first 8>].md`), so a chat captured before
    the rename to round brackets never gets a second note. A chat moved out to
    `4 Archive/chats-excluded/<label>/` is found there too (confirmed by source as well, since that
    folder has no source level), so capture never writes a second copy of it."""
    _check_source(source)
    suffixes = (f"({chat_id8(chat_id)}).md", f"[{str(chat_id)[:8]}].md")
    roots = [(Path(vault) / CHATS_DIR / source, False), (Path(vault) / EXCLUDED_DIR, True)]
    for root, archived in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/*.md")):
            if path.name.endswith(suffixes):
                try:
                    meta, _ = read_note(path)
                except (OSError, ValueError, UnicodeDecodeError, yaml.YAMLError):
                    continue
                if str(meta.get("chat_id")) != str(chat_id):
                    continue
                if archived and str(meta.get("source")) != source:
                    continue
                return path
    return None


def is_excluded(vault: Path, path: Path, meta: dict | None = None) -> bool:
    """True for a chat that was sorted out: it sits under `4 Archive/chats-excluded/`, or its
    frontmatter has `excluded:` set."""
    try:
        Path(path).resolve().relative_to((Path(vault) / EXCLUDED_DIR).resolve())
        return True
    except ValueError:
        pass
    return bool((meta or {}).get("excluded"))
