"""What a trusted client may read from the vault, and the two tools that read it directly.

The rules are the index's own (index/chunking.py), never a copy, so what an agent can read can't
drift from what's indexed:
- the path rules, `chunking.off_limits()`: everything on EXCLUDED_PATHS (the inbox `0 Inbox/`, all
  of `4 Archive/` with `chats-excluded/` in it, the profile diary, the long view and
  `rejected.md`), plus any dot folder or dot file (`.obsidian-win/`, `.trash/`);
- the frontmatter rules, `chunking.skip_reason()`: an `excluded` key, a `type` containing `diary`
  (`profile-diary`, `[[diary]]`, any key case), frontmatter that won't parse (also with odd fences),
  or a file that isn't UTF-8. Nobody can tell what an unreadable note says, so it's refused.

A path is checked twice: as asked (with trailing dots and spaces taken off each part, as Windows
does), and as it really is once every link, junction, short name and case difference is resolved.
Only `.md` files inside the vault with a single hard link are served. Every refusal gives the
client the same message, so a refusal never says whether a note exists; the reason goes to the
call log. The vault is only read.
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from common import notes
from index import chunking
from index.models import trim_tokens

# The most text one read_note call returns, in the chunker's token estimate (a CJK character is
# about one token, other text about 3.5 characters a token). Claude Code warns on a tool result over
# 10,000 tokens and cuts one over 25,000 (MAX_MCP_OUTPUT_TOKENS), so a page stays under both even
# for Chinese or Korean. `start` pages through the rest.
PAGE_TOKENS = 8_000

REFUSED = ("no note an agent can read at that path (it's missing, outside the vault, not a .md "
           "note, or off limits)")


class Refused(Exception):
    """A path or note the gateway won't serve. str() is the one message the client sees, the same
    for every refusal; `reason` says why, for the call log only."""

    def __init__(self, reason: str):
        super().__init__(REFUSED)
        self.reason = reason


def _clean(path: str) -> str:
    raw = str(path if isinstance(path, str) else "").strip().replace("\\", "/")
    if not raw:
        raise Refused("empty path")
    if "\x00" in raw or ":" in raw:          # drive letters, URLs, NTFS streams (`x.md::$DATA`)
        raise Refused("a `:` or NUL in the path")
    if raw.startswith("/") or PureWindowsPath(raw).drive or PureWindowsPath(raw).is_absolute() \
            or PurePosixPath(raw).is_absolute():
        raise Refused("an absolute path")
    parts = []
    for part in raw.split("/"):
        if part == "..":
            raise Refused("`..` in the path")
        part = part.rstrip(" .")             # Windows ignores these, so the rules must too
        if part:
            parts.append(part)
    if not parts:
        raise Refused("empty path")
    return "/".join(parts)


def resolve(vault: Path, path: str) -> tuple[Path, str]:
    """(the real file, its vault-relative path with forward slashes), or Refused."""
    raw = _clean(path)
    if chunking.off_limits(raw):
        raise Refused("off limits as asked")
    try:
        root = Path(vault).resolve(strict=True)
        real = (root / raw).resolve(strict=True)
    except (OSError, RuntimeError):
        raise Refused("no such file") from None
    try:
        rel = real.relative_to(root).as_posix()
    except ValueError:
        raise Refused("leads outside the vault") from None
    if chunking.off_limits(rel):
        raise Refused("off limits once resolved")
    if real.suffix.lower() != ".md" or not real.is_file():
        raise Refused("not a .md file")
    try:
        if real.stat().st_nlink > 1:         # another name for it could sit in an excluded folder
            raise Refused("more than one hard link")
    except OSError:
        raise Refused("can't stat it") from None
    return real, rel


def load(vault: Path, path: str) -> tuple[str, str, dict]:
    """(vault-relative path, text, frontmatter) of a note an agent may read, or Refused. The
    frontmatter rules run on the same bytes that are returned. Anything that goes wrong on the way
    is a refusal too, with the same message, never an error of another kind."""
    try:
        real, rel = resolve(vault, path)
        data = real.read_bytes()
        text = chunking.decode(data)
        why = chunking.skip_reason(text)
        if why:
            raise Refused(f"frontmatter rule: {why}")
        meta, _ = notes.parse_lenient(text)
        return rel, text, meta
    except Refused:
        raise
    except Exception as e:       # noqa: BLE001  RecursionError, OSError, anything: refused
        raise Refused(f"failed: {type(e).__name__}") from None


def readable(vault: Path, rel: str) -> bool:
    """True when read_note would serve this vault-relative path."""
    try:
        load(vault, rel)
        return True
    except Refused:
        return False


def read_note(vault: Path, path: str, start: int = 0, page_tokens: int = PAGE_TOKENS) -> dict:
    rel, text, meta = load(vault, path)
    start = max(0, int(start))
    page_tokens = max(1, min(int(page_tokens), PAGE_TOKENS))
    piece = trim_tokens(text[start:start + page_tokens * 4], page_tokens)
    if not piece and start < len(text):
        piece = text[start]
    end = start + len(piece)
    return {"path": rel, "title": str(meta.get("title") or meta.get("name") or PurePosixPath(rel).stem),
            "chars": len(text), "start": start, "end": end, "more": end < len(text), "text": piece}


@dataclass
class Recent:
    path: str
    title: str
    kind: str          # chat | memory
    source: str
    date: str
    modified: str      # local time, to the second


def recent(vault: Path, days: float = 2, limit: int = 20) -> list[Recent]:
    """Notes the index covers (chats and memory notes) changed in the last `days`, newest first.
    The candidates come from chunking.note_files() (the path rules already applied); each goes
    through load() before it counts."""
    vault = Path(vault)
    cutoff = time.time() - float(days) * 86400
    found = []
    for path, kind in chunking.note_files(vault):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            found.append((mtime, path, kind))
    found.sort(key=lambda t: (-t[0], t[1].as_posix()))
    out: list[Recent] = []
    for mtime, path, kind in found:
        if len(out) >= limit:
            break
        try:
            rel, _, meta = load(vault, path.relative_to(vault).as_posix())
        except Refused:
            continue
        if kind == "chat":
            title = str(meta.get("title") or path.stem)
            source = str(meta.get("source") or "chat")
            date = str(meta.get("updated") or meta.get("created") or "")
        else:
            title = str(meta.get("name") or meta.get("title") or path.stem)
            source = chunking.memory_source(rel)
            date = str(meta.get("date") or meta.get("updated") or "")
        out.append(Recent(rel, title, kind, source, date,
                          dt.datetime.fromtimestamp(mtime).isoformat(timespec="seconds")))
    return out
