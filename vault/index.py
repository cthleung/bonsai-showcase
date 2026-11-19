"""Rebuild `Index.md` at the vault root: one line per note, short enough to import into every session.

    python -m vault.index            # write Index.md
    python -m vault.index --print    # print it instead

Left out, so the file stays short: chat notes (one count line per source instead), templates,
`4 Archive/` (one count line), dot folders, and Index.md itself. Chats sorted out of the vault
(`excluded:` set, or anywhere under `4 Archive/chats-excluded/`) aren't counted at all. Each line is
`- [[<path>|<title>]] - <description>`, where the description is the note's `description`
field, else its project status, else nothing. Anything with an `@` is put in backticks, because
Claude Code reads a bare `@path` in an imported file as another import.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

from common import config, notes
from vault.templates import TEMPLATES_DIR

INDEX = "Index.md"
CHATS = ("3 Resources", "chats")
ARCHIVE = "4 Archive"
EXCLUDED = tuple(notes.EXCLUDED_DIR.parts)
MAX_LINES = 150          # Claude Code warns when a memory file passes 200 lines
DESC_LEN = 110
ORDER = ["North Star.md", "0 Inbox", "1 Projects", "2 Areas", "3 Resources", "Tasks"]


def _top_rank(rel: Path) -> tuple[int, str]:
    top = rel.parts[0]
    return (ORDER.index(top) if top in ORDER else len(ORDER), top)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip().replace("`", "'")
    if len(text) > DESC_LEN:
        text = text[:DESC_LEN].rsplit(" ", 1)[0] + " ..."
    return re.sub(r"(\S*@\S*)", r"`\1`", text)


def _describe(meta: dict) -> str:
    if meta.get("description"):
        return _clean(meta["description"])
    if "project" in (meta.get("tags") or []) and meta.get("status"):
        return _clean(f"project, {meta['status']}")
    if "task" in (meta.get("tags") or []) and meta.get("status"):
        return _clean(f"task, {meta['status']}")
    return ""


def _skip(rel: Path) -> bool:
    return (any(p.startswith(".") for p in rel.parts)
            or rel.parts[0] == TEMPLATES_DIR
            or rel.parts[:2] == CHATS
            or rel.parts[0] == ARCHIVE
            or rel.as_posix() == INDEX)


def _excluded(path: Path) -> bool:
    """True when the chat note's frontmatter has `excluded:` set."""
    try:
        meta, _ = notes.read_note(path)
    except (OSError, ValueError, UnicodeDecodeError, yaml.YAMLError):
        return False
    return bool(meta.get("excluded"))


def build(vault: Path) -> str:
    vault = Path(vault)
    rows: list[tuple[Path, str]] = []
    chats: Counter = Counter()
    archived = 0
    for path in sorted(vault.rglob("*.md")):
        rel = path.relative_to(vault)
        if rel.parts[:2] == EXCLUDED:
            continue
        if rel.parts[:2] == CHATS and len(rel.parts) > 2:
            if not _excluded(path):
                chats[rel.parts[2]] += 1
            continue
        if rel.parts[0] == ARCHIVE:
            archived += 1
            continue
        if _skip(rel):
            continue
        try:
            meta, _ = notes.read_note(path)
        except (OSError, ValueError, yaml.YAMLError):
            meta = {}
        title = _clean(meta.get("title") or meta.get("name") or path.stem).replace("|", "-")
        link = rel.with_suffix("").as_posix()
        line = f"- [[{link}|{title}]]" if title != path.stem else f"- [[{link}]]"
        desc = _describe(meta)
        rows.append((rel, f"{line} - {desc}" if desc else line))

    rows.sort(key=lambda r: (_top_rank(r[0]), r[0].as_posix().lower()))
    out = ["# Index", "",
           "One line per note in this vault, rebuilt by `python -m vault.index`. "
           "Chats, templates and the archive are counted, not listed.", ""]
    section = None
    for rel, line in rows:
        top = rel.parts[0] if len(rel.parts) > 1 else "Root"
        if top != section:
            if section is not None:
                out.append("")
            out += [f"## {top}", ""]
            section = top
        out.append(line)
    counts = []
    for source, n in sorted(chats.items()):
        counts.append(f"- Chats from {source}: {n} in `3 Resources/chats/{source}/`")
    if archived:
        counts.append(f"- Archived notes: {archived} in `4 Archive/`")
    if counts:
        out += ["", "## Counts", "", *counts]
    return "\n".join(out) + "\n"


def write(vault: Path) -> tuple[bool, int]:
    text = build(vault)
    lines = text.count("\n")
    changed = notes.write_note(Path(vault) / INDEX, {}, text)
    return changed, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--print", action="store_true", dest="print_only")
    args = ap.parse_args(argv)
    vault = config.load().require().vault
    if args.print_only:
        sys.stdout.write(build(vault))
        return 0
    changed, lines = write(vault)
    print(f"{'wrote' if changed else 'unchanged'}: {INDEX} ({lines} lines)")
    if lines > MAX_LINES:
        print(f"warning: {INDEX} is over {MAX_LINES} lines; every session pays for it", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
