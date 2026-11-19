"""The morning digest: `0 Inbox/digest <YYYY-MM-DD>.md`, one box per proposal (docs/note-format.md):

    - [ ] add | 1 Projects/Bonsai/memory/night-job-time.md | the user set the time | source: [[<note>]] ^p01
        > The night job runs at 02:00 on the PC.

The `^p01` at the end is an Obsidian block id: Obsidian hides it. The sidecar,
`<vault state>/proposals/<date>.json`, is what gets applied, and it stores each box line exactly as
written (`line`). apply.py counts a tick only when one line of the digest equals that stored line
except for the box character, so a line smuggled in elsewhere can't tick anything. A proposal whose
box line would hold a line break or other control character is held, never shown as a box.
The quoted text under each box shows the user what gets written.

check_text() is the plain grep for the vault's writing bans. Grep can't tell a metaphor from a
literal use, so the banned words are banned outright. The digest is checked before it's written.
"""
from __future__ import annotations

import re
from pathlib import Path

from common.notes import read_note, write_note
from common.scrub import scrub_text
from dream.quotes import strip_quoted

DIGEST_DIR = "0 Inbox"
ORDER = ("add", "profile", "merge", "move", "flag")
HEADINGS = {"add": "New notes", "profile": "Profile facts", "merge": "Additions to notes you have",
            "move": "Moves", "flag": "Contradictions to look at"}

BANS = [
    ("dash", re.compile("[" + chr(0x2014) + chr(0x2013) + "]")),   # em and en dash
    ("banned word", re.compile(r"\bhal(f|ves)\b", re.I)),
    ("banned word", re.compile(r"\bweigh", re.I)),
    ("banned word", re.compile(r"\bgravit", re.I)),
    ("banned word", re.compile(r"\bstructural", re.I)),
    ("banned word", re.compile(r"\bload[- ]?bearing", re.I)),
    ("banned word", re.compile(r"\bscaffold", re.I)),
    ("banned word", re.compile(r"\bframeworks?\b", re.I)),
    ("banned word", re.compile(r"\barchitectur", re.I)),
    ("relative date", re.compile(r"\b(yesterday|today|tonight|tomorrow|last night|(last|next|this) "
                                 r"(week|month|year))\b", re.I)),
    ("hedging", re.compile(r"\b(maybe|perhaps|probably|possibly|might|likely|seems?|seemingly|apparently|"
                           r"i think|i guess|sort of|kind of)\b", re.I)),
]
_LINK_RX = re.compile(r"\[\[[^\n]*?\]\]+")
_ITEM_RX = re.compile(r"^\s*- (\[.\] )?\w+ \| ")
_MOVES_RX = re.compile(r"(Moves to ).*$")


def wikilink(stem: str) -> str:
    """The link to a note by its file stem. Every link dream writes is built here, so a change to
    the link form is one line. Chat names end in ` (<id8>)` (docs/note-format.md)."""
    return f"[[{stem}]]"


def check_text(text: str) -> list[str]:
    """['hedging: "maybe"', ...] for every ban the text breaks. [] means clean."""
    hits = []
    for label, rx in BANS:
        for m in rx.finditer(text):
            hits.append(f'{label}: "{m.group(0)}"')
    return hits


def check_digest(text: str, quoted=()) -> list[str]:
    """check_text over the digest's own words. Note names are left out (links, the target column of
    any item line, the folder after "Moves to"), since those are names, not prose. So are `quoted`
    spans in quote marks: quotes the night job checked against the user's own words (dream/quotes.py)."""
    text = strip_quoted(text, quoted)
    lines = []
    for line in text.splitlines():
        if _ITEM_RX.match(line):
            parts = line.split(" | ")
            line = " | ".join([parts[0]] + [x for x in parts[2:] if not x.startswith("source:")])
        line = _MOVES_RX.sub(lambda m: m.group(1), line)
        lines.append(_LINK_RX.sub("", line))
    return check_text("\n".join(lines))


# --- paths and the sidecar ------------------------------------------------------------------

def digest_rel(day: str) -> str:
    return f"{DIGEST_DIR}/digest {day}.md"


def sidecar_path(vstate: Path, day: str) -> Path:
    return Path(vstate) / "proposals" / f"{day}.json"


def pending_keys(vstate: Path) -> set[tuple[str, str]]:
    """(action, target) for every proposal not yet applied in any sidecar, so a night doesn't
    propose the same thing twice while the user hasn't got to it."""
    from dream.guards import load_json
    keys = set()
    folder = Path(vstate) / "proposals"
    if folder.is_dir():
        for f in folder.glob("*.json"):
            for p in load_json(f, {}).get("proposals", []):
                if not p.get("applied") and not p.get("undone"):
                    keys.add((p["action"], p["target"]))
    return keys


# --- rendering ------------------------------------------------------------------------------

def _quote(text: str, indent: str = "    ") -> str:
    return "".join(f"{indent}> {line}".rstrip() + "\n" for line in str(text).strip().splitlines() or [""])


# every character str.splitlines() breaks a line on, plus the other control characters
CONTROL_RX = re.compile("[" + "".join(re.escape(chr(c)) for c in [*range(32), 0x7f, 0x85, 0x2028, 0x2029]) + "]")


def printable(text: str) -> str:
    """Text with control characters shown as `?`, for lines that only report (held, read)."""
    return CONTROL_RX.sub("?", str(text))


def box_line(p: dict) -> str:
    """The unticked box line for a proposal, with no line break at the end."""
    links = ", ".join(wikilink(l) for l in p["links"])
    return f"- [ ] {p['action']} | {p['target']} | {p['reason']} | source: {links} ^{p['id']}"


def render_item(p: dict) -> str:
    out = box_line(p) + "\n"
    a = p["action"]
    if a in ("add", "profile"):
        out += _quote(f"{p['description']} ({p['type']}, {p['date']}, {p['visibility']})")
        out += _quote(p["text"])
    elif a == "merge":
        out += _quote(f"Adds to the note:\n{p['text']}")
    elif a == "move":
        out += _quote(f"Moves to {p['move_to']}")
    elif a == "flag":
        out += _quote(f"{p['text']}\nTicking this changes nothing. It marks it as seen.")
    return out


MISMATCH_INTRO = ("These boxes are ticked, but their line changed after the digest wrote it, so nothing was "
                  "applied. To apply one, put its line back the way it was (no indent, the same words and "
                  "links) and leave the tick.\n\n")


def render_section(proposals: list[dict], held: list[dict], read: list[dict], stats: str,
                   mismatched: list[tuple[str, str]] = ()) -> str:
    """The body of one run: ticks that didn't count, stats, proposals by action, held items, and
    what was read."""
    out = []
    if mismatched:
        out.append("## Ticks that didn't count\n\n" + MISMATCH_INTRO)
        out += [f"- ticked but the line doesn't match: {printable(pid)} in {wikilink('digest ' + printable(day))}\n"
                for day, pid in mismatched]
        out.append("\n")
    out.append(stats.strip() + "\n")
    for action in ORDER:
        items = [p for p in proposals if p["action"] == action]
        if items:
            out.append(f"\n## {HEADINGS[action]}\n\n")
            out += [render_item(p) for p in items]
    if held:
        out.append("\n## Held back for wording\n\nThese broke the writing rules twice, so they aren't up for "
                   "a tick. Their text is kept in the sidecar under BONSAI_STATE.\n\n")
        for p in held:
            labels = sorted({h.split(":")[0] for h in p.get("held", [])})
            out.append(f"- {printable(p['action'])} | {printable(p['target'])} | broke: {', '.join(labels)} ({p['id']})\n")
    if read:
        out.append("\n## Read\n\n")
        for r in read:
            turns = f" (turns {r['turns'][0]} to {r['turns'][1]})" if r.get("turns") else ""
            out.append(f"- {wikilink(printable(r['link']))}{turns}\n")
    return "".join(out)


INTRO = ("Tick a box to approve it. The next night run applies ticked boxes, or run "
         "`python -m dream.apply` on the PC. It snapshots the vault first, so `python -m dream.apply "
         "--undo` puts things back. The quoted text under a box is what gets written; to change the "
         "words, tick it and edit the note after.\n")


def write(vault: Path, vstate: Path, day: str, run_time: str, proposals: list[dict], held: list[dict],
          read: list[dict], stats: str, dry_run: bool = False,
          mismatched: list[tuple[str, str]] = ()) -> tuple[Path | None, str]:
    """Number the proposals, save the sidecar, and write (or add to) the digest note. `mismatched`
    is (day, id) for ticked boxes whose line doesn't match, listed at the top of this run.
    Returns (digest path or None for a dry run, the text written). Raises ValueError if the digest's
    words break the writing rules."""
    from dream.guards import load_json, save_json
    side = sidecar_path(vstate, day)
    data = load_json(side, {"day": day, "proposals": [], "held": []})
    n = len(data["proposals"]) + len(data["held"])
    for p in proposals + held:
        n += 1
        p["id"] = f"p{n:02d}"
    for p in list(proposals):   # a banned word in a model-chosen name costs that proposal, not the night
        line = box_line(p)
        hits = ["control character"] if CONTROL_RX.search(line) else check_digest(render_item(p), p.get("quoted"))
        p["line"] = scrub_text(line)   # what write_note will leave in the file
        if hits:
            p.pop("line")
            p["held"] = hits
            proposals.remove(p)
            held.append(p)
    section = render_section(proposals, held, read, stats, mismatched)
    path = Path(vault) / digest_rel(day)
    if path.exists():
        meta, body = read_note(path)
        body = body.rstrip() + f"\n\n# Run at {run_time}\n\n" + section
    else:
        meta = {"title": f"digest {day}", "date": day, "visibility": "private", "tags": ["digest"]}
        body = f"# Digest {day}\n\n{INTRO}\n" + section
    quoted = [q for p in data["proposals"] + proposals for q in p.get("quoted") or ()]
    hits = check_digest(body, quoted)
    if hits:
        raise ValueError(f"the digest breaks the writing rules: {hits}")
    if dry_run:
        return None, body
    data["proposals"] += proposals
    data["held"] += held
    save_json(side, data)
    write_note(path, meta, body)
    return path, body
