"""The user's own words, for two checks on what the model proposes.

1. Quotes are exempt from the writing rules. A span inside double quotes ("..." or the curly kind)
   is left out of the ban check when its words really appear in the user's own words in one of the
   proposal's sources. The user's own words are the "You" turns of a chat note (not the answers, and
   not the `> quote` lines where the user quoted an answer back), and the body of a note the user
   wrote (`source: author`, or no `source`). A quote found only in an AI answer, in an agent's note,
   or nowhere gets no exemption, so the model can't get a banned word through by quoting itself.

2. Provenance. When a source is a note an agent wrote (`source: codex`, `source: agent`, anything
   but `author`) and a quote in the proposal is found in a "You" turn of a captured chat, that chat
   is added as a source, and the proposal's visibility is worked out again with it.

Matching ignores case, runs of spaces, the kind of quote mark or apostrophe, and the spacing
around a hyphen or dash (tidy() turns a dash into " - "). It matches whole words only, and a quote
needs at least MIN_EXEMPT_WORDS (2) words to be exempt, so "might" in quote marks doesn't pass just
because the user wrote "might" once.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from common.notes import CHATS_DIR, EXCLUDED_DIR, parse

OPEN_QUOTE, CLOSE_QUOTE = chr(0x201C), chr(0x201D)   # curly double quotes
SINGLE_MARKS = "".join(map(chr, (0x2018, 0x2019, 0x201B, 0x2032, 0x60)))    # curly apostrophes, prime, backtick
DOUBLE_MARKS = "".join(map(chr, (0x201C, 0x201D, 0x201E, 0x2033)))
DASH_MARKS = "-" + "".join(map(chr, range(0x2010, 0x2016)))                  # hyphen and the unicode dashes
QUOTE_RX = re.compile('"([^"\\n]+)"|' + OPEN_QUOTE + "([^" + OPEN_QUOTE + CLOSE_QUOTE + "\\n]+)" + CLOSE_QUOTE)
MIN_PROVENANCE_CHARS = 20     # a quote shorter than this is too common to name a chat by
MIN_EXEMPT_WORDS = 2          # a one-word "quote" is a scare quote, and would let any banned word
                              # the user ever used through ("might"), so it gets no exemption
MAX_PROVENANCE_CHATS = 3
_TURN_RX = re.compile(r"^## \d+\. You\b.*$", re.M)
_ANSWER_RX = re.compile(r"^### ", re.M)
AUTHOR_SOURCES = ("", "author")


def norm(text: str) -> str:
    t = str(text or "").lower()
    t = re.sub("[" + SINGLE_MARKS + "]", "'", t)
    t = re.sub("[" + DOUBLE_MARKS + "]", '"', t)
    t = re.sub(r"\s*[" + DASH_MARKS + r"]\s*", "-", t)
    return " ".join(t.split())


def found_in(quote: str, words: str) -> bool:
    """True when the normalized quote appears in the normalized words as whole words, so a quote
    can't match the middle of a longer word."""
    q = norm(quote)
    if not q:
        return False
    return re.search(r"(?<!\w)" + re.escape(q) + r"(?!\w)", words) is not None


def quotes_in(text: str) -> list[str]:
    """The text inside each pair of double quotes on one line."""
    return [(m.group(1) or m.group(2)).strip() for m in QUOTE_RX.finditer(str(text or ""))]


def you_turns(body: str) -> str:
    """The user's side of a chat note: each "You" turn up to its answer heading, minus `>` lines."""
    out = []
    marks = list(_TURN_RX.finditer(body))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        turn = body[m.end():end]
        answer = _ANSWER_RX.search(turn)
        if answer:
            turn = turn[:answer.start()]
        out += [line for line in turn.splitlines() if not line.lstrip().startswith(">")]
    return "\n".join(out)


def is_chat(rel: str) -> bool:
    return rel.startswith(CHATS_DIR.as_posix() + "/")


class OwnWords:
    """The user's own words per vault note, read on first use."""

    def __init__(self, vault: Path):
        self.vault = Path(vault)
        self._notes: dict[str, tuple[str, dict]] = {}
        self._chats: list[str] | None = None

    def _load(self, rel: str) -> tuple[str, dict]:
        if rel not in self._notes:
            try:
                meta, body = parse((self.vault / rel).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
                meta, body = {}, ""
            if is_chat(rel):
                words = you_turns(body)
            elif str(meta.get("source") or "").strip().lower() in AUTHOR_SOURCES:
                words = body
            else:
                words = ""
            self._notes[rel] = (norm(words), meta)
        return self._notes[rel]

    def of(self, rel: str) -> str:
        """The user's words in the note, normalized. '' for an agent's note."""
        return self._load(rel)[0]

    def meta(self, rel: str) -> dict:
        return self._load(rel)[1]

    def agent_written(self, rel: str) -> bool:
        return not is_chat(rel) and str(self.meta(rel).get("source") or "").strip().lower() not in AUTHOR_SOURCES

    def chats(self) -> list[str]:
        if self._chats is None:
            root = self.vault / CHATS_DIR
            excluded = (self.vault / EXCLUDED_DIR).as_posix() + "/"
            files = sorted(p for p in root.rglob("*.md") if p.is_file()) if root.is_dir() else []
            self._chats = [p.relative_to(self.vault).as_posix() for p in files
                           if not p.name.startswith(".") and not p.as_posix().startswith(excluded)]
        return self._chats

    def chats_with(self, quote: str) -> list[str]:
        """Captured chats (not excluded) whose "You" turns hold the quote."""
        q = norm(quote)
        if len(q) < MIN_PROVENANCE_CHARS:
            return []
        out = []
        for rel in self.chats():
            if self.meta(rel).get("excluded"):
                continue
            if found_in(q, self.of(rel)):
                out.append(rel)
                if len(out) >= MAX_PROVENANCE_CHATS:
                    break
        return out


def _texts(p: dict) -> str:
    return "\n".join(str(p.get(k, "")) for k in ("reason", "description", "text"))


def verify(p: dict, words: OwnWords) -> list[str]:
    """Set p["quoted"] to the quotes in the proposal found in the user's words in its sources, as
    whole words. A quote of fewer than MIN_EXEMPT_WORDS words never counts."""
    own = [words.of(rel) for rel in p.get("sources", [])]
    found = []
    for q in quotes_in(_texts(p)):
        if len(norm(q).split()) < MIN_EXEMPT_WORDS:
            continue
        if q not in found and any(found_in(q, w) for w in own):
            found.append(q)
    p["quoted"] = found
    return found


def add_provenance(p: dict, words: OwnWords, visibility_of) -> list[str]:
    """Add the chats a quote came from when a source is an agent's note. Returns the chats added.
    `visibility_of(meta)` is collect.visibility_of."""
    if not any(words.agent_written(rel) for rel in p.get("sources", [])):
        return []
    added = []
    for q in quotes_in(_texts(p)):
        if any(found_in(q, words.of(rel)) for rel in p["sources"]):
            continue
        for rel in words.chats_with(q):
            if rel not in p["sources"]:
                p["sources"].append(rel)
                p["links"].append(Path(rel).stem)
                added.append(rel)
    if added:
        vis = {p["visibility"]} | {visibility_of(words.meta(rel)) for rel in added}
        p["visibility"] = vis.pop() if len(vis) == 1 else "private"
    return added


def strip_quoted(text: str, quoted) -> str:
    """The text with each verified quote, marks and all, taken out, for the ban check."""
    allowed = {norm(q) for q in quoted or ()}
    if not allowed:
        return text
    return QUOTE_RX.sub(lambda m: '""' if norm(m.group(1) or m.group(2)) in allowed else m.group(0), text)
