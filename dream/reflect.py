"""The profile spike: replay a few months of chats into diary entries, a facts
digest and a gist of questions, with every claim checked by script and by a cold fact-checker.

    python -m dream.reflect --vault <copy> --pick 3 --out <folder> --cap-usd 4.50
    python -m dream.reflect --vault <copy> --from 2025-07 --to 2025-07 --out <scratch folder> --cap-usd 2

Per month, in date order (replay):

1. **Pick chats.** The chats dream.filter scored as the user's and about the user (filter.about_user),
   best score first, up to MAX_CHATS and MAX_CALL_CHARS. `--pick N` takes filter.pick_months.
2. **Extract.** One `claude -p` call (propose.call: tools off, pinned model, JSON schema, a dollar
   cap) reads the user's turns in full (trimmed answers as context only), the People and Profile
   notes, the rejected list, and the facts already kept from earlier months (never their
   interpretations). It returns claims: said, told, did (facts), interpretation, contradiction or
   unknown, each with a theme and quotes `{chat, quote}`. "Today" is the month's last day.
3. **Script checks.** Each quote must appear word for word (same whitespace, quote marks and case
   folding on both sides, nothing fuzzy) in a "You" turn of the chat it names, with quoted answer
   lines left out, and be at least MIN_QUOTE_WORDS words. One bad quote drops the claim. Then:
   interpretations and contradictions need quotes from 2 or more chats; interpretations, unknowns
   and contradictions can't name a motive, feeling or trait (FEELING_RX); claims matching
   `2 Areas/Profile/rejected.md` or an excluded topic are dropped; the writing bans get one reword
   call (propose.reword) and anything still failing is held. The script sets the date (the turn the
   quote came from), the confidence (distinct chats: 2 low, 3 or 4 medium, 5+ high) and visibility.
4. **Fact-checker.** A separate cold call that sees only the claims and their quotes, the cited
   chats' "You" turns, the People and Profile notes, the rejected list and the ChatGPT memory
   exports (a second opinion only). It never sees the extractor's prompt or output beyond the
   claims. Statuses: supported, unsupported, invented, contradicted, answered. A claim with no
   verdict counts as unsupported. Only supported and answered claims are kept.

Then one gist call turns the kept interpretations, contradictions, unknowns and working rules into
3 to 5 questions; each must quote the user back with a quote the script already checked, and the
script drops leading or feelings questions. Contradictions go first.

Outputs are rendered by the script from the kept claims, never written by a model, and go only to
`--out` as new files (an existing file is never overwritten, and `2 Areas/Profile/` is refused):
one diary entry per month (five fixed headings), a facts digest (a box per
fact, sources linked), a gist, a list of claims to label, and a README with the counts and cost.

Every model call's result is cached under `<vault state>/profile/reflect/`, keyed by its prompt, so
a rerun pays only for what changed. Spend goes on the ledger line "<day> profile" (shared with
dream.filter). Exit codes: 0 done, 1 failed, 2 refused by a guard.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import calendar
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from capture.chatnote import SOURCE_NAMES
from common import config
from common.notes import read_note, write_note
from dream import collect, digest, filter as filt, guards, propose

EXTRACT_MODEL = propose.MODEL
CHECK_MODEL = propose.MODEL
MAX_CHATS = 25
MAX_CALL_CHARS = 100_000
MAX_TURN_CHARS = 2_500
MAX_CHAT_CHARS = 12_000
ANSWER_CONTEXT_CHARS = 200
MAX_IMPORT_CHARS = 30_000
MAX_CONTEXT_CHARS = 8_000
MIN_QUOTE_WORDS = 4
MIN_QUOTE_CJK = 6
MIN_FRAGMENT_WORDS = 3
MAX_QUESTIONS = 5
MIN_QUESTIONS = 3
EXTRACT_CALL_USD = 1.00
CHECK_CALL_USD = 1.00
GIST_CALL_USD = 0.50
REJECTED_REL = "2 Areas/Profile/rejected.md"
IMPORTS_DIR = "3 Resources/imports"
PROFILE_REL = "2 Areas/Profile"
CONTEXT_DIRS = ("2 Areas/People", "2 Areas/Profile")

FACT_KINDS = ("said", "told")          # the user's own words: headings 1 and 3, and the facts digest
KINDS = FACT_KINDS + ("did", "interpretation", "contradiction", "unknown")
MULTI_CHAT_KINDS = ("interpretation", "contradiction")
THEMES = ("work", "code", "ai", "writing", "school", "taste", "health", "money", "people", "plans",
          "life", "other")
STATUSES = ("supported", "unsupported", "invented", "contradicted", "answered")
KEEP = ("supported", "answered")
QUESTION_KINDS = ("contradiction", "working", "interesting")

HEADINGS = [
    ("said", "What you created or said"),
    ("did", "What Bonsai observed"),
    ("told", "What you explicitly told it"),
    ("interpretation", "What Bonsai inferred, with confidence"),
    ("open", "What Bonsai does not know"),
]

# Words that name a motive, feeling, trait or diagnosis. Banned in interpretations, unknowns,
# contradictions and questions: those are the user's to say, not Bonsai's.
FEELING_RX = re.compile(
    r"\b(feel|feels|felt|feeling|feelings|emotion\w*|afraid|fears?|fearful|anxi\w+|insecur\w+|passion\w*|"
    r"perfectionis\w*|personality|traits?|deep down|motiv\w+|proud|pride|ashamed|shame|lonely|loneliness|"
    r"stress\w*|frustrat\w+|happy|happiness|unhappy|sad|sadness|excit\w+|worr\w+|nervous|guilt\w*|"
    r"self-esteem|identity|diagnos\w+|adhd|autis\w+|depress\w+|values|cares? about|deeply|wants? to be seen|"
    r"struggles? with|insists?|obsess\w*|craves?|longs? for|enjoys?|loves?|hates?)\b", re.I)
LEADING_RX = re.compile(r"\b(don't you|do you not|isn't it|aren't you|wouldn't you|right\?|surely|"
                        r"why do you|why did you|what made you|how do you feel|how did you feel)\b", re.I)
_QUOTE_SPAN_RX = re.compile(r"[\"“]([^\"“”]+)[\"”]")
_SINGLE_QUOTE_RX = re.compile(r"(?<!\w)['‘]([^'‘’\n]+)['’](?!\w)")   # 'like this', not it's
_TURN_HEAD_RX =re.compile(r"^## (\d+)\. You(?: \((\d{4}-\d{2}-\d{2})[^)]*\))?", re.M)
_CJK_RX = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_ELLIPSIS_RX = re.compile(r"\.{3,}|…")
_QMAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"',
                       "”": '"', "„": '"', " ": " ", "‐": "-", "‑": "-"})

SYSTEM_PROMPT = propose.SYSTEM_PROMPT

EXTRACT_INSTRUCTIONS = """\
You read one month of the user's AI chats and list what they show about the user. A script checks
every quote you give against the chat, and a separate checker reads every claim, so write only what
the user's own words show. Use they/them for the user.

Evidence is the user's own words only: the "You" turns. The AI's answers are cut short and shown only
as context. They are never evidence, and nothing the AI said about the user counts. Text the user
pasted in (code, errors, someone else's writing) isn't a statement about the user. Take the user's
words at face value and literally. Jokes and hypotheticals aren't facts. A one-off question shows a
topic, not a preference. If a statement could mean two things, write it as an unknown instead of
picking one.

Kinds of claim:
- said: a fact the user stated about themself: background, history, what they use, like, own, plan
  or decided.
- told: a preference or working rule the user told the AI about how they want it (or their tools)
  to work with them.
- did: something the user plainly did in the chats: what they built, asked for, corrected or chose.
- interpretation: a pattern in what the user said or did across 2 or more different chats. Write it
  as the pattern ("asked for X in two separate chats"), never as a motive, feeling, personality
  trait, value or diagnosis. It needs quotes from 2 or more different chats. Don't put chat keys
  (C1, C2...) in the text: the quotes already say which chats.
- contradiction: two things the user said in different chats that pull different ways. Quote both.
- unknown: something the chats raise about the user but don't settle. Quote what raises it.

Rules:
- Every claim has at least one quote. A quote is the user's exact words copied from one "You" turn
  of the chat you name: at least 5 words, character for character, no changes, no [brackets], no
  ellipsis. Never quote the AI's answer.
- One idea per claim, in one short plain sentence about the user.
- No compliments and no flattery. A pattern that isn't flattering stays if the quotes show it.
- Nothing about health, identity, money trouble or relationships unless the user stated it plainly.
- Nothing about the people in the People notes. Their chats aren't the user's.
- theme: one of {themes}.
- Already noted lists facts kept from earlier months. Don't repeat one unless this month says it
  again; then copy its text exactly and add this month's quote.
- An empty list is a good answer. Most chats are lookups and show nothing about the user.

{rules}
Today is {today}.
"""

CHECK_INSTRUCTIONS = """\
You check claims about the user that were drawn from their AI chats. You report only. Never rewrite
a claim. Give every claim exactly one status:

- answered: the claim matches something in the People or Profile notes below (those are things
  the user told us themself).
- supported: the user's own words in the cited turns say this, read literally. For an
  interpretation, the cited turns really show the same pattern in each chat it names. For an
  unknown, the turns really raise the question and don't settle it. For a contradiction, the two
  quotes really pull different ways.
- unsupported: the words are real but don't say this. The claim reads more into them, generalizes
  from one remark, takes a joke or a hypothetical literally, or names a motive, feeling, value or
  personality trait.
- invented: nothing in the user's turns says anything like this, or it rests on the AI's side of the
  chat, or it's about someone else.
- contradicted: the user's turns, the People or Profile notes, or the rejected list say otherwise.

The ChatGPT memory exports are a second opinion only. They never make a claim supported, but say so
in evidence when they agree or disagree. evidence: under 25 words, what's true. source: the chat
key (C1...), a note name, or "imports". Return one verdict for every claim id (K1, K2...).

The user goes by they/them: write "they", never "he" or "she".

{rules}"""

GIST_INSTRUCTIONS = """\
You write 3 to 5 questions for the user about what was noticed in their chats, so the user can confirm
or correct it in a sentence. The user can ignore any of them.

Rules for each question:
- Quote the user back: include at least one of the quotes given below, copied exactly, inside double
  quotes, and name the claim ids it's based on in based_on.
- Neutral wording. Never leading, never suggest the answer, no "don't you", "isn't it", "right?".
- Only about what the user said or did. Never about feelings, motives, personality or why.
- Answerable in one sentence.
- kind: contradiction (two of the user's statements pull different ways), working (a preference or
  working rule that would change what an agent does), or interesting. Contradictions first.
- Model them on: You said "X" (in one chat) and "Y" (in another). Did your view change, or were
  these about different things?

{rules}"""

CHECK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "status", "evidence", "source"],
        "properties": {"id": {"type": "string"}, "status": {"type": "string", "enum": list(STATUSES)},
                       "evidence": {"type": "string"}, "source": {"type": "string"}}}}},
}

EXTRACT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["claims"],
    "properties": {"claims": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["kind", "theme", "text", "quotes"],
        "properties": {
            "kind": {"type": "string", "enum": list(KINDS)},
            "theme": {"type": "string", "enum": list(THEMES)},
            "text": {"type": "string"},
            "quotes": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["chat", "quote"],
                "properties": {"chat": {"type": "string"}, "quote": {"type": "string"}}}},
        }}}},
}

GIST_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["questions"],
    "properties": {"questions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["kind", "based_on", "text"],
        "properties": {"kind": {"type": "string", "enum": list(QUESTION_KINDS)},
                       "based_on": {"type": "array", "items": {"type": "string"}},
                       "text": {"type": "string"}}}}},
}


class ReflectError(RuntimeError):
    pass


# --- The user's turns ----------------------------------------------------------------------------

@dataclass
class Turn:
    n: int
    date: str
    text: str            # the user's words, quoted answer lines left out
    answer: str          # the start of the answer, context only


@dataclass
class ChatDoc:
    rel: str
    stem: str
    title: str
    date: str
    source: str
    visibility: str
    score: int
    turns: list[Turn]
    key: str = ""


def evidence_text(turn_text: str, source: str) -> tuple[str, str]:
    """(the user's words in one `## n. You` turn, the start of the answer). Every `>` line in
    the user's part is left out: those quote the AI's answer, so they're never the user's words."""
    lines = turn_text.split("\n")[1:]
    answer_head = f"### {SOURCE_NAMES.get(source, source)}"
    cut = None
    if answer_head in lines:
        cut = lines.index(answer_head)
    else:
        cut = next((i for i, l in enumerate(lines) if l.startswith("### ")), None)
    mine = lines if cut is None else lines[:cut]
    answer = [] if cut is None else lines[cut + 1:]
    mine = [l for l in mine if not l.lstrip().startswith(">")]
    return "\n".join(mine).strip(), " ".join(" ".join(answer).split())


def read_turns(body: str, source: str) -> list[Turn]:
    out = []
    for n, text in collect.split_turns(body):
        head = _TURN_HEAD_RX.match(text)
        date = (head.group(2) if head and head.group(2) else "") or ""
        mine, answer = evidence_text(text, source)
        out.append(Turn(n, date, mine, answer))
    return out


def load_chat(vault: Path, rel: str, score: int = 0) -> ChatDoc | None:
    try:
        meta, body = read_note(Path(vault) / rel)
    except Exception:
        return None
    source = str(meta.get("source") or "")
    turns = read_turns(body, source)
    if not turns:
        return None
    date = str(meta.get("created") or meta.get("updated") or "")[:10]
    for t in turns:
        t.date = t.date or date
    return ChatDoc(rel, Path(rel).stem, str(meta.get("title") or Path(rel).stem), date, source,
                   collect.visibility_of(meta), int(score), turns)


# --- the quote check ----------------------------------------------------------------------------

def norm(text: str) -> str:
    """The one normalization used on both sides of the quote check: NFKC, curly quotes made
    straight, no-break spaces made plain, whitespace collapsed, case folded. Nothing fuzzy."""
    text = unicodedata.normalize("NFKC", str(text)).translate(_QMAP)
    return " ".join(text.split()).casefold()


def clean_quote(q: str) -> str:
    """The quote as the model gave it, with outer whitespace, outer quote marks and trailing
    punctuation taken off (which keeps it a substring of what it quoted)."""
    q = str(q or "").strip()
    q = q.strip("\"'“”‘’ ")
    return q.rstrip(".,;:!? ").lstrip(" ")


def long_enough(q: str) -> bool:
    return len(re.findall(r"\w+", q)) >= MIN_QUOTE_WORDS or len(_CJK_RX.findall(q)) >= MIN_QUOTE_CJK


def find_quote(quote: str, chat: ChatDoc) -> tuple[Turn | None, str]:
    """(the turn the quote is in, why not). An ellipsis splits the quote into pieces that must all
    be in the same turn, in order, each MIN_FRAGMENT_WORDS words or more."""
    q = clean_quote(quote)
    if not q:
        return None, "empty quote"
    if not long_enough(_ELLIPSIS_RX.sub(" ", q)):
        return None, "quote too short"
    pieces = [p.strip() for p in _ELLIPSIS_RX.split(q)]
    pieces = [p for p in pieces if p]
    if len(pieces) > 1 and any(len(re.findall(r"\w+", p)) < MIN_FRAGMENT_WORDS
                               and len(_CJK_RX.findall(p)) < MIN_QUOTE_CJK for p in pieces):
        return None, "quote piece too short around an ellipsis"
    wanted = [norm(p) for p in pieces]
    for t in chat.turns:
        hay = norm(t.text)
        pos, ok = 0, True
        for w in wanted:
            i = hay.find(w, pos)
            if i < 0:
                ok = False
                break
            pos = i + len(w)
        if ok:
            return t, ""
    return None, "not found in a You turn of that chat"


@dataclass
class Claim:
    id: str
    month: str
    kind: str
    theme: str
    text: str
    quotes: list[dict] = field(default_factory=list)   # {rel, stem, turn, date, quote}
    status: str = ""          # fact-checker status
    evidence: str = ""
    checker_source: str = ""
    checker_note: str = ""    # the checker's note when it broke the writing rules (run record only)
    dropped: str = ""         # why the script or the checker dropped it
    visibility: str = "private"

    @property
    def chats(self) -> list[str]:
        return list(dict.fromkeys(q["rel"] for q in self.quotes))

    @property
    def stems(self) -> list[str]:
        return list(dict.fromkeys(q["stem"] for q in self.quotes))

    @property
    def date(self) -> str:
        dates = sorted(q["date"] for q in self.quotes if q.get("date"))
        return dates[0] if dates else ""

    @property
    def is_fact(self) -> bool:
        return self.kind in FACT_KINDS

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "month", "kind", "theme", "text", "quotes", "status",
                                              "evidence", "checker_source", "checker_note", "dropped",
                                              "visibility")}


def confidence(n_chats: int) -> str:
    """From the count of separate chats (Honcho's floor): 2 low, 3 or 4 medium, 5+ high."""
    if n_chats >= 5:
        return "high"
    if n_chats >= 3:
        return "medium"
    if n_chats == 2:
        return "low"
    return "one chat"


def rejected_lines(vault: Path) -> list[str]:
    path = Path(vault) / REJECTED_REL
    if not path.is_file():
        return []
    try:
        _, body = read_note(path)
    except Exception:
        return []
    out = []
    for line in body.splitlines():
        m = re.match(r"^\s*[-*]\s+(?:\[.\]\s+)?(.+)$", line)
        if m:
            out.append(m.group(1).strip())
    return out


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", norm(text)))


def matches_rejected(text: str, rejected: list[str]) -> str | None:
    """The rejected line this claim matches: the same words, or 60% or more of them shared."""
    mine = _words(text)
    for r in rejected:
        theirs = _words(r)
        if not mine or not theirs:
            continue
        if norm(r) in norm(text) or len(mine & theirs) / len(mine | theirs) >= 0.6:
            return r
    return None


def check_claims(raw: dict, month: str, chats: list[ChatDoc], rejected: list[str], topics: list[str],
                 start: int = 1) -> tuple[list[Claim], list[Claim], Counter]:
    """(claims that passed, claims dropped, counts). Quotes are checked on the raw model output,
    before any tidying, and never changed."""
    by_key = {c.key: c for c in chats}
    kept, dropped = [], []
    counts: Counter = Counter()
    n = start
    seen: dict[str, Claim] = {}
    for item in raw.get("claims") or []:
        counts["extracted"] += 1
        kind = item.get("kind")
        claim = Claim(f"{month}#{n:02d}", month, kind if kind in KINDS else str(kind),
                      item.get("theme") if item.get("theme") in THEMES else "other",
                      str(item.get("text") or "").strip())
        n += 1
        why = None
        quote_failed = False
        if kind not in KINDS:
            why = "unknown kind"
        elif not claim.text:
            why = "no text"
        quotes = item.get("quotes") or []
        if not why and not quotes:
            why, quote_failed = "quote check: no quote", True
        for q in quotes if not why else []:
            counts["quotes"] += 1
            chat = by_key.get(str(q.get("chat", "")).strip())
            turn, miss = find_quote(q.get("quote", ""), chat) if chat else (None, "names a chat that isn't in the month")
            if not turn:
                why, quote_failed = f"quote check: {miss}", True
                counts["quotes failed"] += 1
                break
            claim.quotes.append({"rel": chat.rel, "stem": chat.stem, "turn": turn.n, "date": turn.date,
                                 "quote": " ".join(clean_quote(q.get("quote", "")).split())})
        if quote_failed:
            counts["dropped by quote check"] += 1
        if not why and claim.kind in MULTI_CHAT_KINDS and len(claim.chats) < 2:
            why = "needs quotes from 2 or more chats"
        if not why and claim.kind not in FACT_KINDS and FEELING_RX.search(claim.text):
            why = f"names a motive, feeling or trait: {FEELING_RX.search(claim.text).group(0)!r}"
        if not why:
            r = matches_rejected(claim.text, rejected)
            if r:
                why = f"matches the rejected list: {r[:80]}"
        if not why:
            blob = claim.text + "\n" + "\n".join(q["quote"] for q in claim.quotes)
            t = filt.excluded_topic(blob, topics)
            if t:
                why = "an excluded topic"
        if not why:
            vis = {by_key[k].visibility for k in by_key if by_key[k].rel in claim.chats}
            claim.visibility = vis.pop() if len(vis) == 1 else "private"
            claim.text = name_chats(propose.tidy(claim.text).replace("\n", " "), by_key)
            key = norm(claim.text) + "|" + claim.kind
            if key in seen:   # the same claim twice: one claim, both sets of quotes
                old = seen[key]
                old.quotes += [q for q in claim.quotes if q not in old.quotes]
                counts["merged repeats"] += 1
                continue
            seen[key] = claim
            kept.append(claim)
        else:
            claim.dropped = why
            dropped.append(claim)
            if not quote_failed:
                counts["dropped by rules"] += 1
    return kept, dropped, counts


# --- prompts ------------------------------------------------------------------------------------

def profile_context(vault: Path) -> str:
    """The People and Profile notes (rejected.md left out, since it's passed on its own), with each
    note's source, capped at MAX_CONTEXT_CHARS."""
    parts = []
    for rel in CONTEXT_DIRS:
        folder = Path(vault) / rel
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            if f"{rel}/{path.name}" == REJECTED_REL:
                continue
            try:
                meta, body = read_note(path)
            except Exception:
                continue
            if meta.get("excluded"):
                continue
            parts.append(f"## {rel}/{path.stem} (source: {meta.get('source', 'unknown')})\n\n{body.strip()}\n")
    return "\n".join(parts)[:MAX_CONTEXT_CHARS]


def imports_text(vault: Path) -> str:
    folder = Path(vault) / IMPORTS_DIR
    if not folder.is_dir():
        return ""
    parts = []
    for path in sorted(folder.glob("*.md")):
        try:
            _, body = read_note(path)
        except Exception:
            continue
        parts.append(f"## {path.stem}\n\n{body.strip()}\n")
    return "\n".join(parts)[:MAX_IMPORT_CHARS]


def _cut(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " [cut]"


def chat_block(c: ChatDoc, answers: bool = True, turns: set[int] | None = None) -> str:
    lines = [f"## {c.key}\n\ntitle: {c.title}\nstarted {c.date}, {len(c.turns)} turns\n"]
    size = 0
    shown = [t for t in c.turns if turns is None or t.n in turns]
    for i, t in enumerate(shown):
        block = f"\nTurn {t.n} ({t.date}), You:\n{_cut(t.text, MAX_TURN_CHARS) or '(no text)'}\n"
        if answers and t.answer:
            block += f"(AI answer, context only: {_cut(t.answer, ANSWER_CONTEXT_CHARS)})\n"
        if size and size + len(block) > MAX_CHAT_CHARS:
            lines.append(f"\n[{len(shown) - i} more turns not shown]\n")
            break
        lines.append(block)
        size += len(block)
    return "".join(lines)


def month_end(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"{month}-{calendar.monthrange(y, m)[1]:02d}"


def select_chats(vault: Path, scores: dict, kept_rels: set[str], month: str,
                 max_chats: int = MAX_CHATS, max_chars: int = MAX_CALL_CHARS) -> list[ChatDoc]:
    """The chats about the user with turns in `month`, best score first, keyed C1, C2..., within the
    size caps. Only the turns dated in that month are kept (shown to the model and searched by the
    quote check), so a chat started in July and picked up again in December gives its July turns to
    July and its December turns to December, and no month sees a later month's words."""
    rows = [(rel, e) for rel, e in scores.items()
            if rel in kept_rels and str(e.get("month", "")) <= month and filt.about_user(e)]
    rows.sort(key=lambda r: (-int(r[1].get("score", 0)), r[1].get("date", ""), r[0]))
    out, size = [], 0
    for rel, e in rows:
        if len(out) >= max_chats:
            break
        c = load_chat(vault, rel, int(e.get("score", 0)))
        if not c:
            continue
        c.turns = [t for t in c.turns if t.date[:7] == month]
        if not c.turns:
            continue
        c.key = f"C{len(out) + 1}"
        n = len(chat_block(c))
        if out and size + n > max_chars:
            continue
        out.append(c)
        size += n
    return out


def extract_prompt(month: str, chats: list[ChatDoc], noted: list[str], context: str, rejected: list[str]) -> str:
    parts = [EXTRACT_INSTRUCTIONS.format(themes=", ".join(THEMES), rules=propose.WRITING_RULES,
                                         today=month_end(month))]
    if context:
        parts.append("\n# People and Profile notes (what the user has told us)\n\n" + context)
    parts.append("\n# Rejected (the user said these are wrong; never propose them again)\n\n"
                 + ("".join(f"- {r}\n" for r in rejected) or "(none)\n"))
    parts.append("\n# Already noted (facts kept from earlier months)\n\n"
                 + ("".join(f"- {f}\n" for f in noted) or "(none)\n"))
    parts.append(f"\n# Chats from {month}\n\n")
    parts += [chat_block(c) + "\n" for c in chats]
    return "".join(parts)


def check_prompt(claims: list[Claim], chats: list[ChatDoc], context: str, rejected: list[str],
                 imports: str) -> tuple[str, dict[str, Claim]]:
    by_rel = {c.rel: c for c in chats}
    ids = {f"K{i}": c for i, c in enumerate(claims, 1)}
    parts = [CHECK_INSTRUCTIONS.format(rules=propose.WRITING_RULES), "\n# Claims\n\n"]
    for kid, c in ids.items():
        qs = "; ".join(f'{by_rel[q["rel"]].key} turn {q["turn"]}: "{q["quote"]}"' for q in c.quotes)
        parts.append(f"- {kid} [{c.kind}] {c.text} | quotes: {qs}\n")
    parts.append("\n# The cited chats (the user's turns only)\n\n")
    for chat in chats:
        if any(chat.rel in cl.chats for cl in claims):
            parts.append(chat_block(chat, answers=False) + "\n")
    parts.append("\n# People and Profile notes\n\n" + (context or "(none)\n"))
    parts.append("\n# Rejected list\n\n" + ("".join(f"- {r}\n" for r in rejected) or "(none)\n"))
    parts.append("\n# ChatGPT memory exports (a second opinion only)\n\n" + (imports or "(none)\n"))
    return "".join(parts), ids


_KEY_RX = re.compile(r"\bC(\d+)\b")


def name_chats(text: str, by_key: dict[str, "ChatDoc"]) -> str:
    """Chat keys (C3) in model text become links to the chat, since a key means nothing outside
    the call it was made for."""
    return _KEY_RX.sub(lambda m: link(by_key[m.group(0)].stem) if m.group(0) in by_key else m.group(0), text)


def apply_verdicts(raw: dict, ids: dict[str, Claim], by_key: dict[str, "ChatDoc"] | None = None) -> Counter:
    counts: Counter = Counter()
    seen = set()
    for v in raw.get("verdicts") or []:
        kid = str(v.get("id", "")).strip()
        if kid not in ids or kid in seen or v.get("status") not in STATUSES:
            continue
        seen.add(kid)
        c = ids[kid]
        c.status = v["status"]
        c.evidence = name_chats(propose._one_line(v.get("evidence", ""))[:300], by_key or {})
        if check_written(c.evidence):   # the checker's own words break the writing rules
            c.evidence = "(the checker's note broke the writing rules, so it's left out; it's in the run record)"
            c.checker_note = propose._one_line(v.get("evidence", ""))[:300]
        c.checker_source = propose._one_line(v.get("source", ""))[:80]
    for kid, c in ids.items():
        if kid not in seen:
            c.status = "unsupported"
            c.evidence = "the fact-checker gave no verdict"
            counts["no verdict"] += 1
        counts[c.status] += 1
        if c.status not in KEEP:
            c.dropped = f"fact-checker: {c.status}"
    return counts


# --- questions ----------------------------------------------------------------------------------

def question_candidates(claims: list[Claim]) -> dict[str, Claim]:
    order = {"contradiction": 0, "told": 1, "interpretation": 2, "unknown": 3}
    pool = [c for c in claims if c.kind in order]
    pool.sort(key=lambda c: (order[c.kind], -len(c.chats), c.id))
    return {f"G{i}": c for i, c in enumerate(pool[:30], 1)}


def gist_prompt(cands: dict[str, Claim]) -> str:
    parts = [GIST_INSTRUCTIONS.format(rules=propose.WRITING_RULES), "\n# What was noticed\n\n"]
    for gid, c in cands.items():
        qs = "; ".join(f'"{q["quote"]}" ({q["date"]})' for q in c.quotes)
        parts.append(f"- {gid} [{c.kind}] {c.text} | quotes: {qs}\n")
    return "".join(parts)


def outside_quotes(text: str, fn) -> str:
    """`text` with `fn` applied to every stretch outside a quoted span."""
    out, last = [], 0
    for m in _QUOTE_SPAN_RX.finditer(text):
        out += [fn(text[last:m.start()]), m.group(0)]
        last = m.end()
    out.append(fn(text[last:]))
    return " ".join("".join(out).split())


def check_question(item: dict, cands: dict[str, Claim]) -> tuple[dict | None, str]:
    """A question the script can stand behind, or (None, why). Every quoted span must be inside a
    quote the script already checked for the claims it's based on."""
    kind = item.get("kind")
    based = [g for g in dict.fromkeys(item.get("based_on") or []) if g in cands]
    text = " ".join(str(item.get("text") or "").split())
    if kind not in QUESTION_KINDS:
        return None, "unknown kind"
    if not based:
        return None, "based on no known claim"
    if not text.endswith("?"):
        return None, "not a question"
    spans = _QUOTE_SPAN_RX.findall(text)
    if not spans:
        return None, "doesn't quote the user"
    pool = [norm(q["quote"]) for g in based for q in cands[g].quotes]
    for s in spans:
        if not any(norm(clean_quote(s)) in p for p in pool) or not clean_quote(s):
            return None, "a quote isn't one of the checked quotes"
    outside = _QUOTE_SPAN_RX.sub(" ", text)
    if FEELING_RX.search(outside):
        return None, f"asks about a motive or feeling: {FEELING_RX.search(outside).group(0)!r}"
    if LEADING_RX.search(outside):
        return None, f"leading: {LEADING_RX.search(outside).group(0)!r}"
    # dashes outside the quotes become spaced hyphens; quotes stay exactly as written
    fixed = outside_quotes(text, lambda s: re.sub(r"\s*[" + propose.DASHES + r"]\s*", " - ", s))
    hits = digest.check_text(_QUOTE_SPAN_RX.sub(" ", fixed))
    if hits:
        return None, "breaks the writing rules: " + ", ".join(hits)
    return {"kind": kind, "based_on": based, "text": fixed,
            "stems": list(dict.fromkeys(s for g in based for s in cands[g].stems)),
            "months": sorted({cands[g].month for g in based})}, ""


def pick_questions(raw: dict, cands: dict[str, Claim]) -> tuple[list[dict], list[str]]:
    good, bad = [], []
    for item in raw.get("questions") or []:
        q, why = check_question(item, cands)
        if q:
            good.append(q)
        else:
            bad.append(why)
    good.sort(key=lambda q: QUESTION_KINDS.index(q["kind"]))
    return good[:MAX_QUESTIONS], bad


# --- the cached call ----------------------------------------------------------------------------

def cached_call(cache_dir: Path, name: str, prompt: str, schema: dict, budget, cwd: Path, runner, stop,
                model: str, max_call_usd: float, log=print) -> tuple[dict, float, bool]:
    """(structured output, what the call behind it cost, came from the cache). A cached answer
    reports what it cost when it was paid for, so the README's costs cover every call behind the
    notes; the ledger is charged only for new calls. The cache file is keyed by a hash of the prompt
    and schema, so a changed prompt calls again."""
    key = hashlib.sha256((model + "\n" + json.dumps(schema, sort_keys=True) + "\n" + prompt).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"{name}.json"
    hit = guards.load_json(path, {})
    if hit.get("key") == key and isinstance(hit.get("output"), dict):
        log(f"reflect: {name} from the cache (paid ${hit.get('cost', 0):.4f} on {hit.get('when', '?')})")
        return hit["output"], float(hit.get("cost") or 0.0), True
    out, cost, models = propose.call(prompt, schema, budget, cwd, runner, stop, model=model,
                                     max_call_usd=max_call_usd)
    guards.save_json(path, {"key": key, "output": out, "cost": cost, "models": models,
                            "when": _dt.datetime.now().isoformat(timespec="seconds")})
    return out, cost, False


# --- rendering ----------------------------------------------------------------------------------

def link(stem: str) -> str:
    return digest.wikilink(stem)


def quote_lines(c: Claim, indent: str = "    ") -> str:
    out = []
    for q in c.quotes:
        out.append(f'{indent}> "{q["quote"]}" ({link(q["stem"])}, turn {q["turn"]}, {q["date"]})\n')
    return "".join(out)


def claim_label(c: Claim) -> str:
    if c.is_fact:
        return "fact"
    if c.kind == "did":
        return "observed"
    if c.kind == "interpretation":
        return f"interpretation, {confidence(len(c.chats))} confidence: {len(c.chats)} chats"
    return "contradiction" if c.kind == "contradiction" else "open"


def heading_of(c: Claim) -> str:
    return c.kind if c.kind in ("said", "did", "told", "interpretation") else "open"


def visibility(claims: list[Claim]) -> str:
    vis = {c.visibility for c in claims}
    return vis.pop() if len(vis) == 1 else "private"


def render_diary(month: str, claims: list[Claim], n_chats: int, today: str,
                 questions: list[dict] | None = None) -> tuple[dict, str]:
    meta = {"title": f"Profile diary {month}", "date": month_end(month), "type": "profile-diary",
            "source": "agent", "learned": today, "visibility": visibility(claims) if claims else "private",
            "tags": ["profile", "diary", "spike"]}
    body = [f"# Profile diary, {month}\n\n",
            f"Replay entry for {month}, written {today} by the profile spike from {n_chats} of your chats "
            f"from that month. Only your own turns count as evidence. A script checked every quote below "
            f"against the chat it links, and a separate fact-checker read every line. Facts (your own "
            f"words) sit under the first and third headings, what Bonsai saw you do under the second, and "
            f"interpretations only under the fourth, with a confidence from the number of chats behind "
            f"each.\n"]
    for key, head in HEADINGS:
        rows = [c for c in claims if heading_of(c) == key]
        body.append(f"\n## {head}\n\n")
        if not rows:
            body.append(f"Nothing in {month}.\n")
            continue
        for c in sorted(rows, key=lambda c: (c.date, c.id)):
            body.append(f"- [{claim_label(c)}] {c.text} ({', '.join(link(s) for s in c.stems)}, {c.date})\n")
            body.append(quote_lines(c))
    mine = [q for q in questions or [] if month in q.get("months", [])]
    body.append("\n## Questions\n\n")
    if not mine:
        body.append(f"None from {month}. The gist next to this note has the questions for the whole run.\n")
    for i, q in enumerate(mine, 1):
        body.append(f"{i}. {q['text']} ({', '.join(link(s) for s in q['stems'])})\n")
    return meta, "".join(body)


def merge_facts(claims: list[Claim]) -> list[dict]:
    """Facts with the same words across months become one digest line with every source."""
    out: dict[str, dict] = {}
    for c in claims:
        if not c.is_fact:
            continue
        k = norm(c.text)
        row = out.setdefault(k, {"text": c.text, "kind": c.kind, "theme": c.theme, "claims": []})
        row["claims"].append(c)
    rows = list(out.values())
    for r in rows:
        r["chats"] = list(dict.fromkeys(rel for c in r["claims"] for rel in c.chats))
        r["stems"] = list(dict.fromkeys(s for c in r["claims"] for s in c.stems))
        r["dates"] = sorted({c.date for c in r["claims"] if c.date})
    return rows


def render_digest(facts: list[dict], months: list[str], today: str) -> tuple[dict, str]:
    meta = {"title": f"Profile spike facts {today}", "date": today, "type": "digest", "source": "agent",
            "visibility": visibility([c for f in facts for c in f["claims"]]) if facts else "private",
            "tags": ["profile", "digest", "spike"]}
    body = [f"# Profile spike facts, {today}\n\n",
            f"Facts about you from your chats in {', '.join(months)}, one box each, grouped by theme. "
            f"Each one passed the script's quote check and the fact-checker. Ticking a box changes "
            f"nothing yet: this is a spike, so nothing goes into `{PROFILE_REL}/` on its own. The date "
            f"is when you said it; every fact was learned {today}.\n"]
    themes = [t for t in THEMES if any(f["theme"] == t for f in facts)]
    if not facts:
        body.append("\nNo facts passed both checks.\n")
    for t in themes:
        body.append(f"\n## {t.capitalize()}\n\n")
        for f in sorted((f for f in facts if f["theme"] == t), key=lambda f: (-len(f["chats"]), f["dates"])):
            n = len(f["chats"])
            src = ", ".join(link(s) for s in f["stems"])
            when = f["dates"][0] if len(f["dates"]) == 1 else f"{f['dates'][0]} to {f['dates'][-1]}"
            body.append(f"- [ ] {f['text']} | {f['kind']} | {when} | {n} chat{'s' if n != 1 else ''} | sources: {src}\n")
            for c in f["claims"]:
                body.append(quote_lines(c))
    return meta, "".join(body)


def render_gist(questions: list[dict], facts: list[dict], kept: list[Claim], months: list[str],
                today: str, note: str = "") -> tuple[dict, str]:
    meta = {"title": f"Profile spike gist {today}", "date": today, "type": "gist", "source": "agent",
            "visibility": visibility(kept) if kept else "private", "tags": ["profile", "gist", "spike"]}
    n_int = sum(1 for c in kept if c.kind == "interpretation")
    body = [f"# Profile spike gist, {today}\n\n", "## What's new\n\n",
            f"- From your chats in {', '.join(months)}: {len(facts)} facts and {n_int} interpretations kept "
            f"after the quote check and the fact-checker. They're in the diary entries and the facts digest "
            f"next to this note.\n"]
    for f in sorted(facts, key=lambda f: -len(f["chats"]))[:3]:
        body.append(f"- {f['text']} ({', '.join(link(s) for s in f['stems'])})\n")
    body.append("\n## Questions\n\nAnswer any of these under the question in a sentence, or ignore them. "
                "Ignoring one changes nothing.\n")
    if not questions:
        body.append("\nNo questions passed the checks this time.\n")
    for i, q in enumerate(questions, 1):
        body.append(f"\n{i}. {q['text']} ({', '.join(link(s) for s in q['stems'])})\n\n   Your answer:\n")
    if note:
        body.append(f"\n{note}\n")
    return meta, "".join(body)


def render_labels(claims: list[Claim], today: str) -> tuple[dict, str]:
    meta = {"title": f"Profile spike claims to label {today}", "date": today, "type": "review",
            "source": "agent", "visibility": visibility(claims) if claims else "private",
            "tags": ["profile", "spike", "labels"]}
    body = [f"# Profile spike claims to label, {today}\n\n",
            "Every claim whose quotes passed the script's quote check, with the fact-checker's verdict, "
            "kept or not. To label one, put true, unsupported, invented or wrong after `label:` "
            "(MirageBench's four buckets). Claims whose quotes weren't found "
            "are left out of the vault; their count is in the README.\n"]
    for status in STATUSES:
        rows = [c for c in claims if c.status == status]
        if not rows:
            continue
        body.append(f"\n## Fact-checker said {status} ({len(rows)})\n\n")
        for c in sorted(rows, key=lambda c: c.id):
            body.append(f"- {c.id} [{claim_label(c)}] {c.text} ({', '.join(link(s) for s in c.stems)})\n"
                        f"    - checker: {c.evidence or '(no evidence given)'}\n    - label:\n")
            body.append(quote_lines(c))
    return meta, "".join(body)


def check_written(text: str) -> list[str]:
    """The writing bans over the script's and the model's words: quoted lines (the user's own words,
    kept exactly) and links are left out."""
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith(">"):
            continue
        line = _SINGLE_QUOTE_RX.sub(" ", _QUOTE_SPAN_RX.sub(" ", line))
        lines.append(digest._LINK_RX.sub("", line))
    return digest.check_text("\n".join(lines))


# --- the run ------------------------------------------------------------------------------------

def _refuse_out(out: Path, vault: Path, real_vault: Path) -> None:
    for v in {Path(vault), Path(real_vault)}:
        prof = (v / PROFILE_REL).resolve()
        try:
            Path(out).resolve().relative_to(prof)
            raise guards.Refused(f"the spike never writes under {PROFILE_REL}/: {out}")
        except ValueError:
            pass


def write_new(out: Path, files: dict[str, tuple[dict, str]]) -> list[Path]:
    """Write each file as new. Refuses before writing anything if one already exists."""
    out = Path(out)
    clash = [n for n in files if (out / n).exists()]
    if clash:
        raise guards.Refused(f"these already exist in {out}, and the spike never overwrites: {', '.join(clash)}")
    written = []
    for name, (meta, body) in files.items():
        path = out / name
        write_note(path, meta, body)
        written.append(path)
    return written


def run(cfg, vault: Path | None = None, months: list[str] | None = None, pick: int | None = None,
        out: Path | None = None, cap_usd: float | None = None, runner=None, now: _dt.datetime | None = None,
        max_chats: int = MAX_CHATS, dry_run: bool = False, log=None) -> int:
    now = now or _dt.datetime.now()
    day = now.date().isoformat()
    state = Path(cfg.state)
    state.mkdir(parents=True, exist_ok=True)
    say = log or (lambda m: guards.log(state, m))
    try:
        vault = Path(vault) if vault else cfg.vault
        if not vault.is_dir():
            raise guards.Refused(f"no vault at {vault}")
        if cap_usd is None:
            raise guards.Refused("pass --cap-usd: reflect has no default cap")
        if out is None and not dry_run:
            raise guards.Refused("pass --out: the folder the spike's notes go in")
        if out is not None:
            _refuse_out(out, vault, cfg.vault)
        filt.check_stops(state, vault, cfg.vault)
        with filt.profile_lock(state, now):
            return _run(cfg, state, vault, months, pick, out, cap_usd, runner, day, max_chats, dry_run, say)
    except guards.Refused as e:
        say(f"reflect: refused: {e}")
        return 2


def _run(cfg, state, vault, months, pick, out, cap_usd, runner, day, max_chats, dry_run, say) -> int:
    pstate = filt.profile_state(state, vault, cfg.vault)
    cache = pstate / "reflect"
    cwd = state / filt.PROFILE_DIR / "work"
    stop = lambda: filt.check_stops(state, vault, cfg.vault)
    budget = guards.Budget(state, filt.spend_key(day), cap_usd)
    scores = filt.load_scores(pstate)
    topics = filt.exclude_topics(state)
    kept_chats, _ = filt.kept_chats(vault, topics, say)
    kept_rels = {c.rel for c in kept_chats}
    counts_by_month = filt.month_counts(scores, kept_rels)
    if not months:
        months = filt.pick_months(counts_by_month, pick or 3)
    if not months:
        say("reflect: no scored chats, so nothing to reflect on (run dream.filter first)")
        return 1
    say(f"reflect: months {', '.join(months)}")
    context = profile_context(vault)
    rejected = rejected_lines(vault)
    imports = imports_text(vault)
    stats: Counter = Counter()
    costs: dict[str, float] = {"extract": 0.0, "reword": 0.0, "fact-check": 0.0, "gist": 0.0}
    noted: list[str] = []
    all_kept: list[Claim] = []
    all_checked: list[Claim] = []     # passed the quote check and rules, with verdicts
    all_dropped: list[Claim] = []
    per_month: dict[str, dict] = {}
    error = None
    for month in months:
        chats = select_chats(vault, scores, kept_rels, month, max_chats)
        per_month[month] = {"chats": len(chats), "about": counts_by_month.get(month, {}).get("about", 0)}
        stats["chats read"] += len(chats)
        if not chats:
            say(f"reflect: {month} has no chats about the user")
            continue
        try:
            raw, cost, _ = cached_call(cache, f"{month} extract", extract_prompt(month, chats, noted, context, rejected),
                                       EXTRACT_SCHEMA, budget, cwd, runner, stop, EXTRACT_MODEL, EXTRACT_CALL_USD, say)
            costs["extract"] += cost
            kept, dropped, c = check_claims(raw, month, chats, rejected, topics)
            stats.update(c)
            kept, held, cost = reword_claims(kept, month, cache, budget, cwd, runner, stop, say)
            costs["reword"] += cost
            stats["held by writing rules"] += len(held)
            dropped += held
            all_dropped += dropped
            if kept:
                prompt, ids = check_prompt(kept, chats, context, rejected, imports)
                raw_v, cost, _ = cached_call(cache, f"{month} check", prompt, CHECK_SCHEMA, budget, cwd, runner,
                                             stop, CHECK_MODEL, CHECK_CALL_USD, say)
                costs["fact-check"] += cost
                stats.update({f"checker {k}": v for k, v in apply_verdicts(raw_v, ids, {c.key: c for c in chats}).items()})
            all_checked += kept
            month_kept = [k for k in kept if k.status in KEEP]
            all_kept += month_kept
            per_month[month]["kept"] = len(month_kept)
            noted += [f"{k.text} ({k.date})" for k in month_kept if k.is_fact]
            say(f"reflect: {month}: {len(chats)} chats, {c['extracted']} claims, {c['dropped by quote check']} "
                f"dropped by the quote check, {len(kept)} to the checker, {len(month_kept)} kept")
        except propose.ProposeError as e:
            error = f"{month}: {e}"
            say(f"reflect: {error}")
            break
    # questions
    questions: list[dict] = []
    q_bad: list[str] = []
    q_note = ""
    cands = question_candidates(all_kept)
    if cands and not error:
        try:
            raw_q, cost, _ = cached_call(cache, "gist " + "+".join(months), gist_prompt(cands), GIST_SCHEMA,
                                         budget, cwd, runner, stop, CHECK_MODEL, GIST_CALL_USD, say)
            costs["gist"] += cost
            questions, q_bad = pick_questions(raw_q, cands)
        except propose.ProposeError as e:
            error = f"gist: {e}"
            say(f"reflect: {error}")
    if len(questions) < MIN_QUESTIONS:
        q_note = (f"Only {len(questions)} question{'s' if len(questions) != 1 else ''} passed the checks "
                  f"(the rule is 3 to 5), so this gist is short.")
    stats["questions kept"] = len(questions)
    stats["questions dropped"] = len(q_bad)
    facts = merge_facts(all_kept)
    files: dict[str, tuple[dict, str]] = {}
    for month in months:
        files[f"spike diary {month}.md"] = render_diary(month, [c for c in all_kept if c.month == month],
                                                        per_month.get(month, {}).get("chats", 0), day, questions)
    files["spike facts digest.md"] = render_digest(facts, months, day)
    files["spike gist.md"] = render_gist(questions, facts, all_kept, months, day, q_note)
    files["spike claims to label.md"] = render_labels(all_checked, day)
    record = {"day": day, "vault": str(vault), "months": months, "per_month": per_month, "stats": dict(stats),
              "costs": dict(costs), "spent_on_ledger": budget.spent(), "cap": cap_usd, "error": error,
              "questions_dropped": q_bad, "claims": [c.to_dict() for c in all_checked],
              "dropped": [c.to_dict() for c in all_dropped], "month_table": counts_by_month,
              "excluded_topics": len(topics),
              "codex_chats": sum(1 for c in kept_chats if c.source == "codex"),
              "filter_runs": guards.load_json(pstate / filt.FILTER_RUNS, []),
              "models": {"extract": EXTRACT_MODEL, "check": CHECK_MODEL, "gist": CHECK_MODEL,
                         "filter": filt.FILTER_MODEL}}
    guards.save_json(cache / f"run {day}.json", record)
    files["README.md"] = render_readme(record, scores, kept_rels, facts, questions, cache / f"run {day}.json")
    hits = {name: check_written(body) for name, (_, body) in files.items()}
    hits = {k: v for k, v in hits.items() if v}
    if hits:
        say(f"reflect: writing-rule hits in the output: {hits}")
    if dry_run or out is None:
        for name, (meta, body) in files.items():
            print(f"===== {name}\n{body}")
        say(f"reflect: dry run, nothing written; ${budget.spent():.4f} of ${budget.cap:.2f} spent")
        return 1 if error else 0
    filt.check_stops(state, vault, cfg.vault)
    written = write_new(Path(out), files)
    say(f"reflect: wrote {len(written)} notes to {out}; ${budget.spent():.4f} of ${budget.cap:.2f} spent on "
        f"profile work on {day}")
    return 1 if error else 0


def reword_claims(claims: list[Claim], month: str, cache: Path, budget, cwd: Path, runner, stop,
                  log=print) -> tuple[list[Claim], list[Claim], float]:
    """propose.reword's pass for the writing bans, cached like the other calls so a rerun gets the
    same words: one call rewrites the claims whose text breaks a ban, and any still failing is held.
    Returns (kept, held, cost)."""
    bad = [c for c in claims if propose.prose_hits({"text": c.text})]
    if not bad:
        return claims, [], 0.0
    lines = ["Rewrite these so they follow the writing rules. Keep the meaning and the facts. "
             "Return every item with its id.\n\n", propose.WRITING_RULES, "\n"]
    for i, c in enumerate(bad):
        lines.append(json.dumps({"id": str(i), "reason": "", "description": "", "text": c.text,
                                 "broken rules": propose.prose_hits({"text": c.text})}, ensure_ascii=False) + "\n")
    cost = 0.0
    try:
        out, cost, _ = cached_call(cache, f"{month} reword", "".join(lines), propose.REWORD_SCHEMA, budget, cwd,
                                   runner, stop, propose.MODEL, 0.25, log)
        fixed = {str(it.get("id")): it for it in out.get("items") or []}
    except propose.ProposeError:
        fixed = {}
    held = []
    for i, c in enumerate(bad):
        new = propose.tidy((fixed.get(str(i)) or {}).get("text", "")).replace("\n", " ")
        if new and not propose.prose_hits({"text": new}):
            c.text = new
        else:
            c.dropped = "writing rules: " + ", ".join(propose.prose_hits({"text": new or c.text}))
            held.append(c)
    return [c for c in claims if c not in held], held, cost


def _n(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def render_readme(record: dict, scores: dict, kept_rels: set[str], facts: list[dict], questions: list[dict],
                  record_path: Path) -> tuple[dict, str]:
    s, day = record["stats"], record["day"]
    scored = sum(1 for r in scores if r in kept_rels)
    about = sum(1 for r, e in scores.items() if r in kept_rels and filt.about_user(e))
    checker = {k[len("checker "):]: v for k, v in s.items() if k.startswith("checker ")}
    kept_of = lambda *kinds: sum(1 for c in record["claims"] if c["status"] in KEEP and c["kind"] in kinds)
    filter_cost = sum(float(r.get("cost") or 0) for r in record.get("filter_runs") or [])
    meta = {"title": f"Profile spike {day}", "date": day, "type": "report", "source": "agent",
            "visibility": "private", "tags": ["profile", "spike"]}
    body = [f"# Profile spike, {day}\n\n",
            "What the profile spike did, run on a copy "
            "of the vault. Nothing here changed any other note, and nothing went into "
            f"`{PROFILE_REL}/`.\n\n",
            "## Files\n\n",
            "".join(f"- [[spike diary {m}]]: the diary entry for {m}\n" for m in record["months"]),
            "- [[spike facts digest]]: one box per fact that passed both checks\n",
            "- [[spike gist]]: what's new and the questions\n",
            "- [[spike claims to label]]: every claim that passed the quote check, with the fact-checker's verdict, "
            "ready for your labels\n\n",
            "## Counts\n\n",
            f"- Chats scored by the filter: {scored} (of {len(kept_rels)} kept chats in the copy); {about} of "
            f"them scored {filt.PROFILE_MIN} or more and were labelled yours.\n",
            f"- Months picked: {', '.join(record['months'])}. The rule, set before any scores were seen: the 3 "
            f"months in a row with the most chats about you, and a tie goes to the more recent run.\n",
            "".join(f"    - {m}: {v.get('about', 0)} chats about you, {v.get('chats', 0)} read, "
                    f"{v.get('kept', 0)} claims kept\n" for m, v in record["per_month"].items()),
            f"- Claims extracted: {s.get('extracted', 0)} (quotes checked: {s.get('quotes', 0)}, quotes not "
            f"found: {s.get('quotes failed', 0)}).\n",
            f"- Dropped by the quote check: {s.get('dropped by quote check', 0)} claims.\n",
            f"- Dropped by the other script rules (2-chat floor, motive or feeling words, rejected list, "
            f"excluded topics): {s.get('dropped by rules', 0)}; held by the writing rules: "
            f"{s.get('held by writing rules', 0)}; repeats merged: {s.get('merged repeats', 0)}.\n",
            "- Fact-checker verdicts: " + (", ".join(f"{k} {v}" for k, v in sorted(checker.items())) or "none")
            + ".\n",
            f"- Kept: {_n(len(facts), 'fact')} in the digest, {_n(kept_of('did'), 'observation')}, "
            f"{_n(kept_of('interpretation'), 'interpretation')}, {_n(kept_of('unknown', 'contradiction'), 'open item')}.\n",
            f"- Questions in the gist: {len(questions)} (dropped by the question checks: "
            f"{s.get('questions dropped', 0)}).\n\n",
            "## Cost\n\n",
            "The calls behind these notes, by stage (a cached answer counts what it cost when it was paid for):\n\n",
            "".join(f"- {k}: ${v:.4f}\n" for k, v in record["costs"].items()),
            f"- filter (every run on this copy): ${filter_cost:.4f}\n",
            f"- On the spend ledger line `{filt.spend_key(day)}`, which also holds trial runs while the spike "
            f"was being tuned: ${record['spent_on_ledger']:.4f} of a ${record['cap']:.2f} cap.\n",
            f"- Models: extract {record['models']['extract']}, fact-check and gist {record['models']['check']}, "
            f"filter {record['models']['filter']}.\n\n",
            "## What isn't verified\n\n",
            "- The fact-checker is a model. It hasn't been run against a gold set of claims you labelled, so "
            "its false alarm rate is unknown. Your labels in [[spike claims to label]] are that gold set.\n",
            "- The quote check proves the words are in your turn of that chat. It doesn't prove the claim "
            "reads them right; that's the fact-checker's job and yours.\n",
            "- The filter's scores are from a cheap model and weren't checked by hand. A chat it scored low "
            "never reached the extractor.\n",
            "- Other people's chats that weren't sorted yet are kept out only by the filter's `whose` "
            "label and the People notes.\n",
            "- No web checks were run (dates, companies, schools) in this spike.\n",
            f"- The exclusion list (`BONSAI_STATE/{filt.EXCLUDE_FILE}`) "
            + ("was empty for this run, so health and other personal facts you stated plainly in a chat "
               "were kept. Add topics there to keep them away from the model next time.\n"
               if not record.get("excluded_topics") else
               f"held {record['excluded_topics']} topics.\n"),
            ("- The comparison with Codex's own reflections wasn't run: there are no Codex chat notes in "
             "the copy yet.\n"
             if not record.get("codex_chats") else ""),
            "- A chat's turns go to the month they're dated in, so a chat picked up again months later gives "
            "only its in-month turns to each month.\n",
            "- Quotes are your words exactly as captured, so they skip the writing rules.\n",
            f"- The run record with every claim, verdict and drop reason is in BONSAI_STATE, at "
            f"`{record_path}`.\n"]
    if record.get("error"):
        body.append(f"\n## The run stopped early\n\n{record['error']}\n")
    return meta, "".join(body)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="python -m dream.reflect", description=__doc__.split("\n\n")[0])
    ap.add_argument("--vault", help="read this vault instead of BONSAI_VAULT (a copy)")
    ap.add_argument("--from", dest="first", help="first month, YYYY-MM")
    ap.add_argument("--to", dest="last", help="last month, YYYY-MM")
    ap.add_argument("--pick", type=int, help="pick this many months in a row from the filter's scores")
    ap.add_argument("--out", help="the folder the notes go in (new files only)")
    ap.add_argument("--cap-usd", type=float, help="dollar cap for profile work on this day (required)")
    ap.add_argument("--max-chats", type=int, default=MAX_CHATS, help="chats read per month")
    ap.add_argument("--dry-run", action="store_true", help="call the models but print the notes, write nothing")
    args = ap.parse_args(argv)
    months = None
    if args.first or args.last:
        if not (args.first and args.last) or not all(re.fullmatch(r"\d{4}-\d{2}", m) for m in (args.first, args.last)):
            ap.error("--from and --to go together, as YYYY-MM")
        months = filt.months_between(args.first, args.last)
    elif not args.pick:
        ap.error("pass --from and --to, or --pick N")
    try:
        cfg = config.load()
    except config.ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    return run(cfg, Path(args.vault) if args.vault else None, months, args.pick,
               Path(args.out) if args.out else None, args.cap_usd, max_chats=args.max_chats, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
