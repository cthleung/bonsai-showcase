"""Ask Claude what the new notes mean for the user's memory, and check what comes back.

One `claude -p` call per batch of sources (see collect.pack), with:

    claude -p --model claude-sonnet-5-5      pinned full model ID, no fallback model
          --tools ""                         no built-in tools
          --strict-mcp-config                no MCP servers (and so no MCP tools)
          --setting-sources ""               no user or project settings (hooks, plugins)
          --disable-slash-commands           no skills
          --no-session-persistence           the vault text isn't saved as a resumable session
          --output-format json --json-schema <SCHEMA>
          --max-budget-usd <what's left of the night's cap, rounded down to the cent>
          --system-prompt <one line>

A call whose estimated cost won't fit what's left isn't started (see call()).

The instructions and the notes go on stdin, never on the command line. The child runs in a folder
under BONSAI_STATE, so no CLAUDE.md or AGENTS.md from the repo or the vault is picked up.

The model returns proposals keyed to sources S1, S2...; validate() turns them into checked
proposals: real sources, safe paths under the PARA homes, no add over an existing note, and
`visibility` set here from the sources, never by the model. An empty answer is fine.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable

from common.notes import is_reserved, is_safe_part
from dream import digest as digest_mod
from dream import quotes as quotes_mod
from dream.collect import Batch, Piece, visibility_of

MODEL = "claude-sonnet-5-5"
MACHINE_NAMES = {"win": "PC", "mac": "Mac"}
DASHES = chr(0x2014) + chr(0x2013)   # em and en dash
CALL_TIMEOUT_S = 20 * 60
# What a call costs before it runs, from its prompt length. A trial run paid about $2 per million
# prompt characters on Sonnet; this pads that, so a call that won't fit what's left of the cap isn't
# started. The CLI's --max-budget-usd only stops a call between turns, and one turn can pass it.
EST_BASE_USD = 0.01
EST_USD_PER_CHAR = 2.5e-6
REWORD_RESERVE_USD = 0.05     # kept back from the batch calls so the wording pass can still run
ACTIONS =("add", "merge", "move", "flag", "profile")
TYPES = ("user", "feedback", "project", "reference")
SYSTEM_PROMPT = "You are Bonsai's night job. Read the input and answer only in the given JSON schema."

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["proposals"],
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "target", "reason", "sources"],
                "properties": {
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "target": {"type": "string"},
                    "reason": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "type": {"type": "string", "enum": list(TYPES)},
                    "date": {"type": "string"},
                    "text": {"type": "string"},
                    "move_to": {"type": "string"},
                },
            },
        }
    },
}

REWORD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "reason", "description", "text"],
                "properties": {
                    "id": {"type": "string"},
                    "reason": {"type": "string"},
                    "description": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
        }
    },
}

WRITING_RULES = """\
Writing rules for every word you write (reason, description, text):
- Plain, casual, short words.
- No em dashes or en dashes. Use a spaced hyphen ( - ) instead.
- Never use these words, not even literally: half, weight, gravity, structural, load-bearing,
  scaffolding, framework, architecture.
- No hedging. Don't write maybe, perhaps, probably, possibly, might, likely, seems, apparently.
  State a fact plainly or leave it out.
- Absolute dates like 2026-11-15. Never today, tonight, yesterday, tomorrow, last week, next week.
- Use they/them for anyone whose pronouns aren't stated, the user included.
- Never copy a key, password, token or cookie.
- The one exception to these rules: the user's own words, copied exactly inside double quotes
  ("like this"), from a "You" turn of a chat or a note the user wrote. When a name or phrase the user
  used breaks a rule (a team name or a job title, say), quote it exactly or leave it out. Quotes of
  the AI's answers or of an agent's notes get no exception.
"""

INSTRUCTIONS = """\
You read new notes and AI chats from Bonsai, the user's personal notes vault, and propose changes to
the user's memory notes. The user reads your proposals in a morning digest and ticks the ones they
want. Nothing you propose happens unless the user ticks it.

A memory note holds one durable fact that will help the user or an agent in a later session: a
decision the user made, a rule they set, a preference, a plan, a fact about a project, or a fact
about the user themself. Keep only what is still useful in a month.

Chats: the "You" turns are the user's words. The answers are the AI's text, not the user's views.
General knowledge from an answer is not a memory. A fact counts when the user stated it, decided it,
or clearly adopted it. A one-off question shows curiosity, not a fact. Proposing nothing is fine
and common: return an empty list when nothing is worth keeping.

Actions:
- add: a new memory note. target is "<home>/<kebab-case-name>.md", with <home> taken from the Homes
  list (or a new "1 Projects/<Name>/memory/" or "2 Areas/<Name>/" folder when none fits). Fill
  description (one line, used to decide relevance), type (user, feedback, project or reference),
  date (YYYY-MM-DD, when the fact was made or said) and text (the fact; for feedback and project
  notes, follow it with a "**Why:** ..." line and a "**How to apply:** ..." line).
- profile: a fact about the user themself (who they are, how they work, what they like). target is
  "2 Areas/Profile/<kebab-case-name>.md". Fill description, date and text like add.
- merge: an existing note already covers this, and the source adds to or updates it. target is that
  note's path from Existing notes. text is the lines to add to it.
- flag: a source contradicts an existing note. target is that note's path. text says what
  conflicts, quoting both sides briefly. The user settles it by hand.
- move: a note in this computer's inbox (under 0 Inbox/{inbox}/) or an existing note outside the
  inbox is in the wrong place. target is its path; move_to is the folder it belongs in (under
  1 Projects/, 2 Areas/, 3 Resources/ or 4 Archive/). A note in the other computer's inbox
  ({other_inbox}) can't be moved from here: propose the move anyway and it shows as a flag.

Don't add a note that duplicates an existing one: use merge if the source adds something, or skip
it. One fact per proposal. reason is one short line saying why. sources lists the keys (S1, S2...)
of the sources the proposal comes from. Use only paths that appear below, or new paths for add and
profile.

{rules}
Today is {today}.
"""


class ProposeError(RuntimeError):
    pass


Runner = Callable[[list[str], str, Path], dict]


# --- the claude call ------------------------------------------------------------------------

def claude_exe() -> str:
    exe = os.environ.get("CLAUDE_CLI") or shutil.which("claude")
    if not exe:
        raise NotStarted("the claude CLI isn't on PATH (set CLAUDE_CLI to its full path)")
    return exe


def floor_cents(usd: float) -> float:
    """Round down to the cent, so the CLI never gets more than what's left ($0.0774 is 0.07)."""
    return math.floor(round(float(usd) * 100, 6)) / 100


def estimate_usd(prompt: str) -> float:
    """What a call with this prompt costs, padded (see EST_USD_PER_CHAR)."""
    return EST_BASE_USD + len(prompt) * EST_USD_PER_CHAR


class CapReached(ProposeError):
    """The night's cap has no room for this call. Not a failure: what's left waits for another night."""


class NotStarted(ProposeError):
    """The call never ran (no claude CLI), so nothing is charged for it."""


def claude_args(schema: dict, budget_usd: float, model: str = MODEL) -> list[str]:
    """The flags after the executable. Nothing from the vault goes here."""
    return [
        "-p",
        "--model", model,
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--output-format", "json",
        "--json-schema", json.dumps(schema, separators=(",", ":")),
        "--max-budget-usd", f"{floor_cents(budget_usd):.2f}",
        "--system-prompt", SYSTEM_PROMPT,
    ]


def subprocess_runner(args: list[str], stdin_text: str, cwd: Path) -> dict:
    """Run claude with the prompt on stdin. Returns the parsed --output-format json result."""
    cwd.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run([claude_exe(), *args], input=stdin_text, capture_output=True,
                              encoding="utf-8", errors="replace", cwd=str(cwd), timeout=CALL_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise ProposeError(f"claude took longer than {CALL_TIMEOUT_S} s") from e
    try:
        return json.loads(proc.stdout)
    except ValueError:
        tail = (proc.stderr or proc.stdout or "")[-500:]
        raise ProposeError(f"claude exited {proc.returncode} without a JSON result: {tail}")


def call(prompt: str, schema: dict, budget, cwd: Path, runner: Runner | None = None,
         stop: Callable[[], None] | None = None, reserve: float = 0.0, model: str = MODEL,
         max_call_usd: float | None = None) -> tuple[dict, float, list[str]]:
    """One structured call. Returns (structured_output, cost in USD, models used). The cost goes on
    the night's ledger whether the call worked or not. `stop` runs first (guards.check_stop): a STOP
    file made mid-night raises guards.Refused before the next call starts. `model` is the pinned model
    ID for this call, and `max_call_usd` caps this one call below what's left of the cap.

    Holding the cap: no call starts with less than the floor left (Budget.MIN_CALL_USD),
    or when its estimated cost plus `reserve` (kept for a later call, the reword) is more than
    what's left; both raise CapReached. The CLI gets what's left, or `max_call_usd` when that's
    smaller, as --max-budget-usd. When the call times out or gives back no usable result, its cost
    is unknown, so what it was allowed to spend goes on the ledger (never more than `max_call_usd`)."""
    if stop:
        stop()
    left = budget.remaining()
    if not budget.can_call():
        raise CapReached(f"the night's cap is reached (${budget.spent():.2f} of ${budget.cap:.2f})")
    est = estimate_usd(prompt)
    if est + reserve > left:
        raise CapReached(f"the next call needs about ${est:.2f}"
                         f"{f' plus ${reserve:.2f} kept back' if reserve else ''} and ${left:.2f} is left "
                         f"of the ${budget.cap:.2f} cap")
    allowed = floor_cents(left if max_call_usd is None else min(left, float(max_call_usd)))
    try:
        result = (runner or subprocess_runner)(claude_args(schema, allowed, model), prompt, cwd)
    except NotStarted:
        raise
    except ProposeError as e:
        budget.add(allowed)
        raise ProposeError(f"{e} (charged ${allowed:.2f}, what the call was allowed)") from e
    if not isinstance(result, dict) or result.get("total_cost_usd") is None:
        budget.add(allowed)
        raise ProposeError(f"claude gave back no cost (charged ${allowed:.2f}, what the call was allowed)")
    cost = float(result.get("total_cost_usd") or 0.0)
    budget.add(cost)
    models = sorted((result.get("modelUsage") or {}).keys())
    if result.get("subtype") != "success" or result.get("is_error"):
        raise ProposeError(f"claude stopped: {result.get('subtype')} (cost ${cost:.4f})")
    out = result.get("structured_output")
    if not isinstance(out, dict):
        raise ProposeError(f"claude gave no structured output (cost ${cost:.4f})")
    return out, cost, models


# --- the prompt -----------------------------------------------------------------------------

def build_prompt(batch: Batch, existing: list[dict], home_list: list[str], today: str, inbox: str = "win") -> str:
    other = "0 Inbox/" + ("mac" if inbox == "win" else "win") + "/"
    parts = [INSTRUCTIONS.format(rules=WRITING_RULES, today=today, inbox=inbox, other_inbox=other),"\n# Homes\n"]
    parts += [f"- {h}\n" for h in home_list]
    parts.append("\n# Existing notes\n")
    if existing:
        for n in existing:
            parts.append(f"- {n['rel']} | {n['type']} | {n['description']} | {n['snippet']}\n")
    else:
        parts.append("(none yet)\n")
    parts.append("\n# Sources\n")
    for p in batch.pieces:
        turns = f", turns {p.turns[0]} to {p.turns[1]} of {p.turns_total}" if p.turns else ""
        parts.append(f"\n## {p.key}: {p.kind} note {p.rel} (dated {p.date}{turns})\n\n{p.text}\n")
    return "".join(parts)


# --- checking what came back ----------------------------------------------------------------

_DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NAME_RX = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
_MEMORY_HOME_RX = re.compile(r"^(1 Projects/[^/]+/memory|2 Areas/[^/]+(/[^/]+)*)/[^/]+\.md$")
_MOVE_FROM = ("0 Inbox/win/", "0 Inbox/mac/", "1 Projects/", "2 Areas/", "3 Resources/", "4 Archive/")
_MOVE_TO = ("1 Projects/", "2 Areas/", "3 Resources/", "4 Archive/")
_NO_MOVE = ("3 Resources/chats/", "4 Archive/chats-excluded/")   # capture and the excluded archive own these


def clean_path(value: str, folder: bool = False) -> str | None:
    """A vault-relative posix path, or None if it's unsafe: absolute, a drive, `..`, a hidden or
    empty part, or any part that fails common.notes.is_safe_part (control characters such as a
    newline, `|`, `[ ]`, a trailing dot or space, a Windows reserved name). Rejecting control
    characters here is what stops a target from smuggling a second, pre-ticked line into the digest."""
    v = str(value or "").strip(" ").replace("\\", "/").strip("/")
    if not v or ":" in v or v.startswith("~"):
        return None
    parts = v.split("/")
    if digest_mod.CONTROL_RX.search(v):   # also the line breaks is_safe_part doesn't know (NEL, U+2028/9)
        return None
    if any(p.startswith(".") or not is_safe_part(p) for p in parts):
        return None
    return "/".join(parts) + ("/" if folder else "")


def tidy(text: str) -> str:
    """Mechanical fixes the writing rules allow: dashes become a spaced hyphen, `|` can't break the
    digest line."""
    text = re.sub(r"\s*[" + DASHES + r"]\s*", " - ", str(text or ""))
    return text.strip()


def _one_line(text: str) -> str:
    return " ".join(tidy(text).replace("|", "/").split())


def _visibility(pieces: list[Piece]) -> str:
    vis = {p.visibility for p in pieces}
    return vis.pop() if len(vis) == 1 else "private"


def validate(raw: dict, batch: Batch, vault: Path, existing: list[dict], today: str,
             inbox: str = "win", words: quotes_mod.OwnWords | None = None) -> tuple[list[dict], list[str]]:
    """(proposals, rejections). Each proposal is a dict ready for the sidecar and the digest.
    `inbox` is this computer's inbox folder name (cfg.inbox_name): one writer per inbox folder
    (one writer per inbox folder), so a move out of another computer's inbox becomes a flag. With `words`, a
    proposal from an agent's note also cites the chat its quote came from (quotes.add_provenance)."""
    vault = Path(vault)
    by_key = {p.key: p for p in batch.pieces}
    existing_vis = {n["rel"]: n["visibility"] for n in existing}
    out: list[dict] = []
    rejected: list[str] = []
    for item in raw.get("proposals") or []:
        action = item.get("action")
        why = None
        keys = [k for k in (item.get("sources") or []) if k in by_key]
        pieces = [by_key[k] for k in dict.fromkeys(keys)]
        target = clean_path(item.get("target"))
        text = tidy(item.get("text", ""))
        prop = {
            "action": action,
            "target": target,
            "reason": _one_line(item.get("reason", "")),
            "sources": list(dict.fromkeys(p.rel for p in pieces)),
            "links": list(dict.fromkeys(p.link for p in pieces)),
            "visibility": _visibility(pieces),
        }
        if words is not None and pieces:   # before the checks below, which use the visibility
            probe = dict(prop, description=item.get("description", ""), text=text)
            traced = quotes_mod.add_provenance(probe, words, visibility_of)
            if traced:
                prop["visibility"] = probe["visibility"]   # sources and links are the same lists
                prop["traced"] = traced
        if action not in ACTIONS:
            why = f"unknown action {action!r}"
        elif not pieces:
            why = "no known source"
        elif not target or not target.endswith(".md"):
            why = f"bad target {item.get('target')!r}"
        elif not prop["reason"]:
            why = "no reason"
        elif action in ("add", "profile"):
            stem = PurePosixPath(target).stem
            if action == "profile":
                target = f"2 Areas/Profile/{PurePosixPath(target).name}"
            date = str(item.get("date") or "")
            prop.update(target=target, name=stem, description=_one_line(item.get("description", "")),
                        type="user" if action == "profile" else item.get("type", ""),
                        date=date if _DATE_RX.fullmatch(date) else max(p.date for p in pieces)[:10] or today,
                        text=text)
            if not _NAME_RX.fullmatch(stem) or is_reserved(stem):
                why = f"note name isn't kebab-case: {stem!r}"
            elif not _MEMORY_HOME_RX.fullmatch(target):
                why = f"not a memory home: {target}"
            elif (vault / target).exists():
                why = f"already exists: {target}"
            elif prop["type"] not in TYPES:
                why = f"bad type {prop['type']!r}"
            elif not prop["description"] or not text:
                why = "no description or text"
        elif action in ("merge", "flag"):
            prop["text"] = text
            if not (vault / target).is_file():
                why = f"no such note: {target}"
            elif not text:
                why = "no text"
            elif action == "merge":
                tvis = existing_vis.get(target, "private")
                if tvis != "private" and tvis != prop["visibility"]:
                    # private facts never go into a less private note: the user decides by hand
                    prop["action"] = "flag"
                    prop["reason"] = f"{prop['reason']} (merge held: the note is {tvis}, the source is {prop['visibility']})"
        elif action == "move":
            dest = clean_path(item.get("move_to"), folder=True)
            prop["move_to"] = dest
            if not (vault / target).is_file():
                why = f"no such note: {target}"
            elif not target.startswith(_MOVE_FROM) or target.startswith(_NO_MOVE):
                why = f"can't move {target}"
            elif not dest or not dest.startswith(_MOVE_TO) or dest.startswith(_NO_MOVE):
                why = f"bad move_to {item.get('move_to')!r}"
            elif (vault / dest / PurePosixPath(target).name).exists():
                why = f"already exists: {dest}{PurePosixPath(target).name}"
            elif dest == str(PurePosixPath(target).parent) + "/":
                why = "already there"
            elif target.startswith("0 Inbox/") and not target.startswith(f"0 Inbox/{inbox}/"):
                owner = target.split("/")[1]
                prop["action"] = "flag"
                prop["reason"] = f"in the {owner} inbox; move it from the {MACHINE_NAMES.get(owner, owner)}"
                prop["text"] = (f"This note belongs in {dest}. It sits in the {owner} inbox, and only "
                                f"the {MACHINE_NAMES.get(owner, owner)} writes there, so move it from there or by hand.")
                del prop["move_to"]
        if why:
            rejected.append(f"{action} {item.get('target')!r}: {why}")
        else:
            out.append(prop)
    return out, rejected


def dedupe(proposals: list[dict], taken: set[tuple[str, str]] | None = None) -> list[dict]:
    """Drop repeats of (action, target), including ones still waiting in an earlier digest."""
    seen = set(taken or ())
    out = []
    for p in proposals:
        k = (p["action"], p["target"])
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


# --- the wording pass -----------------------------------------------------------------------

def prose_hits(p: dict, words: quotes_mod.OwnWords | None = None) -> list[str]:
    """The writing rules the proposal's own words break. Quotes of the user's own words from its
    sources don't count (dream/quotes.py). With `words`, the quotes are checked again first."""
    if words is not None:
        quotes_mod.verify(p, words)
    text = "\n".join(str(p.get(k, "")) for k in ("reason", "description", "text"))
    return digest_mod.check_text(quotes_mod.strip_quoted(text, p.get("quoted")))


REWORD_INTRO = ("Rewrite these so they follow the writing rules. Keep the meaning and the facts. "
                "When a broken rule sits in a name or phrase the user used (a team name, say), put those "
                "words in double quotes exactly as written instead of changing them. Return every item "
                "with its id.\n\n")


def reword(proposals: list[dict], budget, cwd: Path, runner: Runner | None = None,
           stop: Callable[[], None] | None = None,
           words: quotes_mod.OwnWords | None = None) -> tuple[list[dict], list[dict], float]:
    """One call that rewrites proposals whose words break the writing rules. Returns
    (clean proposals, still failing ones, cost). Paths, sources and actions never change here."""
    bad = [p for p in proposals if prose_hits(p, words)]
    if not bad:
        return proposals, [], 0.0
    lines = [REWORD_INTRO, WRITING_RULES, "\n"]
    for i, p in enumerate(bad):
        item = {"id": str(i), "reason": p.get("reason", ""), "description": p.get("description", ""),
                "text": p.get("text", ""), "broken rules": prose_hits(p)}
        lines.append(json.dumps(item, ensure_ascii=False) + "\n")
    cost = 0.0
    try:
        out, cost, _ = call("".join(lines), REWORD_SCHEMA, budget, cwd, runner, stop)
        fixed = {str(it.get("id")): it for it in out.get("items") or []}
    except ProposeError:
        fixed = {}
    still = []
    for i, p in enumerate(bad):
        it = fixed.get(str(i))
        if it:
            p["reason"] = _one_line(it.get("reason", "")) or p["reason"]
            if "description" in p:
                p["description"] = _one_line(it.get("description", "")) or p["description"]
            if "text" in p:
                p["text"] = tidy(it.get("text", "")) or p["text"]
        hits = prose_hits(p, words)
        if hits:
            p["held"] = hits
            still.append(p)
    good = [p for p in proposals if not p.get("held")]
    return good, still, cost


def propose(batches: list[Batch], vault: Path, existing: list[dict], home_list: list[str], today: str,
            budget, cwd: Path, runner: Runner | None = None, taken=None, log=print, inbox: str = "win",
            stop: Callable[[], None] | None = None) -> dict:
    """Run every batch until one fails or the cap runs out. `stop` is checked before every call and
    raises guards.Refused to end the night (nothing is written then). A batch the cap has no room
    for isn't an error: it and the rest wait for another night ("capped"). Returns
    {"proposals", "held", "read": [pieces whose call worked], "rejected", "cost", "models", "error",
    "capped"}. "cost" is what went on the ledger tonight, failed calls included."""
    result = {"proposals": [], "held": [], "read": [], "rejected": [], "cost": 0.0, "models": set(),
              "error": None, "capped": None}
    words = quotes_mod.OwnWords(vault)
    spent_before = budget.spent()
    for i, batch in enumerate(batches, 1):
        prompt = build_prompt(batch, existing, home_list, today, inbox)
        try:
            raw, cost, models = call(prompt, SCHEMA, budget, cwd, runner, stop, reserve=REWORD_RESERVE_USD)
        except CapReached as e:
            result["capped"] = f"call {i} of {len(batches)} not started: {e}"
            log(f"propose: {result['capped']}")
            break
        except ProposeError as e:
            result["error"] = f"call {i} of {len(batches)}: {e}"
            log(f"propose: {result['error']}")
            break
        result["models"].update(models)
        props, rejected = validate(raw, batch, vault, existing, today, inbox, words)
        result["proposals"] += props
        result["rejected"] += rejected
        result["read"] += batch.pieces
        log(f"propose: call {i} of {len(batches)} read {len(batch.pieces)} pieces "
            f"({batch.chars} chars), {len(props)} proposals, {len(rejected)} rejected, ${cost:.4f}")
    result["proposals"] = dedupe(result["proposals"], taken)
    traced = [p for p in result["proposals"] if p.get("traced")]
    log(f"propose: {len(traced)} proposals from agent notes also cite the chat their quote came from")
    good, held, _ = reword(result["proposals"], budget, cwd, runner, stop, words)
    log(f"propose: {sum(1 for p in good if p.get('quoted'))} proposals quote the user in words the check leaves out")
    result["proposals"], result["held"] = good, held
    result["cost"] = budget.spent() - spent_before
    result["models"] = sorted(result["models"])
    return result
