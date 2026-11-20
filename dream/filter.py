"""Score each kept chat for "does it say something about the user", the first step of the profile.

    python -m dream.filter --vault <copy> --cap-usd 1.50     score every chat not scored yet
    python -m dream.filter --vault <copy> --limit 15         a small smoke run
    python -m dream.filter --vault <copy> --table            print the per-month table, no calls

What gets scored: every chat note under `3 Resources/chats/` (so nothing under `4 Archive/`), minus
notes with `excluded:` set, drafts and capture notices (collect.skip_note), and chats that match a
topic in `BONSAI_STATE/profile-exclude.txt` (one topic per line, `#` comments). Excluded topics
never reach the model.

What the model sees: the title and the user's own turns (the "You" turns, trimmed, quoted answer
lines left out), plus the People and Profile notes (vault_context) so it can tell a chat the user
ran for someone else from one of their own. The answers are never shown. It returns, per chat:

    score   0 to 10, how much the chat says about the user (who they are, what they want, how they
            work, their taste, their history). Someone else's chat is 0.
    whose   user | other | unclear
    reason  one short line

Scores go to a sidecar, `<vault state>/profile/scores.json`, keyed by the note path, with the
note's hash, so a chat is scored again only when it changes. Nothing is written into the notes.
The sidecar is saved after every call, so a failed run never pays twice.

The call is propose.call (pinned model, no tools, no MCP, no settings, no session, JSON schema, a
dollar cap), on the cheap model FILTER_MODEL. Its cost goes on the spend ledger line
"<day> profile", shared with dream.reflect, so the profile work has one cap.

Exit codes: 0 done, 1 failed, 2 refused by a guard.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from common import config
from common.notes import CHATS_DIR, parse, read_note
from common.scrub import scrub_text
from common.turns import parse_turns, split_turn
from dream import collect, guards, propose

FILTER_MODEL = "claude-haiku-4-5"
BATCH_CHATS = 20
MAX_CALL_USD = 0.25
SPEND_KEY = "{day} profile"
PROFILE_DIR = "profile"
SCORES = "scores.json"
FILTER_RUNS = "filter-runs.json"   # one record per run: when, how many scored, cost
EXCLUDE_FILE = "profile-exclude.txt"
WHOSE = ("user", "other", "unclear")
MAX_REASON_CHARS = 200
# Chosen before any scores were seen: a chat is "about the user" when it's the user's own
# (whose = user) and scores PROFILE_MIN or more. dream.reflect reads only those.
PROFILE_MIN = 5

INSTRUCTIONS = """\
You score chats from the user's AI chat accounts. For each chat, say how much the user's own side
of it says about the user: who they are, what they want, how they work, their taste, their history,
their plans and decisions. Use they/them for the user.

score, 0 to 10:
- 0: nothing about the user, or it's someone else's chat (see the People notes below).
- 1 to 3: a lookup or task with only a small hint about the user (a tool they use, a topic).
- 4 to 6: shows what the user works on, makes or likes, in some detail.
- 7 to 10: the user says things about themself: their goals, choices, working rules, likes and
  dislikes, background or history.
whose: user, other (a person from the People notes, or clearly not the user), or unclear.
reason: one short line naming what decided it (a topic, a phrase).

Only the user's side of each chat is shown (the "You" turns, trimmed); the answers are left out.
Return one item for each chat, with its key (C1, C2...).
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["chats"],
    "properties": {
        "chats": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "score", "whose", "reason"],
                "properties": {
                    "key": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 10},
                    "whose": {"type": "string", "enum": list(WHOSE)},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}


# --- shared with dream.reflect -----------------------------------------------------------------

def spend_key(day: str) -> str:
    return SPEND_KEY.format(day=day)


def profile_state(state: Path, vault: Path, real_vault: Path) -> Path:
    """`<vault state>/profile/`: dream/profile for the real vault, dream/copy-<id>/profile for a copy."""
    return guards.vault_state_dir(state, vault, real_vault) / PROFILE_DIR


def profile_lock(state: Path, now: _dt.datetime | None = None) -> guards.Lock:
    """A lock of its own (BONSAI_STATE/profile/lock), so profile runs never block the night job,
    and two profile runs never overlap."""
    lock = guards.Lock(state, now)
    lock.path = Path(state) / PROFILE_DIR / "lock"
    return lock


def check_stops(state: Path, vault: Path, real_vault: Path) -> None:
    """The dream STOP files, for the vault being read and for the real vault too."""
    guards.check_stop(state, vault)
    guards.check_stop(state, real_vault)


def exclude_topics(state: Path) -> list[str]:
    path = Path(state) / EXCLUDE_FILE
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = " ".join(line.split("#", 1)[0].split())
        if line:
            out.append(line.lower())
    return out


def excluded_topic(text: str, topics: list[str]) -> str | None:
    """The first excluded topic found in `text` (whole words, any case, whitespace collapsed on
    both sides so a topic wrapped over two lines still counts), or None."""
    if not topics:
        return None
    low = " ".join(text.split()).lower()
    for t in topics:
        t = " ".join(t.split()).lower()
        if t and re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", low):
            return t
    return None


_HEADING_RX = re.compile(r"^(#{1,6})\s")
_BULLET_RX = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s")
_FENCE_RX = re.compile(r"^\s*(```|~~~)")


def _plain(line: str) -> bool:
    """A line that carries on the block above it (not blank, not a heading, bullet or fence)."""
    return bool(line.strip()) and not (_HEADING_RX.match(line) or _BULLET_RX.match(line)
                                       or _FENCE_RX.match(line))


def _blocks(text: str) -> list[tuple[str, int, list[str]]]:
    """Split markdown into (kind, level, lines): "heading" (level 1 to 6), "bullet" (its indent;
    the bullet line plus the plain lines after it), "para" (its indent; a run of plain lines, or a
    whole fenced code block) and "blank"."""
    out: list[tuple[str, int, list[str]]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        indent = len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())
        fence = _FENCE_RX.match(line)
        if fence:
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(fence.group(1)):
                j += 1
            out.append(("para", indent, lines[i:j + 1]))
            i = j + 1
            continue
        if not line.strip():
            out.append(("blank", 0, [line]))
            i += 1
            continue
        heading = _HEADING_RX.match(line)
        if heading:
            out.append(("heading", len(heading.group(1)), [line]))
            i += 1
            continue
        j = i + 1
        while j < len(lines) and _plain(lines[j]):
            j += 1
        out.append(("bullet" if _BULLET_RX.match(line) else "para", indent, lines[i:j]))
        i = j
    return out


def redact_topics(text: str, topics: list[str]) -> str:
    """`text` (markdown) with the excluded topics taken out, for notes that go into a prompt: a
    bullet or paragraph that names a topic is dropped (a bullet with its indented sub-items), and a
    heading that names one is dropped with everything under it, up to the next heading of the same
    or a higher level. Everything else stays as it was. Call it before any length cut, so a cut
    never decides what's left."""
    if not excluded_topic(text, topics):
        return text
    blocks = _blocks(text)
    keep: list[str] = []
    i = 0
    while i < len(blocks):
        kind, level, lines = blocks[i]
        i += 1
        if kind == "blank" or not excluded_topic("\n".join(lines), topics):
            keep += lines
            continue
        if kind == "heading":
            while i < len(blocks) and not (blocks[i][0] == "heading" and blocks[i][1] <= level):
                i += 1
        elif kind == "bullet":
            while True:   # its sub-items: blocks indented deeper, with any blank lines between
                j = i
                while j < len(blocks) and blocks[j][0] == "blank":
                    j += 1
                if j < len(blocks) and blocks[j][0] in ("bullet", "para") and blocks[j][1] > level:
                    i = j + 1
                else:
                    break
    return re.sub(r"\n{3,}", "\n\n", "\n".join(keep))


def chat_hash(meta: dict, body: str) -> str:
    return collect.note_hash(meta, body)


# --- reading the chats --------------------------------------------------------------------------

MAX_TURN_CHARS = 500        # of one "You" turn
MAX_CHAT_CHARS = 2_500      # of one chat's "You" turns together
MAX_TITLE_CHARS = 150
CONTEXT_DIRS = ("2 Areas/People", "2 Areas/Profile")
MAX_CONTEXT_CHARS = 6000


def _cut(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rstrip() + " [cut]"


def you_text(turn: str, source: str) -> str:
    """The user's words in one turn of a chat note (its `## n. You` heading line first): the lines up
    to the answer heading, with every `>` line (the quoted bit of an answer) left out. The answer
    heading is found by common.turns, so `source` is no longer needed; it stays for callers."""
    return split_turn(turn)[0]


def you_turns(body: str, source: str, message_count=None) -> tuple[str, int]:
    """(the user's side of the chat, trimmed to MAX_CHAT_CHARS, number of turns)."""
    turns = parse_turns(body, message_count).turns
    if not turns:   # no "You" headings, so there's no telling the user's words from the answers
        return "(no turns found)", 0
    out, size = [], 0
    for i, t in enumerate(turns):
        line = f"Turn {t.n}: {_cut(t.you, MAX_TURN_CHARS) or '(no text)'}"
        if out and size + len(line) > MAX_CHAT_CHARS:
            out.append(f"[{len(turns) - i} more turns not shown]")
            break
        out.append(line)
        size += len(line)
    return "\n".join(out), len(turns)


def chat_files(vault: Path) -> list[Path]:
    root = Path(vault) / CHATS_DIR
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.md") if p.is_file() and not p.name.startswith("."))


def vault_context(vault, topics: list[str] | None = None) -> str:
    """What the user has told the vault about themself and the people in their chats: the body of
    each note in 2 Areas/People and 2 Areas/Profile, capped at MAX_CONTEXT_CHARS. Empty when there
    are none. A note whose name holds an excluded topic is skipped, and redact_topics takes the
    topics out of the rest before the cap."""
    parts = []
    for rel in CONTEXT_DIRS:
        folder = Path(vault) / rel
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            try:
                meta, body = read_note(path)
            except Exception:   # noqa: BLE001  (one unreadable note leaves out only itself)
                continue
            if meta.get("excluded"):
                continue
            if topics:
                if excluded_topic(path.stem, topics):
                    continue
                body = redact_topics(body, topics)
            parts.append(f"## {rel}/{path.stem}\n\n{body.strip()}\n")
    return "\n".join(parts)[:MAX_CONTEXT_CHARS]


@dataclass
class Chat:
    rel: str
    source: str
    chat_id: str
    title: str
    date: str           # created, YYYY-MM-DD
    stem: str
    sorted: bool        # `sorted:` in its frontmatter (kept in the sidecar, never in the prompt)
    visibility: str
    hash: str
    text: str           # the user's side, trimmed and scrubbed, for the filter prompt
    turns: int
    key: str = ""


def kept_chats(vault: Path, topics: list[str], log=print) -> tuple[list[Chat], Counter]:
    """(chats to score, counts of what was skipped and why). Oldest first."""
    vault = Path(vault)
    out: list[Chat] = []
    skipped: Counter = Counter()
    for path in chat_files(vault):
        rel = collect.rel_path(vault, path)
        try:
            meta, body = parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as e:
            log(f"filter: can't read {rel}, so it's skipped: {type(e).__name__}")
            skipped["unreadable"] += 1
            continue
        if collect.skip_note(meta) or not meta.get("chat_id") or not meta.get("source"):
            skipped["excluded or not a chat"] += 1
            continue
        # The whole title and body, answers included, before any trim: reflect shows more of a chat
        # than the filter does, so a topic past the filter's cut must still keep the chat out.
        if excluded_topic(f"{meta.get('title') or path.stem}\n{body}", topics):
            skipped["excluded topic"] += 1
            continue
        source = str(meta["source"])
        text, n = you_turns(body, source, meta.get("message_count"))
        if not n:
            skipped["no turns"] += 1
            continue
        title = _cut(meta.get("title") or path.stem, MAX_TITLE_CHARS)
        date = str(meta.get("created") or meta.get("updated") or "")[:10]
        out.append(Chat(rel, source, str(meta["chat_id"]), scrub_text(title), date, path.stem,
                        bool(meta.get("sorted")), collect.visibility_of(meta), chat_hash(meta, body),
                        scrub_text(text), n))
    out.sort(key=lambda c: (c.date, c.rel))
    return out, skipped


# --- scores sidecar -----------------------------------------------------------------------------

def scores_path(pstate: Path) -> Path:
    return Path(pstate) / SCORES


def load_scores(pstate: Path) -> dict:
    return guards.load_json(scores_path(pstate), {})


def save_scores(pstate: Path, scores: dict) -> None:
    guards.save_json(scores_path(pstate), scores)


def to_score(chats: list[Chat], scores: dict) -> list[Chat]:
    return [c for c in chats if (scores.get(c.rel) or {}).get("hash") != c.hash]


# --- the call -----------------------------------------------------------------------------------

def build_prompt(batch: list[Chat], context: str = "") -> str:
    parts = [INSTRUCTIONS]
    if context:
        parts.append("\n# People and Profile notes (what the user has told us; these override any guess)\n\n"
                     + context)
    parts.append("\n# Chats\n")
    for c in batch:
        # `sorted:` is set on unsure chats too, so it says nothing about whose chat it is and stays
        # out of the prompt.
        parts.append(f"\n## {c.key}\n\ntitle: {c.title}\nsource: {c.source}, started {c.date or 'unknown'}, "
                     f"{c.turns} turns\n\n{c.text}\n")
    return "".join(parts)


def validate(raw: dict, batch: list[Chat], log=print) -> dict[str, dict]:
    """{key: {score, whose, reason}} for the answers that make sense. Unknown or repeated keys and
    bad values are dropped and logged; a chat with no answer waits for the next run."""
    by_key = {c.key: c for c in batch}
    out: dict[str, dict] = {}
    seen: set[str] = set()
    for item in raw.get("chats") or []:
        key = str(item.get("key", ""))
        if key not in by_key or key in seen:
            log(f"filter: dropped an answer for key {key[:20]!r} (unknown or repeated)")
            continue
        seen.add(key)
        score, whose = item.get("score"), item.get("whose")
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 10 or whose not in WHOSE:
            log(f"filter: dropped {key}: bad score or whose")
            continue
        out[key] = {"score": score, "whose": whose,
                    "reason": propose._one_line(item.get("reason", ""))[:MAX_REASON_CHARS]}
    for c in batch:
        if c.key not in out:
            log(f"filter: no answer for {c.rel}, so it waits for the next run")
    return out


def score_chats(chats: list[Chat], scores: dict, pstate: Path, day: str, budget, cwd: Path,
                runner=None, stop=None, log=print, context: str = "", batch_size: int = BATCH_CHATS,
                model: str = FILTER_MODEL) -> dict:
    """Score `chats` batch by batch, saving the sidecar after each call. Stops at the first failed
    call or when the cap runs out. Returns {"scored", "cost", "models", "error", "calls"}."""
    result = {"scored": 0, "cost": 0.0, "models": set(), "error": None, "calls": 0}
    groups = [chats[i:i + batch_size] for i in range(0, len(chats), batch_size)]
    for i, batch in enumerate(groups, 1):
        for n, c in enumerate(batch, 1):
            c.key = f"C{n}"
        try:
            raw, cost, models = propose.call(build_prompt(batch, context), SCHEMA, budget, cwd, runner, stop,
                                             model=model, max_call_usd=MAX_CALL_USD)
        except propose.ProposeError as e:
            result["error"] = f"call {i} of {len(groups)}: {e}"
            log(f"filter: {result['error']}")
            break
        result["calls"] += 1
        got = validate(raw, batch, log)
        for c in batch:
            if c.key in got:
                scores[c.rel] = {**got[c.key], "hash": c.hash, "chat_id": c.chat_id, "date": c.date,
                                 "month": c.date[:7], "sorted": c.sorted, "model": model, "scored": day}
        save_scores(pstate, scores)
        result["scored"] += len(got)
        result["cost"] += cost
        result["models"].update(models)
        log(f"filter: call {i} of {len(groups)} scored {len(got)} of {len(batch)} chats, ${cost:.4f}")
    result["models"] = sorted(result["models"])
    return result


# --- months -------------------------------------------------------------------------------------

def about_user(entry: dict) -> bool:
    return entry.get("whose") == "user" and int(entry.get("score", 0)) >= PROFILE_MIN


def month_counts(scores: dict, rels: set[str] | None = None) -> dict[str, dict]:
    """{"2025-07": {"scored": n, "about": n}} over the scored chats (only `rels` when given)."""
    out: dict[str, dict] = {}
    for rel, e in scores.items():
        if rels is not None and rel not in rels:
            continue
        m = str(e.get("month") or "")
        if not re.fullmatch(r"\d{4}-\d{2}", m):
            continue
        row = out.setdefault(m, {"scored": 0, "about": 0})
        row["scored"] += 1
        row["about"] += 1 if about_user(e) else 0
    return dict(sorted(out.items()))


def next_month(m: str) -> str:
    y, mo = int(m[:4]), int(m[5:7])
    return f"{y + (mo == 12)}-{1 if mo == 12 else mo + 1:02d}"


def months_between(first: str, last: str) -> list[str]:
    out, m = [], first
    while m <= last:
        out.append(m)
        m = next_month(m)
    return out


def pick_months(counts: dict[str, dict], n: int = 3) -> list[str]:
    """The rule (set before any scores were seen): the run of `n` consecutive calendar
    months with the most chats about the user; a tie goes to the most recent run. Months with no
    chats count as 0."""
    if not counts:
        return []
    months = months_between(min(counts), max(counts))
    if len(months) <= n:
        return months
    best, best_sum = None, -1
    for i in range(len(months) - n + 1):
        run = months[i:i + n]
        total = sum(counts.get(m, {}).get("about", 0) for m in run)
        if total >= best_sum:   # >= so a later run wins a tie
            best, best_sum = run, total
    return best or []


def table(counts: dict[str, dict]) -> str:
    lines = ["month    scored  about the user"]
    for m, row in counts.items():
        lines.append(f"{m}  {row['scored']:>6}  {row['about']:>6}")
    return "\n".join(lines)


# --- the run ------------------------------------------------------------------------------------

def run(cfg, vault: Path | None = None, cap_usd: float | None = None, limit: int | None = None,
        runner=None, now: _dt.datetime | None = None, only_table: bool = False,
        batch_size: int = BATCH_CHATS) -> int:
    now = now or _dt.datetime.now()
    day = now.date().isoformat()
    state = Path(cfg.state)
    state.mkdir(parents=True, exist_ok=True)
    say = lambda m: guards.log(state, m)
    try:
        vault = Path(vault) if vault else cfg.vault
        if not vault.is_dir():
            raise guards.Refused(f"no vault at {vault}")
        pstate = profile_state(state, vault, cfg.vault)
        topics = exclude_topics(state)
        chats, skipped = kept_chats(vault, topics, say)
        if only_table:   # reads only, so no lock
            print(table(month_counts(load_scores(pstate), {c.rel for c in chats})))
            return 0
        if cap_usd is None:
            raise guards.Refused("pass --cap-usd: the filter has no default cap")
        check_stops(state, vault, cfg.vault)
        with profile_lock(state, now):
            scores = load_scores(pstate)   # under the lock, so another run's scores are never saved over
            todo = to_score(chats, scores)
            if limit is not None:
                todo = todo[:max(0, limit)]
            say(f"filter: {len(chats)} chats kept ({dict(skipped) or 'none skipped'}), {len(todo)} to score")
            if not todo:
                print(table(month_counts(scores, {c.rel for c in chats})))
                return 0
            budget = guards.Budget(state, spend_key(day), cap_usd)
            result = score_chats(todo, scores, pstate, day, budget, state / PROFILE_DIR / "work", runner,
                                 stop=lambda: check_stops(state, vault, cfg.vault), log=say,
                                 context=scrub_text(vault_context(vault, topics)), batch_size=batch_size)
            runs = guards.load_json(pstate / FILTER_RUNS, [])
            runs.append({"when": now.isoformat(timespec="seconds"), "scored": result["scored"],
                         "calls": result["calls"], "cost": round(result["cost"], 6), "models": result["models"],
                         "error": result["error"]})
            guards.save_json(pstate / FILTER_RUNS, runs)
            say(f"filter: scored {result['scored']} chats in {result['calls']} calls, models {result['models']}, "
                f"${result['cost']:.4f}; ${budget.spent():.4f} of ${budget.cap:.2f} spent on profile work on {day}")
            print(table(month_counts(scores, {c.rel for c in chats})))
            return 1 if result["error"] else 0
    except guards.Refused as e:
        say(f"filter: refused: {e}")
        return 2


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="python -m dream.filter", description=__doc__.split("\n\n")[0])
    ap.add_argument("--vault", help="score this vault instead of BONSAI_VAULT (a copy)")
    ap.add_argument("--cap-usd", type=float, help="dollar cap for profile work on this day (required to call)")
    ap.add_argument("--limit", type=int, help="score at most this many chats")
    ap.add_argument("--batch", type=int, default=BATCH_CHATS, help="chats per call")
    ap.add_argument("--table", action="store_true", help="print the per-month table and stop")
    args = ap.parse_args(argv)
    try:
        cfg = config.load()
    except config.ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    return run(cfg, Path(args.vault) if args.vault else None, args.cap_usd, args.limit,
               only_table=args.table, batch_size=args.batch)


if __name__ == "__main__":
    sys.exit(main())
