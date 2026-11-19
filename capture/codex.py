"""Local Codex sessions -> one vault note per thread.

    python -m capture codex [--dry-run]     # also part of `python -m capture hourly`

Reads only the session transcripts: `$CODEX_HOME/sessions/**/rollout-*.jsonl` (BONSAI_CODEX_SESSIONS
to point elsewhere; by default ~/.codex/sessions). Nothing else in `.codex` is opened:
not the login file, the SQLite databases, their WAL files, the logs or any other device state. Nothing
is copied or synced; `raw:` points at the rollout file(s) on this computer.

A note makes a Codex thread searchable and readable by agents everywhere. It doesn't make the thread
resumable in Codex on another computer: Codex Remote and Handoff do that, and this doesn't try to.

What a rollout file holds (as of Codex 0.169):
- One JSON record a line: {timestamp, ordinal, type, payload}. The first is `session_meta`: the
  thread `id` (a UUID v7, made on the computer that started the thread), `cwd`, `source`,
  `thread_source`, and for a subagent thread `parent_thread_id`.
- `response_item` records are what the model saw and said: `message` (role user, assistant or
  developer), `reasoning` (`summary` holds the reasoning summaries shown in the app; the rest is
  encrypted and dropped), tool calls (`custom_tool_call`, `function_call` and any other `*_call`) and
  their output (`*_output`), and `compaction`.
- `event_msg` records repeat the same content for the app (`item_completed` with UserMessage,
  AgentMessage, Reasoning, CommandExecution, McpToolCall, FileChange and ContextCompaction items,
  `task_complete`, token counts). Only FileChange is read from them, for the changed-file marker.
- `turn_context` (the model), `compacted` (a compaction, with a replay of old messages that's
  skipped), `world_state`, `token_usage_record`: settings and bookkeeping, not kept.

What's kept: the user's messages, the assistant's replies (progress notes and final answers) and the
reasoning summaries. Tool calls become one marker line per run of calls, file changes a marker with
the file names, and a compaction a marker. Injected context (`<environment_context>` and the like)
is dropped. A message the user wrote around a quote (the app's "Response annotations") keeps the
quoted text and the comment; one that brings in a ChatGPT chat keeps the request and the chat's title.

Continued threads: a thread can go on in a second file, `rollout-<time>-<thread>_<segment>.jsonl`,
whose `session_meta` has the same thread id and a `history_base` with `end_ordinal_exclusive`. The
second file picks up from that point of the first: the first file's records from that ordinal on are
a branch the thread left (the last turn, redone, say), so they're dropped and the second
file's records follow. Any number of segments join this way, in time order, into the parent's note.

Subagent threads (`source` is a subagent, like the guardian reviews of `thread_source:
guardian_review`) are skipped and counted: their "user" is Codex, not the user.

Not seen yet, so not handled specially: archived threads (Codex moves their files out of
`sessions/`; the note stays as last captured), renamed threads (the name isn't in the rollout, so the
title stays the first message's), and whether a thread id is the same on another computer after a
handoff (a UUID v7 is unique, and the note's file name is fixed by it, so a second host would rewrite
the same note and set `host:` to itself).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from capture import sessionnote as sn
from capture.chatnote import Turn
from capture.pull import error_kind
from capture.status import log
from common import config

SOURCE = "codex"
CAPTURE = "codex-rollout"
AGENT_NAME = "Codex"
FILE_RX = re.compile(r"^rollout-(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-(?P<id>[0-9A-Fa-f-]{36})"
                     r"(?:_(?P<segment>[0-9A-Fa-f-]{36}))?\.jsonl$")
INJECTED = re.compile(r"^\s*<(?P<tag>[A-Za-z_][\w -]*)>.*</(?P=tag)>\s*$", re.S)
INJECTED_PREFIXES = ("# AGENTS.md instructions", "<environment_context>", "<user_instructions>")
REQUEST = re.compile(r"(?:^|\n)## My request:\s*\n")


def sessions_dir(cfg) -> Path:
    if getattr(cfg, "codex_sessions", None):
        return Path(cfg.codex_sessions)
    home = os.environ.get("CODEX_HOME")
    return (Path(home) if home else Path.home() / ".codex") / "sessions"


def parse_time(value) -> dt.datetime | None:
    if not value:
        return None
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


# --- one file -> records ---------------------------------------------------------------------------

@dataclass
class Segment:
    path: Path
    meta: dict
    records: list[dict]
    bad: int = 0
    size: int = 0

    @property
    def thread_id(self) -> str:
        m = FILE_RX.match(self.path.name)
        return str(self.meta.get("id") or (m.group("id") if m else self.path.stem))

    @property
    def started(self) -> dt.datetime:
        t = parse_time(self.meta.get("timestamp"))
        if t:
            return t
        m = FILE_RX.match(self.path.name)
        if m:
            return dt.datetime.strptime(m.group("ts"), "%Y-%m-%dT%H-%M-%S").astimezone()
        return dt.datetime.fromtimestamp(self.path.stat().st_mtime, dt.timezone.utc)


def read_segment(path: Path) -> Segment:
    """Every record that parses. A bad line (a partial last line on a live file, say) is counted."""
    records, meta, bad = [], {}, 0
    raw = path.read_bytes()
    # split on "\n" only: splitlines() also breaks on U+2028 and friends, which JSON strings may hold
    for line in raw.decode("utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if not isinstance(rec, dict):
            bad += 1
            continue
        if rec.get("type") == "session_meta" and not meta:
            meta = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
            continue
        records.append(rec)
    return Segment(path, meta, records, bad, len(raw))


def join_segments(segments: list[Segment]) -> list[dict]:
    """One thread's records in order: each continued segment cuts the records before it at its
    history_base.end_ordinal_exclusive, then adds its own."""
    out: list[dict] = []
    for seg in sorted(segments, key=lambda s: s.started):
        base = seg.meta.get("history_base") or {}
        end = base.get("end_ordinal_exclusive") if isinstance(base, dict) else None
        if out and isinstance(end, int) and base.get("thread_id") in (None, seg.thread_id):
            out = [r for r in out if not isinstance(r.get("ordinal"), int) or r["ordinal"] < end]
        out += seg.records
    return out


def is_subagent(meta: dict) -> bool:
    src = meta.get("source")
    return (isinstance(src, dict) and "subagent" in src) or bool(meta.get("parent_thread_id")) \
        or str(meta.get("thread_source") or "") == "guardian_review"


# --- records -> turns ------------------------------------------------------------------------------

def content_text(payload: dict) -> str:
    out = []
    for part in payload.get("content") or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type") or ""
        if kind in ("input_text", "output_text", "text") and isinstance(part.get("text"), str):
            out.append(part["text"])
        elif kind in ("input_image", "image"):
            out.append("[image]")
        elif kind in ("input_file", "file"):
            out.append(f"[file: {part.get('filename') or 'file'}]")
    return "\n\n".join(t for t in out if t.strip())


def unwrap_user(text: str) -> tuple[str | None, str]:
    """(the message as the note shows it, or None for injected context; the text a title comes from)."""
    stripped = text.strip()
    if not stripped or stripped.startswith(INJECTED_PREFIXES) or INJECTED.match(stripped):
        return None, ""
    m = REQUEST.search(text)
    if not m:
        return stripped, stripped
    before, request = text[:m.start()], text[m.end():].strip()
    lead = []
    if "# Response annotations:" in before:
        found = re.search(r"<response-annotations>\s*(\[.*?\])\s*</response-annotations>", before, re.S)
        try:
            items = json.loads(found.group(1)) if found else []
        except ValueError:
            items = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict) or not str(it.get("text") or "").strip():
                continue
            lead.append(sn.quote_block(str(it["text"])))
            if str(it.get("annotation") or "").strip():
                lead.append(f"On that: {str(it['annotation']).strip()}")
        if not lead:
            lead.append("*[quoted an earlier answer]*")
    elif "## Referenced ChatGPT conversation:" in before:
        title = ""
        for line in before.splitlines():
            if line.strip().startswith("{"):
                try:
                    title = str((json.loads(line) or {}).get("title") or "")
                except (ValueError, AttributeError):
                    title = ""
                break
        lead.append(f"*[brought in a ChatGPT chat{': ' + title if title else ''}]*")
    else:
        lead.append(before.strip())
    return "\n\n".join(x for x in lead + [request] if x), request


def changed_files_marker(item: dict) -> str | None:
    changes = item.get("changes")
    if not isinstance(changes, dict) or not changes:
        return None
    names = []
    for path in changes:
        name = re.split(r"[\\/]", str(path))[-1]
        if name and name not in names:
            names.append(name)
    shown = ", ".join(names[:8]) + (f" and {len(names) - 8} more" if len(names) > 8 else "")
    return f"*[changed files: {shown}]*"


def turns_from_records(records: list[dict]) -> tuple[list[Turn], str, str, list[dt.datetime]]:
    """(turns, model, the first message for the title, record times)."""
    turns: list[dict] = []
    model, first, times = "unknown", "", []

    def current() -> dict:
        if not turns:
            turns.append({"prompt": "", "when": None, "parts": []})
        return turns[-1]
    for rec in records:
        rtype, p = rec.get("type"), rec.get("payload")
        if not isinstance(p, dict):
            continue
        when = parse_time(rec.get("timestamp"))
        if when:
            times.append(when)
        if rtype == "turn_context":
            model = str(p.get("model") or model)
        elif rtype == "compacted":
            current()["parts"].append(("mark", "*[context compacted]*"))
        elif rtype == "event_msg" and p.get("type") == "item_completed":
            item = p.get("item") if isinstance(p.get("item"), dict) else {}
            if item.get("type") == "FileChange":
                marker = changed_files_marker(item)
                if marker:
                    current()["parts"].append(("mark", marker))
        elif rtype == "response_item":
            ptype = str(p.get("type") or "")
            if ptype == "message":
                role = p.get("role")
                text = content_text(p)
                if role == "user":
                    shown, plain = unwrap_user(text)
                    if shown is None:
                        continue
                    if not first and plain:
                        first = plain
                    if turns and not turns[-1]["parts"] and turns[-1]["prompt"]:
                        turns[-1]["prompt"] += "\n\n" + shown
                    else:
                        turns.append({"prompt": shown, "when": when, "parts": []})
                elif role == "assistant" and text.strip():
                    current()["parts"].append(("text", text))
            elif ptype == "reasoning":
                summary = "\n\n".join(str(s.get("text") or "") for s in p.get("summary") or []
                                      if isinstance(s, dict) and s.get("text"))
                if summary.strip():
                    current()["parts"].append(("think", summary))
            elif ptype == "compaction":
                current()["parts"].append(("mark", "*[context compacted]*"))
            elif ptype.endswith("_call"):
                current()["parts"].append(("tool", str(p.get("name") or ptype.replace("_call", "").replace("_", " "))))
    out = [Turn(prompt=t["prompt"], answer=sn.render_parts(t["parts"]), date=sn.local_date(t["when"]))
           for t in turns if t["prompt"] or any(k != "tool" for k, _ in t["parts"])]
    return out, model, first, times


def visibility_for(cwd: str, repos: dict | None = None) -> str:
    """By repo, like capture/claude_code.py: a session that ran inside a repo named in `repos`
    (BONSAI_REPO_VISIBILITY) gets that, the rest private."""
    repos = config.REPO_VISIBILITY if repos is None else repos
    for part in re.split(r"[\\/]", cwd or ""):
        if part.lower() in repos:
            return repos[part.lower()]
    return "private"


def build_session(segments: list[Segment], host: str, today: str, repos: dict | None = None) -> sn.Session | None:
    segments = sorted(segments, key=lambda s: s.started)
    head = segments[0]
    turns, model, first, times = turns_from_records(join_segments(segments))
    if not turns:
        return None
    started = head.started
    created = sn.local_date(started) or today
    updated = sn.local_date(max(times)) if times else created
    cwd = str(head.meta.get("cwd") or "")
    title = sn.short_title(first) or "untitled Codex session"
    return sn.Session(
        source=SOURCE, chat_id=head.thread_id, title=title, created=created, updated=max(updated, created),
        turns=turns, agent="codex", agent_name=AGENT_NAME, folder=created[:4],
        raw=[sn.posix(s.path) for s in segments], visibility=visibility_for(cwd, repos), host=host,
        capture=CAPTURE, model=model,
        intro=(f"Codex session, captured from its local transcript. The user's messages, the replies and "
               f"the reasoning summaries are kept as written; tool calls and their output are cut to "
               f"one-line markers, and the full transcript is the `raw:` file on the `host:` computer. This "
               f"note makes the session searchable and readable by agents everywhere; it doesn't make the "
               f"thread resumable in Codex on another computer."),
        extra={"cwd": cwd})


# --- the run ---------------------------------------------------------------------------------------

def run(cfg, dry_run: bool = False, today: str | None = None, host: str | None = None) -> sn.Report:
    today = today or dt.date.today().isoformat()
    report = sn.Report(SOURCE, dry_run=dry_run)
    cursor = sn.FileCursor(cfg, SOURCE, host)
    host = cursor.host
    index = sn.NoteIndex(cfg.vault, SOURCE)
    root = sessions_dir(cfg)
    groups: dict[str, dict[str, tuple[Path, dict]]] = {}
    changed: set[str] = set()
    held: set[str] = set()
    if root.is_dir():
        for path in sorted(root.rglob("rollout-*.jsonl")):
            m = FILE_RX.match(path.name)
            if not m:
                continue
            key = sn.posix(path)
            report.files += 1
            tid = m.group("id").lower()
            try:
                stamp = sn.FileCursor.stamp(path)
            except OSError as e:     # gone since the listing, or locked: wait for it, and leave its thread alone
                report.counts["unreadable files"] += 1
                report.notes_.append(f"couldn't read {path.name} ({error_kind(e)}), skipped this run")
                held.add(tid)
                continue
            groups.setdefault(tid, {})[key] = (path, stamp)
            if cursor.changed(key, stamp):
                report.changed_files += 1
                changed.add(tid)
    report.sessions = len(groups)
    secrets = None
    report.counts["failed"] += len(held & changed)
    for n, tid in enumerate(sorted(changed - held), 1):
        group = groups[tid]
        try:
            segments = [read_segment(path) for path, _ in group.values()]
            report.read += 1
            report.source_bytes += sum(s.size for s in segments)
            report.malformed += sum(s.bad for s in segments)
            head = min(segments, key=lambda s: s.started)
            note = None
            if is_subagent(head.meta):
                report.counts["skipped"] += 1
                report.counts["subagent threads"] += 1
                report.kinds["codex subagent"] += 1
            else:
                report.kinds["codex"] += 1
                if secrets is None:
                    # by value: only the repos in Bonsai's own settings, never a folder a transcript names
                    from capture.claude_code import repo_of
                    secrets = sn.secrets_for([repo_of(d) for d, _ in cfg.specstory_dirs])
                session = build_session(segments, host, today, cfg.repo_visibilities)
                if session is None:
                    report.counts["empty"] += 1
                else:
                    outcome, path, text, redacted = sn.write_session(cfg, session, index, secrets, today, dry_run)
                    report.add_note(outcome, path, text, redacted)
                    note = sn.posix(path.relative_to(cfg.vault))
            cursor.mark({k: v[1] for k, v in group.items()}, tid, note)
            if not dry_run and n % 25 == 0:
                cursor.save()
        except Exception as e:   # noqa: BLE001  (one bad thread must not stop the rest)
            report.counts["failed"] += 1
            if not dry_run:
                log(cfg, f"{SOURCE}: thread {tid[:8]} failed ({error_kind(e)}), will try again next run")
            else:
                report.notes_.append(f"thread {tid[:8]} failed ({error_kind(e)})")
    if not dry_run:
        cursor.save()
        log(cfg, f"{SOURCE}: {report.summary()}")
        for line in report.notes_[:20]:
            log(cfg, f"{SOURCE}: {line}")
    return report
