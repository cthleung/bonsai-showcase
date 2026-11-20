"""Claude Code (and the other agents SpecStory records) -> one vault note per session.

    python -m capture claude-code [--dry-run]     # also part of `python -m capture hourly`

Reads `.specstory/history/*.md` in each folder of BONSAI_SPECSTORY_DIRS (no default; entries split
on `;`, each with an optional `=<visibility>`). SpecStory
writes one Markdown file per session there, already synced through each repo's git.

What a SpecStory file looks like (Markdown v2.1):
    <!-- Claude Code Session <id> (<UTC time>) -->      the agent and the session id
    _**User (2026-11-19 02:42:55Z)**_                    a message from the user
    _**Agent (claude-opus-5-5 2026-11-19 02:43:02Z)**_   a reply (VS Code Copilot: `(copilot/<model>)`)
    <tool-use data-tool-type="shell" data-tool-name="Bash"><details> ... </details></tool-use>
    <think><details><summary>Thought Process</summary> ... </details></think>
The history holds four agents: Claude Code, VS Code Copilot, Copilot CLI and Codex CLI. All four are
captured under this source by default, each note with `agent:` set and its own answer heading;
BONSAI_SPECSTORY_AGENTS (comma separated: claude-code, copilot, copilot-cli, codex-cli) narrows it.
A Codex CLI session whose rollout file is in Codex's own sessions folder is left to `capture codex`,
so the same thread doesn't get two notes.

Trimming: the user's messages and the agent's replies are kept, and so is the agent's reasoning, as a
quote. Every tool call and its output becomes one marker line (`*[3 tool calls: Bash x2, Read]*`),
since a tool block holds file dumps and command output. A tool block is skipped with its code fences
and nested tool blocks tracked, so a transcript quoted inside a tool's output (a role line, a
`</tool-use>`) can't end it early. Text the editor injects into a message (system reminders, the open
file, a selection, local command output) becomes a short marker or is dropped, in replies too. A
subagent's steps (`_**Agent - sidechain (...)**_`, interleaved with the main thread) become one
marker per run (`*[a subagent worked: 12 steps]*`), so they never fold into the main answer.

Duplicates: SpecStory sometimes writes one session to two or more files (a second file name for the
same session, or a resumed session in a new file). Files are grouped by session id. A file whose turns
are all in a bigger file of the group is dropped, and the rest are joined in time order.

The note: `3 Resources/chats/claude-code/<repo>/<YYYY-MM-DD> <title> (<id8>).md`, `capture: specstory`,
`raw:` the SpecStory file(s) on this computer (never copied). Visibility is set by repo: the folder's
`=<visibility>`, else the repo's entry in BONSAI_REPO_VISIBILITY (`novel=creative`, say), else private.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

from capture import sessionnote as sn
from capture.chatnote import Turn
from capture.status import log
from capture.pull import error_kind
from common import config

SOURCE = "claude-code"
CAPTURE = "specstory"
HEAD_BYTES = 4096

KINDS = {   # SpecStory's name for the agent -> (agent key, answer heading)
    "claude code": ("claude-code", "Claude Code"),
    "vs code copilot ide": ("copilot", "Copilot"),
    "copilot cli": ("copilot-cli", "Copilot CLI"),
    "codex cli": ("codex-cli", "Codex"),
}

HEADER = re.compile(r"<!--\s*(?P<kind>[^<>]*?)\s+Session\s+(?P<id>[A-Za-z0-9_-]+)\s*(?:\((?P<when>[^)]*)\))?\s*-->")
# `_**Agent - sidechain (...)**_` is a subagent's step (Claude Code's Task tool), interleaved with the
# main thread. A long session can hold thousands of these lines.
ROLE = re.compile(r"^_\*\*(?P<role>User|Agent)(?P<side>\s*-\s*sidechain)?(?:\s*\((?P<info>[^)]*)\))?\*\*_\s*$")
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})Z?\s*$")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
TOOL_NAME = re.compile(r'data-tool-name="([^"]*)"')
TOOL_OPEN = re.compile(r"^<tool-use[\s>]")     # not `<tool-use-id>`, which task notifications hold
FILE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}Z-?")


def kind_of(name: str) -> tuple[str, str]:
    key = (name or "").strip().lower()
    if key in KINDS:
        return KINDS[key]
    slug = re.sub(r"[^a-z0-9]+", "-", key).strip("-") or "unknown"
    return slug, name.strip() or "Agent"


def parse_stamp(text: str | None) -> dt.datetime | None:
    m = STAMP.search(text or "")
    if not m:
        return None
    return dt.datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}+00:00")


# --- one file -> blocks -------------------------------------------------------------------------

@dataclass
class Block:
    role: str                   # User | Agent
    when: dt.datetime | None
    model: str | None
    parts: list = field(default_factory=list)   # ("text", s) | ("tool", name) | ("think", s)

    def add_text(self, line: str) -> None:
        if self.parts and self.parts[-1][0] == "text":
            self.parts[-1] = ("text", self.parts[-1][1] + "\n" + line)
        else:
            self.parts.append(("text", line))

    def text(self) -> str:
        return "\n".join(v for k, v in self.parts if k == "text")


@dataclass
class Header:
    kind: str = ""
    session_id: str = ""
    when: dt.datetime | None = None


def read_header(path: Path) -> Header | None:
    with open(path, "rb") as f:
        head = f.read(HEAD_BYTES).decode("utf-8", errors="replace").replace("\x00", "")
    m = HEADER.search(head)
    if not m:
        return None
    return Header(m.group("kind"), m.group("id"), parse_stamp(m.group("when")))


def _closes(line: str, fence: tuple[str, int]) -> bool:
    s = line.strip()
    return bool(s) and set(s) == {fence[0]} and len(s) >= fence[1]


def parse_blocks(text: str, stats: dict | None = None) -> list[Block]:
    """The role blocks in a SpecStory file, tool blocks collapsed to ("tool", name) and reasoning
    to ("think", text). stats, when given, counts a file that ends inside a tool block."""
    lines = text.replace("\x00", "").replace("\r\n", "\n").split("\n")
    blocks: list[Block] = []
    cur: Block | None = None
    mode, depth, fence, think = None, 0, None, []
    for line in lines:
        if mode == "tool":
            if fence:
                if _closes(line, fence):
                    fence = None
                elif fence[1] == 3 and line.rstrip().endswith("</details></tool-use>"):
                    # SpecStory doesn't always lengthen its fence around content with fences in it
                    # (tool arguments, say), which leaves a plain ``` fence open. Its own
                    # closing line ends the block anyway. A transcript quoted in a tool's output sits
                    # in a ```` fence (or longer), where this line stays content.
                    fence = None
                    depth -= 1
                    if depth <= 0:
                        mode = None
                continue
            m = FENCE.match(line)
            if m:
                fence = (m.group(1)[0], len(m.group(1)))
                continue
            stripped = line.strip()
            if TOOL_OPEN.match(stripped):
                depth += 1
            if stripped.endswith("</tool-use>"):
                depth -= 1
                if depth <= 0:
                    mode = None
            continue
        m = ROLE.match(line)
        if mode == "think":
            if "</think>" in line or m:      # a think block left open ends at the next turn
                if not m:
                    think.append(line.split("</think>", 1)[0])
                cur.parts.append(("think", _think_text(think)))
                mode, think = None, []
                if not m:
                    continue
            else:
                think.append(line)
                continue
        if m:
            info = m.group("info") or ""
            when = parse_stamp(info)
            model = (STAMP.sub("", info).strip() or None) if m.group("role") == "Agent" and not m.group("side") else None
            if model and "/" in model:                 # VS Code Copilot: `copilot/claude-haiku-4.5`
                model = model.split("/", 1)[1] or model
            cur = Block("Sidechain" if m.group("side") else m.group("role"), when, model)
            blocks.append(cur)
            continue
        if cur is None:
            continue
        stripped = line.strip()
        if TOOL_OPEN.match(stripped):
            name = TOOL_NAME.search(line)
            cur.parts.append(("tool", name.group(1) if name else "tool"))
            if not stripped.endswith("</tool-use>"):
                mode, depth, fence = "tool", 1, None
            continue
        if stripped.startswith("<think>"):
            rest = stripped[len("<think>"):]
            if "</think>" in rest:
                cur.parts.append(("think", _think_text([rest.split("</think>", 1)[0]])))
            else:
                mode, think = "think", [rest]
            continue
        cur.add_text(line)
    if mode == "think" and cur is not None and think:
        cur.parts.append(("think", _think_text(think)))
    if stats is not None and mode == "tool":
        stats["open_tool"] = stats.get("open_tool", 0) + 1   # a tool block never closed: the rest of the file was skipped
    return blocks


def _think_text(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"</?details>|<summary>.*?</summary>", "", text)
    return text.strip()


# --- cleaning ------------------------------------------------------------------------------------

def file_name(path: str) -> str:
    """The last part of a path from any computer: `c:\\x\\notes.md` and `/x/notes.md` are both
    `notes.md`. Path() on a Mac or Linux box reads a backslash as part of the name, so a Windows path
    there would come back whole. PureWindowsPath splits on both kinds of slash and drops a drive
    (`c:notes.md`) too, on any computer."""
    return PureWindowsPath(path.strip()).name


_DROP = re.compile(r"<(system-reminder|browser_instruction|local-command-caveat|user-prompt-submit-hook)>.*?</\1>", re.S)
_MARKS = [
    (re.compile(r"<ide_opened_file>.*?opened the file (.+?) in the IDE.*?</ide_opened_file>", re.S),
     lambda m: f"*[had {file_name(m.group(1))} open in the editor]*"),
    (re.compile(r"<ide_opened_file>.*?</ide_opened_file>", re.S), lambda m: "*[had a file open in the editor]*"),
    (re.compile(r"<ide_selection>.*?</ide_selection>", re.S), lambda m: "*[selected text in the editor]*"),
    (re.compile(r"<local-command-stdout>.*?</local-command-stdout>", re.S), lambda m: "*[command output]*"),
    (re.compile(r"<persisted-output>.*?</persisted-output>", re.S), lambda m: "*[tool output]*"),
    (re.compile(r"<task-notification>.*?</task-notification>", re.S), lambda m: "*[a background task finished]*"),
    (re.compile(r"<command-message>.*?</command-message>", re.S), lambda m: ""),
    (re.compile(r"<command-name>(.*?)</command-name>", re.S), lambda m: m.group(1).strip()),
    (re.compile(r"<command-args>(.*?)</command-args>", re.S), lambda m: m.group(1).strip()),
    (re.compile(r"\[Image: source: [^\]\n]*\]"), lambda m: "[image]"),
]


# Whole "user" messages that the user didn't write: the agent's own summary after compaction, another
# agent's message, Codex's history handoff and a skill's instructions. Each becomes one marker, so the
# profile and dream jobs never read them as the user's words.
_NOT_THE_USER = [
    (re.compile(r"^This session is being continued from a previous conversation"),
     "*[context compacted: the session went on from a summary]*"),
    (re.compile(r"^Another Claude session sent a message:"), "*[a message from another Claude session]*"),
    (re.compile(r"^The following is the Codex agent history"), "*[Codex added its agent history]*"),
    (re.compile(r"^(?:Base directory for this skill:|# [^\n]{1,60} Skill\n)"), "*[skill instructions loaded]*"),
]


def clean_user(text: str) -> str:
    text = _DROP.sub("", text)
    for rx, fn in _MARKS:
        text = rx.sub(fn, text)
    text = _trim(text)
    for rx, marker in _NOT_THE_USER:
        if rx.match(text):
            return marker
    return text


def _trim(text: str) -> str:
    """Drop SpecStory's `---` turn separators and blank lines at the ends of a block."""
    lines = text.strip("\n").split("\n")
    while lines and lines[-1].strip() in ("", "---"):
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


# --- a session (one or more files) -> turns ------------------------------------------------------

@dataclass
class ParsedFile:
    path: Path
    header: Header
    blocks: list[Block]
    size: int
    mtime: float

    def signature(self) -> set:
        """What a file's turns are, to spot a file whose turns are all in another file."""
        return {(b.role, b.when) for b in self.blocks if b.when is not None}

    def first(self) -> dt.datetime:
        times = [b.when for b in self.blocks if b.when] or ([self.header.when] if self.header.when else [])
        return min(times) if times else dt.datetime.fromtimestamp(self.mtime, dt.timezone.utc)


def pick_files(files: list[ParsedFile]) -> list[ParsedFile]:
    """Drop files whose turns are all in the kept files, biggest first; the rest in time order."""
    kept, seen = [], set()
    for f in sorted(files, key=lambda f: (len(f.signature()), f.size, f.mtime), reverse=True):
        sig = f.signature()
        if kept and sig <= seen:
            continue
        kept.append(f)
        seen |= sig
    return sorted(kept, key=lambda f: f.first())


def turns_from_blocks(blocks: list[Block]) -> tuple[list[Turn], str, list[dt.datetime]]:
    turns: list[dict] = []
    model, times = "unknown", []
    for b in blocks:
        if b.when:
            times.append(b.when)
        if b.role == "User":
            text = clean_user(b.text())
            if not text:
                continue
            if turns and not turns[-1]["parts"]:      # a message in two blocks (text, then an image)
                turns[-1]["prompt"] = f"{turns[-1]['prompt']}\n\n{text}".strip()
                continue
            turns.append({"prompt": text, "when": b.when, "parts": []})
            continue
        if not turns:
            turns.append({"prompt": "", "when": b.when, "parts": []})
        if b.role == "Sidechain":        # a subagent's step: one marker for a run of them
            turns[-1]["parts"].append(("side", ""))
            continue
        if b.model:
            model = b.model
        for kind, value in b.parts:
            if kind in ("text", "think"):
                value = _DROP.sub("", value)
            if kind == "text":
                value = _trim(value)
            turns[-1]["parts"].append((kind, value))
    out = [Turn(prompt=t["prompt"], answer=sn.render_parts(t["parts"]), date=sn.local_date(t["when"]))
           for t in turns]
    return out, model, times


def title_from(path: Path, turns: list[Turn]) -> str:
    slug = FILE_DATE.sub("", path.stem).replace("-", " ").strip()
    if slug and slug.lower() != "untitled":
        return slug
    first = next((t.prompt for t in turns if t.prompt), "")
    if not first:      # a subagent's session has no message from the user: the first reply line
        lines = (line for t in turns for line in t.answer.split("\n"))
        first = next((line for line in lines if line.strip() and not line.startswith(("*[", ">"))), "")
    return sn.short_title(first) or "untitled session"


def repo_of(folder: Path) -> Path:
    """The repo a history folder belongs to: `<repo>/.specstory/history` -> `<repo>`."""
    folder = Path(folder)
    if folder.name == "history" and folder.parent.name == ".specstory":
        return folder.parent.parent
    return folder.parent


def visibility_for(folder: Path, explicit: str | None, repos: dict | None = None) -> str:
    """The folder's own `=<visibility>`, else the repo's entry in `repos` (BONSAI_REPO_VISIBILITY),
    else private."""
    if explicit:
        return explicit
    repos = config.REPO_VISIBILITY if repos is None else repos
    return repos.get(repo_of(folder).name.lower(), "private")


def build_session(files: list[ParsedFile], folder: Path, visibility: str, host: str | None, today: str) -> sn.Session | None:
    """One session from its files. host isn't recorded: the SpecStory history syncs through git, so
    the computer reading it isn't where the session ran."""
    kept = pick_files(files)
    blocks: list[Block] = []
    earlier: set = set()
    for f in kept:     # a block a joined file repeats from an earlier one is dropped (same block within a file stays)
        keys = []
        for b in f.blocks:
            key = (b.role, b.when, repr(b.parts)[:500])
            keys.append(key)
            if b.when is None or key not in earlier:
                blocks.append(b)
        earlier.update(keys)
    turns, model, times = turns_from_blocks(blocks)
    if not turns:
        return None
    head = kept[0].header
    agent, agent_name = kind_of(head.kind)
    primary = max(kept, key=lambda f: (f.size, f.mtime))
    created = sn.local_date(min(times) if times else kept[0].first()) or today
    updated = sn.local_date(max(times)) if times else created
    repo = repo_of(folder)
    return sn.Session(
        source=SOURCE, chat_id=head.session_id, title=title_from(primary.path, turns)[:200],
        created=created, updated=max(updated, created), turns=turns, agent=agent, agent_name=agent_name,
        folder=repo.name or "unknown", raw=[sn.posix(f.path) for f in kept], visibility=visibility,
        host=None, capture=CAPTURE, model=model,
        intro=(f"{agent_name} session in {repo.name}, captured from SpecStory. The user's messages and the "
               f"replies are kept as written; tool calls, their output and subagent steps are cut to one-line "
               f"markers, and the full transcript is the `raw:` file. This note makes the "
               f"session searchable and readable by agents everywhere; it doesn't make it resumable on "
               f"another computer."),
        extra={"repo": repo.name})


def parse_file(path: Path, header: Header, stats: dict | None = None) -> ParsedFile:
    raw = path.read_bytes()
    st = path.stat()
    return ParsedFile(path, header, parse_blocks(raw.decode("utf-8", errors="replace"), stats),
                      st.st_size, st.st_mtime)


# --- the run ---------------------------------------------------------------------------------------

def codex_ids(cfg) -> set[str]:
    """Thread ids that have a rollout file in Codex's sessions folder (capture/codex.py takes those)."""
    from capture import codex
    root = codex.sessions_dir(cfg)
    if not root.is_dir():
        return set()
    return {m.group("id").lower() for p in root.rglob("rollout-*.jsonl") if (m := codex.FILE_RX.match(p.name))}


def inputs(cfg) -> list[tuple[Path, str | None]]:
    return [(Path(d), vis) for d, vis in cfg.specstory_dirs if Path(d).is_dir()]


def run(cfg, dry_run: bool = False, today: str | None = None, host: str | None = None) -> sn.Report:
    today = today or dt.date.today().isoformat()
    report = sn.Report(SOURCE, dry_run=dry_run)
    cursor = sn.FileCursor(cfg, SOURCE, host)
    host = cursor.host
    index = sn.NoteIndex(cfg.vault, SOURCE)
    folders = inputs(cfg)
    secrets = sn.secrets_for([repo_of(d) for d, _ in folders])
    wanted = set(cfg.specstory_agents or ())
    taken_by_codex = codex_ids(cfg)
    stats: dict = {}

    for folder, explicit in folders:
        visibility = visibility_for(folder, explicit, cfg.repo_visibilities)
        groups: dict[str, dict[str, tuple[Path, dict, Header | None]]] = {}
        changed_ids: set[str] = set()
        held: set[str] = set()       # sessions with a file that couldn't be read this run: wait for it
        for path in sorted(folder.glob("*.md")):
            key = sn.posix(path)
            report.files += 1
            sid, header = cursor.known_id(key), None
            try:
                stamp = sn.FileCursor.stamp(path)
                if cursor.changed(key, stamp) or not sid:
                    header = read_header(path)
            except OSError as e:     # locked by a sync, or gone since the listing: skip just this file
                report.counts["unreadable files"] += 1
                report.notes_.append(f"couldn't read {path.name} ({error_kind(e)}), skipped this run")
                if sid:
                    held.add(sid)
                continue
            if cursor.changed(key, stamp) or not sid:
                report.changed_files += 1
                if header is None:
                    report.counts["skipped"] += 1
                    report.notes_.append(f"no session header: {path.name}")
                    continue
                sid = header.session_id
                changed_ids.add(sid)
            groups.setdefault(sid, {})[key] = (path, stamp, header)
        report.sessions += len(groups)
        report.counts["failed"] += len(held & changed_ids)
        for n, sid in enumerate(sorted(changed_ids - held), 1):
            group = groups[sid]
            try:
                headers = {}
                for key, (path, stamp, header) in group.items():
                    header = header or read_header(path)
                    if header is not None:
                        headers[key] = header
                if not headers:
                    continue
                agent, _ = kind_of(next(iter(headers.values())).kind)
                report.kinds[agent] += 1
                # Not marked in the cursor: a later change of setting (or a rollout that goes away)
                # brings them back. Their headers are read again each run, which is cheap.
                if wanted and agent not in wanted:
                    report.counts["skipped"] += 1
                    continue
                if agent == "codex-cli" and sid.lower() in taken_by_codex:
                    report.counts["skipped"] += 1
                    report.counts["codex-cli left to capture codex"] += 1
                    continue
                parsed = []
                for key, header in headers.items():
                    path, stamp, _ = group[key]
                    parsed.append(parse_file(path, header, stats))
                    report.source_bytes += stamp["size"]
                report.read += 1
                session = build_session(parsed, folder, visibility, host, today)
                note = None
                if session is None:
                    report.counts["empty"] += 1
                else:
                    outcome, path, text, redacted = sn.write_session(cfg, session, index, secrets, today, dry_run)
                    report.add_note(outcome, path, text, redacted)
                    if outcome in ("new", "updated"):
                        report.counts["sidechain markers"] += text.count(sn.SIDE_MARK)
                    note = sn.posix(path.relative_to(cfg.vault))
                cursor.mark({k: v[1] for k, v in group.items()}, sid, note)
                if not dry_run and n % 25 == 0:
                    cursor.save()
            except Exception as e:   # noqa: BLE001  (one bad session must not stop the rest)
                report.counts["failed"] += 1
                if not dry_run:
                    log(cfg, f"{SOURCE}: session {sid[:8]} failed ({error_kind(e)}), will try again next run")
                else:
                    report.notes_.append(f"session {sid[:8]} failed ({error_kind(e)})")
    if stats.get("open_tool"):
        report.notes_.append(f"{stats['open_tool']} file(s) end inside a tool block that never closes; "
                             f"anything after it in those files is missing from the note")
    report.notes_ = list(getattr(cfg, "specstory_warnings", ()) or ()) + report.notes_
    if not dry_run:
        cursor.save()
        log(cfg, f"{SOURCE}: {report.summary()}")
        for line in report.notes_[:20]:
            log(cfg, f"{SOURCE}: {line}")
    return report
