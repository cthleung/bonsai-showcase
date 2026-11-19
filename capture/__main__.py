"""python -m capture <command>

    hourly                     the local agent sessions (claude-code, then codex), for a scheduler
    claude-code [--dry-run]    SpecStory sessions (Claude Code, Copilot, Codex CLI) from BONSAI_SPECSTORY_DIRS
    codex [--dry-run]          local Codex threads from $CODEX_HOME/sessions. Both read files on this
                               computer only; --dry-run reads everything and reports the counts, and
                               writes no note, cursor or log line
    chatgpt-import PATH [--dry-run]   the official ChatGPT export (the ZIP, or the folder it unzipped
                               to): every chat into the vault. --dry-run reports the counts and
                               writes nothing
    status                     the cursors and the end of the capture log

Every source reads files on this computer: no login, no browser, no network.

Exit codes: 0 ok, 1 a source stopped on an error (logged), 2 skipped (nothing to read on this
computer, or another capture run holds the lock).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from capture import status
from common import config

LOCK_STALE = 3 * 3600


@contextlib.contextmanager
def run_lock(cfg):
    """One capture run at a time (they share the cursors). A lock older than 3 hours is left over
    from a crash and gets taken over."""
    path = Path(cfg.state) / "capture.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and time.time() - path.stat().st_mtime > LOCK_STALE:
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        yield False
        return
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield True
    finally:
        path.unlink(missing_ok=True)


SKIPPED = object()     # a job returns this when it had nothing to do on this computer (exit code 2)


def guarded(cfg, source: str, job) -> int:
    """Run one source's job. An error stops it with one log line (the error's kind only, since its
    text can quote a note). Nothing retries. A job that returns SKIPPED counts as skipped (2)."""
    from capture.pull import error_kind
    try:
        return 2 if job() is SKIPPED else 0
    except Exception as e:   # noqa: BLE001  (yaml.YAMLError, OSError, ...: their text can quote a note, so kind only)
        status.log(cfg, f"{source}: stopped ({error_kind(e)})")
        return 1


def show_status(cfg) -> None:
    for f in sorted((Path(cfg.state) / "cursors").glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8") or "{}")
        n = len(data.get("chats") or data.get("threads") or data.get("sessions") or {})
        print(f"cursor {f.stem}: {n} chats, last run {data.get('last_run', 'never')}")
    log = Path(cfg.state) / status.LOG_NAME
    if log.exists():
        print("\n".join(log.read_text(encoding="utf-8").splitlines()[-15:]))


def chatgpt_import(cfg, path: Path, dry_run: bool) -> int:
    """The export import. A real run takes the run lock (it writes the chatgpt cursor) and logs one
    line; a dry run does neither, and writes nothing."""
    from capture import chatgpt_export
    from capture.pull import error_kind

    def run() -> int:
        try:
            report = chatgpt_export.import_export(cfg, path, dry_run=dry_run)
        except chatgpt_export.ExportError as e:
            print(e)
            return 1
        for line in report.lines():
            print(line)
        if not dry_run:
            status.log(cfg, f"chatgpt-export: {report.total} chats in the export, {report.summary()}")
        return 0

    if dry_run:
        return run()
    with run_lock(cfg) as got:
        if not got:
            print("another capture run holds the lock; try again when it's done")
            return 2
        try:
            return run()
        except Exception as e:   # noqa: BLE001  (its text can quote a note or the export, so kind only)
            status.log(cfg, f"chatgpt-export: stopped ({error_kind(e)})")
            print(f"stopped ({error_kind(e)})")
            return 1


LOCAL_SOURCES = {   # sources that read files on this computer: no login, no browser, no network
    "claude-code": "SpecStory history in BONSAI_SPECSTORY_DIRS",
    "codex": "Codex's session transcripts",
}


def local_sources() -> dict:
    """{source: (has_input(cfg), run(cfg, dry_run=False))}."""
    from capture import claude_code, codex
    return {
        "claude-code": (lambda cfg: bool(claude_code.inputs(cfg)), claude_code.run),
        "codex": (lambda cfg: codex.sessions_dir(cfg).is_dir(), codex.run),
    }


def print_report(report, show: bool) -> None:
    if show:
        for line in report.lines():
            print(line)


def local_dry_run(cfg, source: str) -> int:
    """Read everything, write nothing: no note, no cursor, no log line, no lock."""
    from capture.pull import error_kind
    has_input, run = local_sources()[source]
    try:
        if not has_input(cfg):
            print(f"{source}: no input folder on this computer ({LOCAL_SOURCES[source]})")
            return 2
        report = run(cfg, dry_run=True)
    except Exception as e:   # noqa: BLE001  (its text can quote a session, so kind only)
        print(f"{source}: stopped ({error_kind(e)})")
        return 1
    print_report(report, True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m capture", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("hourly", "status"):
        sub.add_parser(name)
    im = sub.add_parser("chatgpt-import", help="import the official ChatGPT export ZIP or folder")
    im.add_argument("path", type=Path)
    im.add_argument("--dry-run", action="store_true", help="report the counts and write nothing")
    for name, what in LOCAL_SOURCES.items():
        lp = sub.add_parser(name, help=f"capture local agent sessions now ({what})")
        lp.add_argument("--dry-run", action="store_true",
                        help="read everything and report the counts; write no note, cursor or log line")
    args = p.parse_args(argv)

    cfg = config.load().require()
    if args.cmd == "status":
        show_status(cfg)
        return 0
    if args.cmd == "chatgpt-import":
        return chatgpt_import(cfg, args.path, args.dry_run)
    if args.cmd in LOCAL_SOURCES and args.dry_run:
        return local_dry_run(cfg, args.cmd)

    with run_lock(cfg) as got:
        if not got:
            status.log(cfg, f"capture {args.cmd}: skipped, another capture run holds the lock")
            return 2
        jobs = []
        for source, (has_input, run) in local_sources().items():
            if args.cmd not in ("hourly", source):
                continue

            def local_job(source=source, has_input=has_input, run=run):
                # inside guarded, so a check that fails (a locked folder, say) stops only this source
                if not has_input(cfg):        # nothing to read on this computer: skipped
                    if args.cmd == source:
                        print(f"{source}: no input folder on this computer ({LOCAL_SOURCES[source]})")
                    return SKIPPED
                print_report(run(cfg), args.cmd != "hourly")
            jobs.append((source, local_job))
        codes = [guarded(cfg, source, fn) for source, fn in jobs]
        ran = [c for c in codes if c != 2]   # 2 is "skipped": it counts only when nothing ran
        return max(ran) if ran else 2


if __name__ == "__main__":
    sys.exit(main())
