"""Apply the boxes the user ticked in the morning digests, after a snapshot.

    python -m dream.apply              apply every ticked box not applied yet
    python -m dream.apply --dry-run    say what would happen, change nothing
    python -m dream.apply --undo       put back what the last apply changed

The snapshot is a git repo kept outside the vault: git --git-dir=<BONSAI_STATE>/vault-history
--work-tree=<BONSAI_VAULT>. Each apply makes a "before" commit, changes the notes, then an "after"
commit, and records both, with the exact paths it changed, in <vault state>/applies.json. --undo
restores only those paths, and leaves alone (and names) any of them edited again since then.

A box counts as ticked only when its whole line matches the line the digest wrote for it (see
dream/digest.py). A ticked box whose line doesn't match is logged by its id (never its text), and
the next digest lists it at the top. An item that errors is marked failed in the sidecar and never
retried.

What a tick does:
    add, profile   write a new memory note (docs/note-format.md), source: author-approved
    merge          add the proposed lines to the end of the existing note
    move           move the note to the new folder (same file name)
    flag           nothing; it marks the contradiction as seen
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

from common import config
from common.notes import read_note, write_note
from dream import guards
from dream.digest import DIGEST_DIR, sidecar_path, wikilink

EXCLUDES = [".obsidian*/", ".trash/", "*.tmp", ".*.tmp", "desktop.ini", ".DS_Store", "Icon?", "~$*"]


class ApplyError(RuntimeError):
    pass


# --- the snapshot repo ----------------------------------------------------------------------

class Snapshot:
    """git with its repo in BONSAI_STATE and its work tree the vault."""

    def __init__(self, git_dir: Path, vault: Path):
        self.git_dir = Path(git_dir)
        self.vault = Path(vault)

    def run(self, *args: str) -> subprocess.CompletedProcess:
        cmd = ["git", f"--git-dir={self.git_dir}", f"--work-tree={self.vault}",
               "-c", "core.autocrlf=false", "-c", "core.quotepath=off", "-c", "safe.directory=*",
               "-c", "user.name=bonsai-dream", "-c", "user.email=dream@bonsai.invalid",
               "-c", "commit.gpgsign=false", "-c", "core.hooksPath=", *args]
        return subprocess.run(cmd, capture_output=True, cwd=str(self.vault))

    def git(self, *args: str, binary: bool = False):
        proc = self.run(*args)
        if proc.returncode != 0:
            raise ApplyError(f"git {args[0]} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
        return proc.stdout if binary else proc.stdout.decode("utf-8", "replace")

    def ensure(self) -> None:
        if not (self.git_dir / "HEAD").exists():
            self.git_dir.parent.mkdir(parents=True, exist_ok=True)
            self.git("init", "-q")
        info = self.git_dir / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "exclude").write_text("\n".join(EXCLUDES) + "\n", encoding="utf-8")

    def commit(self, message: str) -> str:
        self.ensure()
        self.git("add", "-A", ".")
        self.git("commit", "-q", "--allow-empty", "--no-verify", "-m", message)
        return self.git("rev-parse", "HEAD").strip()

    def changed(self, before: str, after: str) -> list[tuple[str, str]]:
        """[(status letter, path)] between two commits, renames shown as delete + add."""
        out = self.git("diff", "--name-status", "--no-renames", "-z", before, after)
        parts = [p for p in out.split("\0") if p]
        return [(parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)]

    def spelled(self, before: str, after: str, paths: list[str]) -> list[str]:
        """The paths as git spells them. Windows and macOS find `Water.md` when the file is
        `water.md`, but git doesn't, so each path becomes the changed path(s) between the two
        commits that match it when letter case is ignored. A path with no match stays as it is."""
        changed = [path for _, path in self.changed(before, after)]
        out = []
        for rel in paths:
            out += [c for c in changed if c.casefold() == rel.casefold()] or [rel]
        return list(dict.fromkeys(out))

    def blob(self, commit: str, path: str) -> bytes | None:
        """The file's bytes at that commit, or None if it wasn't there."""
        proc = self.run("cat-file", "blob", f"{commit}:{path}")
        return proc.stdout if proc.returncode == 0 else None


# --- reading the digests --------------------------------------------------------------------

_BOX_RX = re.compile(r"^- \[(.)\] ")
_TICKED_ID_RX = re.compile(r"^\s*[-*+] \[[xX]\] .*\^(p\d+)$")


def ticked_in(text: str, proposals: list[dict]) -> list[dict]:
    """The proposals ticked in this digest text. A tick counts only when exactly one line equals the
    proposal's stored `line` (the box line as written) except for the box character, and that box
    holds x or X. A proposal with no stored line never counts."""
    lines = [l.rstrip() for l in text.splitlines()]
    plain = [_BOX_RX.sub("- [ ] ", l, count=1) for l in lines]
    out = []
    for p in proposals:
        want = p.get("line")
        if not want:
            continue
        hits = [i for i, l in enumerate(plain) if l == want.rstrip()]
        if len(hits) == 1 and _BOX_RX.match(lines[hits[0]]).group(1) in ("x", "X"):
            out.append(p)
    return out


def mismatched_in(text: str, proposals: list[dict]) -> list[str]:
    """Ids of the proposals with a ticked box ending in their `^pNN` that ticked_in doesn't count:
    the line was indented, edited, or had a link rewritten."""
    ids = {m.group(1) for m in (_TICKED_ID_RX.match(l.rstrip()) for l in text.splitlines()) if m}
    counted = {p["id"] for p in ticked_in(text, proposals)}
    return [p["id"] for p in proposals if p["id"] in ids and p["id"] not in counted]


def ticked(vault: Path, vstate: Path, log=print, mismatched: list | None = None) -> list[tuple[str, dict]]:
    """[(day, proposal)] for every ticked box in every digest that isn't applied, undone or failed.
    A ticked box whose line doesn't match is logged, and added to `mismatched` as (day, id)."""
    out = []
    folder = Path(vault) / DIGEST_DIR
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("digest *.md")):
        day = path.stem[len("digest "):]
        try:
            side = guards.load_json(sidecar_path(vstate, day), None)
            text = path.read_text(encoding="utf-8")
        except (OSError, ValueError) as e:   # ValueError covers bad JSON and bad UTF-8
            log(f"apply: can't read {path.name} or its sidecar, so it's skipped: {e}")
            continue
        if not side:
            continue
        waiting = [p for p in side.get("proposals", [])
                   if not p.get("applied") and not p.get("undone") and not p.get("failed")]
        out += [(day, p) for p in ticked_in(text, waiting)]
        for pid in mismatched_in(text, waiting):
            log(f"apply: ticked but the line doesn't match: {pid} (digest {day}), so it's not applied")
            if mismatched is not None:
                mismatched.append((day, pid))
    return out


def mismatched(vault: Path, vstate: Path) -> list[tuple[str, str]]:
    """(day, id) for every ticked box waiting in a digest whose line doesn't match, for the next digest."""
    out: list[tuple[str, str]] = []
    ticked(vault, vstate, log=lambda m: None, mismatched=out)
    return out


# --- doing one proposal ---------------------------------------------------------------------

def _source_line(p: dict) -> str:
    return "Source: " + ", ".join(wikilink(l) for l in p["links"])


def apply_one(vault: Path, p: dict, today: str, inbox: str = "win") -> tuple[str, list[str]]:
    """Make the change for one ticked proposal. Returns (a short line saying what it did, the
    vault-relative paths it changed), so undo touches only those."""
    vault = Path(vault)
    target = vault / p["target"]
    action = p["action"]
    if action in ("add", "profile"):
        if target.exists():
            raise ApplyError(f"{p['target']} exists now")
        meta = {"name": p["name"], "description": p["description"], "type": p["type"],
                "source": "author-approved", "date": p["date"], "visibility": p["visibility"]}
        write_note(target, meta, f"{p['text'].strip()}\n\n{_source_line(p)}\n")
        return f"wrote {p['target']}", [p["target"]]
    if action == "merge":
        if not target.is_file():
            raise ApplyError(f"{p['target']} is gone")
        meta, body = read_note(target)
        body = f"{body.rstrip()}\n\n{p['text'].strip()} ({today})\n{_source_line(p)}\n"
        write_note(target, meta, body)
        return f"added to {p['target']}", [p["target"]]
    if action == "move":
        if p["target"].startswith("0 Inbox/") and not p["target"].startswith(f"0 Inbox/{inbox}/"):
            raise ApplyError(f"{p['target']} is in another computer's inbox; only that computer moves it")
        if not target.is_file():
            raise ApplyError(f"{p['target']} is gone")
        dest_rel = f"{p['move_to'].rstrip('/')}/{PurePosixPath(p['target']).name}"
        dest = vault / dest_rel
        if dest.exists():
            raise ApplyError(f"{dest_rel} exists")
        dest.parent.mkdir(parents=True, exist_ok=True)
        target.replace(dest)
        return f"moved {p['target']} to {p['move_to']}", [p["target"], dest_rel]
    if action == "flag":
        return f"seen: {p['target']}", []
    raise ApplyError(f"unknown action {action}")


def _mark(vstate: Path, day: str, pid: str, fields: dict) -> None:
    path = sidecar_path(vstate, day)
    data = guards.load_json(path, {})
    for p in data.get("proposals", []):
        if p["id"] == pid:
            p.update(fields)
    guards.save_json(path, data)


def _try_mark(vstate: Path, day: str, pid: str, fields: dict, log) -> None:
    try:
        _mark(vstate, day, pid, fields)
    except Exception as e:
        log(f"apply: couldn't mark {day} {pid} in its sidecar: {type(e).__name__}: {e}")


def apply_ticked(vault: Path, vstate: Path, git_dir: Path, today: str, dry_run: bool = False, log=print,
                 inbox: str = "win") -> dict:
    """Apply every ticked box. Returns {"done", "failed", "paths", "before", "after"}.

    Any error in one item marks that item failed (later nights skip it) and moves on. The after
    commit and the applies.json record are written in a finally, so a crash part way still leaves a
    record of what changed. If the before snapshot fails, nothing is changed and ApplyError is raised."""
    todo = ticked(vault, vstate, log)
    report = {"done": [], "failed": [], "paths": [], "before": None, "after": None}
    if not todo:
        log("apply: no ticked boxes waiting")
        return report
    if dry_run:
        for day, p in todo:
            report["done"].append(f"would {p['action']} {p['target']} ({day} {p['id']})")
            log(f"apply (dry run): would {p['action']} {p['target']} ({day} {p['id']})")
        return report
    snap = Snapshot(git_dir, vault)
    report["before"] = snap.commit(f"before apply {today}")
    log(f"apply: snapshot {report['before'][:10]} taken before {len(todo)} change(s)")
    paths: list[str] = []
    try:
        for day, p in todo:
            try:
                what, changed = apply_one(vault, p, today, inbox)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                report["failed"].append(f"{p['action']} {p['target']}: {msg}")
                log(f"apply: failed {day} {p['id']} {p['action']} {p['target']}: {msg}")
                _try_mark(vstate, day, p["id"], {"failed": today, "error": msg[:300]}, log)
                continue
            paths += changed
            report["done"].append(what)
            log(f"apply: {day} {p['id']} {what}")
            _try_mark(vstate, day, p["id"], {"applied": today, "paths": changed}, log)
        if report["done"]:
            try:   # Index.md has one writer, vault.index; rebuilt inside the snapshot so undo covers it
                from vault import index as vault_index
                if vault_index.write(vault)[0]:
                    paths.append("Index.md")
            except Exception as e:   # a stale index never undoes the night's applied work
                log(f"apply: Index.md not rebuilt: {type(e).__name__}: {e}")
    finally:
        report["paths"] = list(dict.fromkeys(paths))
        try:
            report["after"] = snap.commit(f"after apply {today}: {len(report['done'])} change(s)")
        except Exception as e:
            log(f"apply: the after snapshot failed: {type(e).__name__}: {e}")
        if report["after"]:
            try:   # undo looks paths up in git, so record them the way git spells them
                report["paths"] = snap.spelled(report["before"], report["after"], report["paths"])
            except Exception as e:
                log(f"apply: paths kept as apply wrote them: {type(e).__name__}: {e}")
        records = guards.load_json(Path(vstate) / "applies.json", [])
        records.append({"when": _dt.datetime.now().isoformat(timespec="seconds"), "before": report["before"],
                        "after": report["after"], "done": report["done"], "failed": report["failed"],
                        "paths": report["paths"], "ids": [[d, p["id"]] for d, p in todo]})
        guards.save_json(Path(vstate) / "applies.json", records)
    log(f"apply: done, {len(report['done'])} applied, {len(report['failed'])} failed; "
        f"undo with: python -m dream.apply --undo (plus --vault <path> for a copy)")
    return report


# --- undo -----------------------------------------------------------------------------------

def _sha(data: bytes | None) -> str | None:
    return None if data is None else hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def undo_last(vault: Path, vstate: Path, git_dir: Path, log=print, records_file: str = "applies.json",
              unmark=None) -> dict:
    """Put back the files the last apply changed: only the paths it recorded (what apply_one
    touched, plus Index.md), never anything else that changed in the vault meanwhile. A file
    changed again since the apply is left alone. Another apply step can use this with its own
    `records_file` and `unmark(day, id, when)`, so the two undos never cross. Returns
    {"restored", "kept", "record"}."""
    records = guards.load_json(Path(vstate) / records_file, [])
    live = [r for r in records if not r.get("undone")]
    if not live:
        log("undo: nothing to undo")
        return {"restored": [], "kept": [], "record": None}
    rec = live[-1]
    if "paths" not in rec or not rec.get("after"):
        raise ApplyError("the last apply has no list of changed paths or no after snapshot; "
                         f"undo it by hand from the snapshot {rec.get('before')}")
    snap = Snapshot(git_dir, vault)
    snap.commit("before undo")
    restored, kept = [], []
    for rel in rec["paths"]:
        path = Path(vault) / rel
        now = path.read_bytes() if path.is_file() else None
        if _sha(now) != _sha(snap.blob(rec["after"], rel)):
            kept.append(rel)
            log(f"undo: left alone, changed since the apply: {rel}")
            continue
        old = snap.blob(rec["before"], rel)
        if old is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.undo.tmp")   # temp then rename, so a sync client never uploads a partial file
            tmp.write_bytes(old)
            tmp.replace(path)
        restored.append(rel)
        log(f"undo: put back {rel}")
    snap.commit(f"undo apply {rec['after'][:10]}")
    rec["undone"] = _dt.datetime.now().isoformat(timespec="seconds")
    guards.save_json(Path(vstate) / records_file, records)
    for day, pid in rec.get("ids", []):
        if unmark:
            try:
                unmark(day, pid, rec["undone"])
            except Exception as e:
                log(f"undo: couldn't mark {day} {pid} as undone: {type(e).__name__}: {e}")
        else:
            _try_mark(vstate, day, pid, {"applied": None, "undone": rec["undone"]}, log)
    return {"restored": restored, "kept": kept, "record": rec}


# --- command line ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dream.apply", description=__doc__.split("\n\n")[0])
    ap.add_argument("--vault", help="apply to this vault instead of BONSAI_VAULT (a copy)")
    ap.add_argument("--dry-run", action="store_true", help="say what would happen, change nothing")
    ap.add_argument("--undo", action="store_true", help="put back what the last apply changed")
    args = ap.parse_args(argv)
    cfg = config.load()
    vault = Path(args.vault) if args.vault else cfg.vault
    if not vault.is_dir():
        print(f"no vault at {vault}", file=sys.stderr)
        return 1
    vstate = guards.vault_state_dir(cfg.state, vault, cfg.vault)
    git_dir = guards.history_dir(cfg.state, vault, cfg.vault)
    say = lambda m: guards.log(cfg.state, m)
    try:
        guards.check_stop(cfg.state, vault)
        with guards.Lock(cfg.state):
            if args.undo:
                undo_last(vault, vstate, git_dir, log=say)
            else:
                apply_ticked(vault, vstate, git_dir, _dt.date.today().isoformat(), args.dry_run, log=say,
                             inbox=cfg.inbox_name)
    except guards.Refused as e:
        say(f"refused: {e}")
        return 2
    except Exception as e:   # the record of what changed is already written (see apply_ticked)
        say(f"apply failed: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
