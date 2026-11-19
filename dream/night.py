"""The night run: apply last morning's ticks, read what's new, propose, write the digest.

    python -m dream.night --on-copy --cap-usd 0.50   copy the vault into BONSAI_STATE and run there
    python -m dream.night                             the real vault (only after a clean copy run)
    python -m dream.night --dry-run                   call the model, write nothing to the vault,
                                                      move no watermark; prints the digest instead

Guards, in order: only the PC (BONSAI_MACHINE=win) runs it; the real vault is refused until a run
on a copy has finished cleanly; a STOP file (BONSAI_STATE/dream/STOP, or "0 Inbox/STOP dream.md"
in the vault) stops it; a lock file stops a second copy; a dollar cap per night
(--cap-usd, else BONSAI_DREAM_CAP_USD, else $1.00) stops the claude calls, and no call starts that
won't fit what's left of it.

It reads only new material: the inbox, plus chats captured and updated since the last finished run
(the first run looks back BONSAI_DREAM_FIRST_DAYS days, 3 by default). Older chats that capture
writes later, such as the ChatGPT backfill, are left for the profile spike (dream/collect.py).

Exit codes: 0 done, 1 failed, 2 refused by a guard.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import stat
import sys
from pathlib import Path

from common import config
from dream import apply as apply_mod
from dream import collect, digest, guards, propose

COPY_IGNORE = shutil.ignore_patterns(".obsidian*", ".trash", "*.tmp", "desktop.ini")


def make_copy(src: Path, state: Path, real_vault: Path) -> Path:
    """A fresh copy of the vault at BONSAI_STATE/dream/vault-copy, with fresh state, so every copy
    run is a full rehearsal."""
    dest = guards.dream_root(state) / "vault-copy"
    if dest.exists():
        remove_tree(dest)
    copy_state = guards.vault_state_dir(state, dest, real_vault)
    if copy_state.exists() and copy_state != guards.dream_root(state):
        remove_tree(copy_state)
    shutil.copytree(src, dest, ignore=COPY_IGNORE)
    return dest


def _clear_read_only(func, path, _exc):
    """Give the owner full rights on the folder `path` is in and on `path` itself, then try again.
    Windows refuses to delete a read-only file; macOS and Linux refuse to delete anything in a
    read-only folder, whatever the file's own mark. The current bits are kept (a bare S_IWRITE
    would take away a folder's read and execute bits). A symlink is never changed, since chmod
    would change what it points to."""
    for p in (os.path.dirname(path), path):
        if p and not os.path.islink(p):
            try:
                os.chmod(p, stat.S_IMODE(os.lstat(p).st_mode) | stat.S_IRWXU)
            except OSError:
                pass
    func(path)


def remove_tree(path: Path) -> None:
    """shutil.rmtree that also removes read-only files and folders. A sync client can mark vault
    folders read-only on Windows, and copytree copies that mark, so a second copy run would fail
    with "Access is denied". On the Mac the folder's own mark is what blocks the delete
    ("Permission denied")."""
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_clear_read_only)
    else:
        shutil.rmtree(path, onerror=_clear_read_only)


def _read_list(pieces: list[collect.Piece]) -> list[dict]:
    notes: dict[str, dict] = {}
    for p in pieces:
        r = notes.setdefault(p.rel, {"link": p.link, "turns": list(p.turns) if p.turns else None})
        if p.turns and r["turns"]:
            r["turns"] = [min(r["turns"][0], p.turns[0]), max(r["turns"][1], p.turns[1])]
    return list(notes.values())


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def run(cfg: config.Config, vault: Path | None = None, on_copy: bool = False, dry_run: bool = False,
        cap_usd: float | None = None, do_apply: bool = True, runner=None, now: _dt.datetime | None = None) -> int:
    now = now or _dt.datetime.now()
    day = now.date().isoformat()
    state = cfg.state
    state.mkdir(parents=True, exist_ok=True)
    say = lambda m: guards.log(state, m)
    try:
        if cfg.machine != "win":
            raise guards.Refused("only the PC runs the night job (BONSAI_MACHINE=win)")
        vault = Path(vault) if vault else cfg.vault
        if not vault.is_dir():
            raise guards.Refused(f"no vault at {vault}")
        guards.check_stop(state, vault)
        with guards.Lock(state, now):
            if on_copy:
                vault = make_copy(vault, state, cfg.vault)
                say(f"night: running on a copy at {vault}")
            real = guards.same_path(vault, cfg.vault)
            if real and not dry_run and not guards.rehearsed(state):
                raise guards.Refused("the real vault waits for one clean run on a copy: run with --on-copy first")
            return _night(cfg, vault, real, day, now, dry_run, cap_usd, do_apply, runner, say)
    except guards.Refused as e:
        say(f"night: refused: {e}")
        return 2


def _night(cfg, vault, real, day, now, dry_run, cap_usd, do_apply, runner, say) -> int:
    state = cfg.state
    vstate = guards.vault_state_dir(state, vault, cfg.vault)
    say(f"night: the night run of {day} starts on {vault}{' (dry run)' if dry_run else ''}")

    if do_apply:
        try:
            apply_mod.apply_ticked(vault, vstate, guards.history_dir(state, vault, cfg.vault), day,
                                   dry_run=dry_run, log=say, inbox=cfg.inbox_name)
        except Exception as e:   # apply records what it did in a finally; the night just stops
            say(f"night: apply failed, so the night stops here: {type(e).__name__}: {e}")
            return 1
    guards.check_stop(state, vault)

    st = guards.load_json(vstate / "state.json", {"watermark": None, "seen": {}})
    first_days = int(os.environ.get("BONSAI_DREAM_FIRST_DAYS") or collect.FIRST_DAYS)
    since = collect.since_date(st.get("watermark"), now.date(), first_days)
    history: list[str] = []
    pieces = collect.collect(vault, st["seen"], since=since, history=history)
    say(f"night: reading the inbox and chats new since {since}"
        f"{' (first run, ' + _plural(first_days, 'day') + ' back)' if not st.get('watermark') else ''}; "
        f"{_plural(len(history), 'older chat')} captured since then left for the profile spike")
    if not pieces:
        say("night: nothing new to read")
        if not dry_run:
            st["watermark"] = now.isoformat(timespec="seconds")
            guards.save_json(vstate / "state.json", st)
        return 0
    batches, left = collect.pack(pieces)
    say(f"night: {len(pieces)} pieces to read in {len(batches)} call(s), {len(left)} left for another night")

    cap = cap_usd if cap_usd is not None else float(os.environ.get("BONSAI_DREAM_CAP_USD") or guards.DEFAULT_CAP_USD)
    budget = guards.Budget(state, day, cap)
    existing = collect.existing_notes(vault)
    result = propose.propose(batches, vault, existing, collect.homes(vault), day, budget,
                             cwd=guards.dream_root(state) / "work", runner=runner,
                             taken=digest.pending_keys(vstate), log=say, inbox=cfg.inbox_name,
                             stop=lambda: guards.check_stop(state, vault))
    for r in result["rejected"]:
        say(f"night: rejected {r}")
    if not result["read"]:
        say(f"night: no call worked, so no digest: {result['error'] or result.get('capped')}")
        return 1

    read = result["read"]
    chats = len({p.rel for p in read if p.kind == "chat"})
    inbox = len({p.rel for p in read if p.kind == "inbox"})
    waiting = len({p.rel for p in left}) + len({p.rel for p in pieces if p not in read and p not in left})
    stats = (f"Read {_plural(chats, 'chat')} and {_plural(inbox, 'inbox note')} on {day}: "
             f"{_plural(len(result['proposals']), 'proposal')}. Cost ${result['cost']:.2f}.")
    if waiting:
        stats += f" {_plural(waiting, 'note')} wait for the next night (the nightly cap, or a call that stopped)."
    if history:
        stats += (f" {_plural(len(history), 'older chat')} captured since {since} "
                  f"{'was' if len(history) == 1 else 'were'} left for the profile spike.")
    guards.check_stop(state, vault)   # a STOP made during the calls means no digest
    try:
        path, text = digest.write(vault, vstate, day, now.strftime("%H:%M"), result["proposals"], result["held"],
                                  _read_list(read), stats, dry_run=dry_run,
                                  mismatched=apply_mod.mismatched(vault, vstate))
    except ValueError as e:
        say(f"night: {e}")
        return 1

    if dry_run:
        print(text)
        say(f"night: dry run done, nothing written; models {result['models']}, ${result['cost']:.4f}")
        return 0
    collect.mark_seen(st["seen"], read)
    if waiting or result["error"]:
        # the window stays where it is, so what wasn't read is still new material next night
        say(f"night: the watermark stays at {since}: {_plural(waiting, 'note')} not read yet")
        st["watermark"] = st.get("watermark") or since   # a first run pins its window, so it doesn't slide
    else:
        st["watermark"] = now.isoformat(timespec="seconds")
    guards.save_json(vstate / "state.json", st)
    say(f"night: digest written to {path}; {len(result['proposals'])} proposals, {len(result['held'])} held, "
        f"models {result['models']}, ${result['cost']:.4f}, ${budget.spent():.4f} of ${budget.cap:.2f} spent on {day}")
    if not real and not result["error"]:
        guards.mark_rehearsed(state, vault, now.isoformat(timespec="seconds"))
    return 1 if result["error"] else 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="python -m dream.night", description=__doc__.split("\n\n")[0])
    ap.add_argument("--vault", help="run on this vault instead of BONSAI_VAULT")
    ap.add_argument("--on-copy", action="store_true", help="copy the vault into BONSAI_STATE and run on the copy")
    ap.add_argument("--dry-run", action="store_true", help="write nothing to the vault, move no watermark")
    ap.add_argument("--cap-usd", type=float, help="dollar cap for tonight's claude calls")
    ap.add_argument("--no-apply", action="store_true", help="skip applying ticked boxes")
    args = ap.parse_args(argv)
    try:
        cfg = config.load()
    except config.ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    return run(cfg, Path(args.vault) if args.vault else None, args.on_copy, args.dry_run, args.cap_usd,
               not args.no_apply)


if __name__ == "__main__":
    sys.exit(main())
