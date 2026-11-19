"""Run Bonsai's night job on a made-up vault, with a stand-in for the claude CLI.

    python demo/run_demo.py            # from the repo root
    python demo/run_demo.py --keep     # leave the temp folder in place to look around

What it does:
1. Copies demo/sample-vault/ (a small made-up vault: two inbox notes, two memory notes, a project note
   and four captured chats) into a new temp folder, makes the rest of the PARA folders and the
   note templates there, and points Bonsai at it with BONSAI_VAULT, BONSAI_STATE and
   BONSAI_MACHINE. No .env is read.
2. Runs the night job twice, the way it's meant to be run: first on a copy of the vault (the
   rehearsal the real vault waits for), then on the vault itself. Each run collects what's new,
   asks "claude" what it means for the memory notes, checks every proposal, and writes the
   morning digest to `0 Inbox/digest <date>.md`.
3. Prints where the digest is and its first lines.

The stand-in for claude (fake_claude below) returns canned proposals in the CLI's JSON shape, so
nothing reaches the network, no claude CLI is needed, nothing is spent, and nothing is written
outside the temp folder.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLE = Path(__file__).resolve().parent / "sample-vault"
NOW = dt.datetime(2026, 11, 18, 3, 0, 0)     # the night after the sample chats were captured
DIGEST_LINES = 45

sys.dont_write_bytecode = True     # no __pycache__ in the repo: the demo writes only to its temp folder
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# What the stand-in proposes, by the source note it reads. A source that isn't listed gets no
# proposal (the trivia chat, say), which is the most common real answer too.
CANNED = {
    "3 Resources/chats/chatgpt/2026/2026-11-16 tomato blight on the lower leaves (Ab12Cd34).md": [{
        "action": "add", "target": "1 Projects/Garden/memory/check-lower-leaves-on-saturdays.md",
        "reason": "The user set a weekly blight check", "description": "the weekly tomato blight check",
        "type": "project", "date": "2026-11-16",
        "text": "Check the tomatoes' lower leaves every Saturday and pull any with brown rings.\n\n"
                "**Why:** early blight showed up on the lower leaves on 2026-11-16.\n\n"
                "**How to apply:** garden reminders include the Saturday leaf check.",
    }],
    "3 Resources/chats/chatgpt/2026/2026-11-17 sourdough starter schedule (Jk90Lm12).md": [{
        "action": "profile", "target": "2 Areas/Profile/bakes-sourdough-on-sundays.md",
        "reason": "The user stated their baking day", "description": "weekly sourdough bake",
        "date": "2026-11-17", "text": "The user bakes sourdough every Sunday and feeds the starter on Saturday night.",
    }],
    "0 Inbox/win/watering in the evening.md": [{
        "action": "flag", "target": "1 Projects/Garden/memory/water-in-the-morning.md",
        "reason": "A new note says the watering moved to the evening",
        "text": "The memory note says to water before 08:00. The inbox note says the watering moved to "
                "after 19:00 on 2026-11-10.",
    }],
    "0 Inbox/win/bike to the market.md": [{
        "action": "add", "target": "2 Areas/Errands/bike-to-the-saturday-market.md",
        "reason": "The user decided how to get to the market", "description": "how the user gets to the market",
        "type": "user", "date": "2026-11-17",
        "text": "The user takes the bike to the Saturday market, not the car. Both panniers hold two full bags.",
    }],
    "3 Resources/chats/claude-code/garden-planner/2026-11-17 add a frost date check (5e551011).md": [{
        "action": "add", "target": "1 Projects/Garden/memory/last-frost-date-lives-in-config.md",
        "reason": "The user set where the frost date lives", "description": "where garden-planner keeps the last frost date",
        "type": "feedback", "date": "2026-11-17",
        "text": "The last frost date (2026-04-20) lives in garden.toml as last_frost, never in the code.\n\n"
                "**Why:** the user asked for it in the config file when the frost check was added.\n\n"
                "**How to apply:** dates and other settings in garden-planner go in garden.toml.",
    }],
}

_SOURCE_RX = re.compile(r"^## (S\d+): \w+ note (.+?) \(dated ", re.M)


def fake_claude(args: list[str], stdin_text: str, cwd: Path) -> dict:
    """Stands in for `claude -p ... --output-format json`: reads which sources the prompt lists
    (`## S1: chat note <path> (dated ...)`) and answers with the canned proposals for them, in the
    CLI's result shape."""
    from dream import propose
    proposals = []
    for key, rel in _SOURCE_RX.findall(stdin_text):
        for p in CANNED.get(rel, []):
            proposals.append({**p, "sources": [key]})
    return {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.01,
            "modelUsage": {propose.MODEL: {}}, "structured_output": {"proposals": proposals}}


def run(workdir: Path, out=print) -> tuple[int, Path]:
    """Copy the sample vault into workdir, run the night job there, and return (exit code, digest path)."""
    vault, state = workdir / "vault", workdir / "state"
    shutil.copytree(SAMPLE, vault)
    os.environ.update(BONSAI_VAULT=str(vault), BONSAI_STATE=str(state), BONSAI_MACHINE="win",
                      BONSAI_DREAM_FIRST_DAYS="3")
    from common import config
    from dream import guards, night
    from vault import make_folders
    cfg = config.load(workdir / "no.env").require()       # the env vars above, and no .env file
    make_folders.make(cfg.vault)

    code = night.run(cfg, on_copy=True, cap_usd=0.50, runner=fake_claude, now=NOW)
    if code != 0:
        out(f"the rehearsal on a copy stopped with exit code {code}; see {guards.dream_root(state) / 'night.log'}")
        return code, Path()
    out(f"rehearsal on a copy of the vault: done (the copy is in {guards.dream_root(state) / 'vault-copy'})")

    code = night.run(cfg, cap_usd=0.50, runner=fake_claude, now=NOW)
    path = cfg.vault / "0 Inbox" / f"digest {NOW.date().isoformat()}.md"
    if code != 0 or not path.exists():
        out(f"the night run stopped with exit code {code}; see the log in {guards.dream_root(state)}")
        return code or 1, path
    return 0, path


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--keep", action="store_true", help="leave the temp folder in place")
    args = ap.parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix="bonsai-demo-"))
    try:
        code, path = run(workdir)
        if code == 0:
            lines = path.read_text(encoding="utf-8").splitlines()
            print(f"\nThe digest is at {path}\nIts first {min(DIGEST_LINES, len(lines))} lines:\n")
            print("\n".join(lines[:DIGEST_LINES]))
            print(f"\nThe temp folder stays at {workdir}" if args.keep else
                  "\nThe temp folder is removed now; run with --keep to look around in it.")
        return code
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
