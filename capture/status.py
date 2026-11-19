"""The capture log: BONSAI_STATE/capture.log, one scrubbed line per event.

Every capture source writes here: what it read, what it wrote, and why it stopped. A line names an
error by its kind only (capture/pull.py's error_kind), never its text, since a YAML or JSON error
quotes the note or transcript it choked on. Nothing here ever writes a cookie, token or header value.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from common.scrub import scrub_text

LOG_NAME = "capture.log"


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(cfg, line: str) -> None:
    """One line in BONSAI_STATE/capture.log (scrubbed, so a stray key can't land there either)."""
    path = Path(cfg.state) / LOG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{now()} {scrub_text(line.strip())}\n")
