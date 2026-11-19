"""Create the vault's PARA folders and note templates. Safe to re-run: it never overwrites a file.

    python -m vault.make_folders            # the vault from .env
    python -m vault.make_folders --dry-run  # say what it would make
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import config
from common.notes import write_text_atomic
from common.scrub import scrub_text
from vault.templates import FOLDERS, TEMPLATES, TEMPLATES_DIR


def make(vault: Path, dry_run: bool = False) -> list[str]:
    """Make missing folders and templates. Returns what it made (vault-relative paths)."""
    vault = Path(vault)
    if not vault.is_dir():
        raise SystemExit(f"vault folder not found: {vault}")
    made: list[str] = []
    for rel in FOLDERS:
        path = vault / rel
        if not path.is_dir():
            made.append(rel + "/")
            if not dry_run:
                path.mkdir(parents=True, exist_ok=True)
    for name, text in TEMPLATES.items():
        path = vault / TEMPLATES_DIR / name
        if path.exists():
            continue
        made.append(f"{TEMPLATES_DIR}/{name}")
        if not dry_run:
            write_text_atomic(path, scrub_text(text))
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = config.load()
    made = make(cfg.vault, args.dry_run)
    verb = "would make" if args.dry_run else "made"
    for rel in made:
        print(f"{verb}: {rel}")
    if not made:
        print("nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
