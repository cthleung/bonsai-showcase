"""Bonsai settings, read from the repo's .env (see .env.example).

    from common import config
    cfg = config.load()          # Config(vault=Path, state=Path, machine="win", ...)
    cfg.vault / "0 Inbox" / cfg.inbox_name

Real environment variables win over .env, so tests and one-off runs can point Bonsai at a temp
vault with BONSAI_VAULT=... without touching .env. Nothing here runs at import time, and load()
doesn't check the folders exist: callers that need them call cfg.require().
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_FILE = REPO / ".env"
MACHINES = ("win", "mac")
VISIBILITIES = ("private", "creative", "career", "public")
# The visibility a repo's agent sessions get when BONSAI_SPECSTORY_DIRS doesn't say, by repo folder
# name, lowercased. Empty by default (every repo is private); BONSAI_REPO_VISIBILITY fills it in.
REPO_VISIBILITY: dict[str, str] = {}
KEYS = ("BONSAI_VAULT", "BONSAI_STATE", "BONSAI_MACHINE", "BONSAI_SPECSTORY_DIRS", "BONSAI_SPECSTORY_AGENTS",
        "BONSAI_CODEX_SESSIONS", "BONSAI_REPO_VISIBILITY")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    vault: Path
    state: Path
    machine: str
    # SpecStory history folders for capture/claude_code.py: ((folder, visibility or None), ...).
    specstory_dirs: tuple = ()
    # Which agents' SpecStory sessions to capture (claude-code, copilot, copilot-cli, codex-cli); () is all.
    specstory_agents: tuple = ()
    # entries of BONSAI_SPECSTORY_DIRS or BONSAI_REPO_VISIBILITY that were left out, for capture to log
    specstory_warnings: tuple = ()
    codex_sessions: Path | None = None     # Codex's session transcripts; None means $CODEX_HOME/sessions or ~/.codex/sessions
    # ((repo folder name lowercased, visibility), ...) from BONSAI_REPO_VISIBILITY; () means REPO_VISIBILITY
    repo_visibility: tuple = ()

    @property
    def inbox_name(self) -> str:
        """This computer's inbox folder under `0 Inbox/` (one writer per folder)."""
        return self.machine

    @property
    def repo_visibilities(self) -> dict[str, str]:
        """{repo folder name lowercased: visibility} for agent sessions."""
        return dict(self.repo_visibility) if self.repo_visibility else dict(REPO_VISIBILITY)

    def require(self) -> "Config":
        """Raise ConfigError unless the vault exists. Creates the state folder if missing."""
        if not self.vault.is_dir():
            raise ConfigError(f"BONSAI_VAULT is not a folder: {self.vault}")
        self.state.mkdir(parents=True, exist_ok=True)
        return self


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines, # comments, optional quotes. No variable expansion."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def load(env_file: Path | None = None) -> Config:
    """Settings from env_file (default: the repo .env), overridden by real environment variables."""
    values = read_env_file(env_file or ENV_FILE)
    for key in KEYS:
        if os.environ.get(key):
            values[key] = os.environ[key]
    missing = [k for k in ("BONSAI_VAULT", "BONSAI_STATE") if not values.get(k)]
    if missing:
        raise ConfigError(f"missing {', '.join(missing)}: copy .env.example to .env and fill it in")
    machine = values.get("BONSAI_MACHINE", "").strip().lower() or ("mac" if sys.platform == "darwin" else "win")
    if machine not in MACHINES:
        raise ConfigError(f"BONSAI_MACHINE must be one of {MACHINES}, got {machine!r}")
    dirs, dir_warnings = parse_specstory_dirs(values.get("BONSAI_SPECSTORY_DIRS", "").strip())
    repos, repo_warnings = parse_repo_visibility(values.get("BONSAI_REPO_VISIBILITY", "").strip())
    return Config(
        vault=Path(values["BONSAI_VAULT"]).expanduser(),
        state=Path(values["BONSAI_STATE"]).expanduser(),
        machine=machine,
        specstory_dirs=dirs,
        specstory_warnings=dir_warnings + repo_warnings,
        specstory_agents=tuple(a.strip().lower() for a in values.get("BONSAI_SPECSTORY_AGENTS", "").split(",")
                               if a.strip()),
        codex_sessions=Path(values["BONSAI_CODEX_SESSIONS"]).expanduser() if values.get("BONSAI_CODEX_SESSIONS") else None,
        repo_visibility=repos,
    )


def parse_specstory_dirs(value: str) -> tuple[tuple, tuple]:
    """`~/novel/.specstory/history=creative;~/work/.specstory/history` ->
    (((Path, "creative"), (Path, None)), (warnings...)).
    Entries are split on `;` (a Windows path has a `:` in it). `=<visibility>` is optional; without it
    the repo's visibility comes from the repo visibility map, else private (capture/claude_code.py).
    An entry with an unknown visibility is left out with a warning instead of stopping every command."""
    out, warnings = [], []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        vis = None
        if "=" in part:
            part, vis = (x.strip() for x in part.rsplit("=", 1))
            vis = vis.lower()
            if vis not in VISIBILITIES:
                warnings.append(f"BONSAI_SPECSTORY_DIRS: left out {part}: visibility must be one of "
                                f"{', '.join(VISIBILITIES)}, got {vis!r}")
                continue
        out.append((Path(part).expanduser(), vis))
    return tuple(out), tuple(warnings)


def parse_repo_visibility(value: str) -> tuple[tuple, tuple]:
    """`novel=creative;work=career` -> ((("novel", "creative"), ("work", "career")), (warnings...)).
    Repo names are folder names, matched lowercased. A bad entry is left out with a warning."""
    out, warnings = [], []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, vis = (x.strip() for x in part.partition("="))
        vis = vis.lower()
        if not name or vis not in VISIBILITIES:
            warnings.append(f"BONSAI_REPO_VISIBILITY: left out {part!r}: write it as <repo>=<visibility>, "
                            f"with visibility one of {', '.join(VISIBILITIES)}")
            continue
        out.append((name.lower(), vis))
    return tuple(out), tuple(warnings)
