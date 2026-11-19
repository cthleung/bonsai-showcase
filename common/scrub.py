"""Redact keys and tokens from text before it reaches the vault or a log.

    from common.scrub import scrub_text
    clean = scrub_text(text)                 # redacts known shapes + the secret values in .env
    clean, hits = scrub(text, load_secrets())  # hits = {"OPENAI_KEY": 2, ...}, never the values

Two ways a key gets caught:
  1. By value: every secret-looking value in .env (16+ characters, no spaces, not a path or URL
     of ours) is replaced wherever it turns up.
  2. By shape: Anthropic, OpenAI, Tavily, Hugging Face, TinyFish, Modal, Discord webhooks, Exa,
     GitHub tokens, JWTs and JWEs, AWS access keys and Google API keys.

Each hit becomes [REDACTED:NAME]. common.notes.write_note calls this on every note by default.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from common.config import ENV_FILE, REPO, read_env_file

SECRET_FILES = [ENV_FILE, REPO / ".claude" / "settings.local.json"]
# Config values that are long and space-free but not secrets.
SKIP_NAMES = {"BONSAI_VAULT", "BONSAI_STATE", "BONSAI_MACHINE", "BONSAI_CAPTURE_CLAUDE",
              "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS", "CLAUDE_CLI",
              "BONSAI_SPECSTORY_DIRS", "BONSAI_CODEX_SESSIONS"}

SHAPES = [
    ("MODAL_TOKEN_ID", re.compile(r"\bak-[A-Za-z0-9]{20,}\b")),
    ("MODAL_TOKEN_SECRET", re.compile(r"\bas-[A-Za-z0-9]{20,}\b")),
    ("HF_TOKEN", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("TINYFISH_API_KEY", re.compile(r"\b(?:sk-tinyfish-|tf_)[A-Za-z0-9_-]{16,}")),
    ("OPENAI_KEY", re.compile(r"\bsk-(?!ant-|tinyfish-)[A-Za-z0-9_-]{20,}")),
    ("TAVILY_API_KEY", re.compile(r"\btvly-[A-Za-z0-9_-]{16,}")),
    ("DISCORD_WEBHOOK_URL", re.compile(r"https://(?:ptb\.|canary\.)?discord(?:app)?\.com/api/webhooks/\d+/[A-Za-z0-9_-]+")),
    # Agent sessions (capture/claude_code.py, capture/codex.py) quote command output and config, so
    # these shapes turn up there: GitHub tokens, JWTs (bearer and id tokens), AWS and Google API keys.
    ("GITHUB_TOKEN", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    # JWE (an encrypted token, five parts, the key part often empty: ChatGPT's session cookie is one)
    ("JWE", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{8,}")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    ("EXA_API_KEY", re.compile(r"(?i)(exa[_ -]?api[_ -]?key\W{0,5})([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")),
]


def _looks_secret(name: str, value: object) -> bool:
    if name in SKIP_NAMES or not isinstance(value, str):
        return False
    if len(value) < 16 or " " in value or "/" in value or "\\" in value:
        return False   # paths (D:/..., C:\...) and URLs are config, not keys; webhook URLs are caught by shape
    return True


def load_secrets(files: list[Path] | None = None) -> dict[str, str]:
    """{value: NAME} for every secret-looking value in .env and .claude/settings.local.json."""
    found: dict[str, str] = {}
    for path in files if files is not None else SECRET_FILES:
        if not path.exists():
            continue
        if path.suffix == ".json":
            try:
                items = json.loads(path.read_text(encoding="utf-8-sig")).get("env", {}).items()
            except (ValueError, AttributeError):
                items = []
        else:
            items = read_env_file(path).items()
        for name, value in items:
            if _looks_secret(name, value):
                found[value] = name
    return found


@lru_cache(maxsize=1)
def _default_secrets() -> tuple[tuple[str, str], ...]:
    return tuple(load_secrets().items())


def scrub(text: str, secrets: dict[str, str] | None = None) -> tuple[str, dict[str, int]]:
    """(clean text, {NAME: count}). secrets=None loads them from the repo's config files once."""
    if secrets is None:
        secrets = dict(_default_secrets())
    hits: dict[str, int] = {}
    for value, name in sorted(secrets.items(), key=lambda kv: -len(kv[0])):
        if value in text:
            hits[name] = hits.get(name, 0) + text.count(value)
            text = text.replace(value, f"[REDACTED:{name}]")
    for name, rx in SHAPES:
        def sub(m: re.Match, name: str = name) -> str:
            hits[name] = hits.get(name, 0) + 1
            return (m.group(1) if m.lastindex and m.lastindex >= 2 else "") + f"[REDACTED:{name}]"
        text = rx.sub(sub, text)
    return text, hits


def scrub_text(text: str, secrets: dict[str, str] | None = None) -> str:
    return scrub(text, secrets)[0]


def scrub_obj(obj, secrets: dict[str, str] | None = None):
    """Scrub every string inside a JSON-like structure (dicts, lists, strings). For raw JSON dumps."""
    if isinstance(obj, str):
        return scrub_text(obj, secrets)
    if isinstance(obj, dict):
        return {k: scrub_obj(v, secrets) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v, secrets) for v in obj]
    return obj
