"""The MCP server: three read-only tools over stdio, for your own trusted agents.

    python -m gateway [--env-file PATH]

    search(query, k=10, mode="hybrid")   the index's hybrid search (gateway/searching.py)
    read_note(path, start=0)             one vault note's text (gateway/access.py)
    recent(days=2, limit=20)             chats and memory notes changed lately, newest first

**Settings come from one file only:** the `.env` next to this code, or `--env-file`. Environment
variables are ignored on purpose, because Claude Code starts the server with the environment of
whatever project it's in, and that must never repoint the vault, the state folder or the
llama-server (index/settings.py `environ=False`). The file is read on every call, so a change needs
no restart. BONSAI_GATEWAY_WAIT in it sets how long each llama-server gets to start (seconds,
default 20).

Every call is logged as one strict JSON line in BONSAI_STATE/gateway/calls.jsonl: the tool, its
arguments, how many results, the time in ms, and the error (for a refusal, the real reason, which
the client never sees). Nothing but the protocol goes to stdout; messages go to stderr.

The tools run on worker threads that are given up on when the client goes away, so the server
exits as soon as stdin closes, releasing its llama servers on the way out.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from common import config
from common.config import ConfigError
from gateway import access, searching
from index import settings as index_settings

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                            open_world_hint=False)

INSTRUCTIONS = """Bonsai is the user's personal store: captured AI chats (ChatGPT, Claude Code, \
Copilot, Codex) and memory and profile notes, in an Obsidian vault. Use `search` for \
"what have I said about X" questions, `read_note` to open a hit, and `recent` for what changed in the \
last few days. Everything is read-only. Some folders are off limits and are refused on purpose."""

ENV_FILE: Path = config.ENV_FILE    # set by main() from --env-file
_last_settings = None               # the settings of the last call, for releasing servers on exit


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _settings():
    global _last_settings
    try:
        s = index_settings.load(ENV_FILE, environ=False)
    except (ConfigError, ValueError) as e:
        raise ToolError(f"bonsai isn't set up on this computer: {e}") from None
    _last_settings = s
    return s


def _wait() -> float:
    raw = config.read_env_file(ENV_FILE).get("BONSAI_GATEWAY_WAIT", "").strip()
    try:
        w = float(raw) if raw else searching.DEFAULT_WAIT
    except ValueError:
        return searching.DEFAULT_WAIT
    return min(max(0.5, w), 600.0) if math.isfinite(w) else searching.DEFAULT_WAIT


def _finite(name: str, value: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _plain(obj):
    """The arguments as strict JSON can hold them: a NaN or infinity becomes its name as a string."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return repr(obj)


def _log(s, tool: str, args: dict, results: int | None, ms: int, error: str | None) -> None:
    line = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "tool": tool, "args": _plain(args),
            "results": results, "ms": ms}
    if error:
        line["error"] = error
    try:
        folder = s.state / "gateway"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, ValueError) as e:
        _stderr(f"gateway: couldn't write the call log: {e}")


def _call(tool: str, args: dict, fn, count, refuse_all: bool = False) -> Any:
    """Run one tool with its settings, log it, and turn expected failures into tool errors. With
    `refuse_all` (read_note) every failure, whatever it was, reaches the client as the one fixed
    refusal; the real reason goes to the call log or stderr."""
    if refuse_all:
        try:
            return _call(tool, args, fn, count)
        except ToolError as e:
            if str(e) != access.REFUSED:
                _stderr(f"gateway: {tool} refused after: {e}")
            raise ToolError(access.REFUSED) from None
    t0 = time.time()
    s = None

    def done(results, error=None):
        if s is not None:
            _log(s, tool, args, results, int((time.time() - t0) * 1000), error)

    try:
        s = _settings()
        out = fn(s)
        done(count(out))
        return out
    except access.Refused as e:
        done(0, f"refused: {e.reason}")
        raise ToolError(str(e)) from None
    except (ValueError, FileNotFoundError) as e:
        done(0, str(e))
        raise ToolError(str(e)) from None
    except ToolError as e:
        done(0, str(e))
        raise
    except Exception as e:
        done(0, f"{type(e).__name__}: {e}")
        _stderr(f"gateway: {tool} failed: {type(e).__name__}: {e}")
        raise ToolError(f"{tool} failed ({type(e).__name__})") from None


async def _off_thread(fn):
    """Run a blocking tool on a worker thread that's abandoned if the client goes away, so a search
    waiting on the models never keeps the server alive after stdin closes."""
    return await anyio.to_thread.run_sync(fn, abandon_on_cancel=True)


def build() -> MCPServer:
    app = MCPServer("bonsai", instructions=INSTRUCTIONS, log_level="WARNING")

    @app.tool(annotations=READ_ONLY, structured_output=False)
    async def search(query: str, k: int = 10, mode: str = "hybrid") -> dict:
        """Search the user's captured chats and memory notes. `mode` is hybrid (keywords plus
        meaning, reranked: the default), vector, or bm25 (keywords only, instant). When the local
        models aren't available in time, the search falls back to keywords and `notes` says so.
        Each hit has path, title, date, turn (the chat turn, or null for a memory note), source,
        snippet and score. Open a hit with read_note(path)."""
        args = {"query": query, "k": k, "mode": mode}
        wait = _wait()
        return await _off_thread(functools.partial(
            _call, "search", args, lambda s: searching.run(s, query, k, mode, wait=wait, log=_stderr),
            lambda out: len(out["hits"])))

    @app.tool(annotations=READ_ONLY, structured_output=False)
    async def read_note(path: str, start: int = 0) -> dict:
        """The text of one vault note, frontmatter included, by its vault-relative path (as search
        and recent give it). Long notes come in pieces of about 8,000 tokens: when `more` is true,
        call again with start set to `end`. Off-limits folders and notes are refused."""
        args = {"path": path, "start": start}
        return await _off_thread(functools.partial(
            _call, "read_note", args, lambda s: access.read_note(s.vault, path, start), lambda out: 1,
            refuse_all=True))

    @app.tool(annotations=READ_ONLY, structured_output=False)
    async def recent(days: float = 2, limit: int = 20) -> dict:
        """Chats and memory notes changed in the last `days` days (up to 365), newest first, at most
        `limit` (up to 100). Each has path, title, kind (chat or memory), source, date and modified
        (the file's local change time)."""
        args = {"days": days, "limit": limit}

        def fn(s):
            d = min(max(_finite("days", days), 0.0), 365.0)
            n = max(1, min(int(limit), 100))
            return {"days": d, "notes": [asdict(r) for r in access.recent(s.vault, d, n)]}
        return await _off_thread(functools.partial(_call, "recent", args, fn, lambda out: len(out["notes"])))

    return app


def main(argv: list[str] | None = None) -> int:
    global ENV_FILE
    ap = argparse.ArgumentParser(prog="python -m gateway", description="Bonsai's read-only MCP server (stdio).")
    ap.add_argument("--env-file", type=Path, default=config.ENV_FILE,
                    help="the settings file, an absolute path (default: the .env next to this code); "
                         "environment variables are ignored, and so are paths with ~ or relative ones in it")
    args = ap.parse_args(argv)
    if not args.env_file.is_absolute():       # a relative one would be read from the project's folder
        _stderr(f"gateway: --env-file must be an absolute path (got {args.env_file})")
        return 2
    ENV_FILE = args.env_file.resolve()
    code = 0
    try:
        build().run("stdio")
    except BaseException as e:     # noqa: BLE001  said on stderr; the exit code carries it
        if not isinstance(e, KeyboardInterrupt):
            _stderr(f"gateway: stopped: {type(e).__name__}: {e}")
            code = 1
    finally:
        searching.shutdown(_last_settings)
        sys.stderr.flush()
        os._exit(code)       # never wait on a search thread still stuck on the models
    return code
