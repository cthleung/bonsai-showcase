"""The search tool: the index's hybrid search with a time limit, falling back to BM25.

- **bm25** runs straight away: it's SQLite only and needs no model.
- **hybrid** and **vector** run `index.search.search()` in a worker thread. Each llama-server gets
  `wait` seconds to come up (index/server.py stops one that misses it), and the whole search gets
  `wait + SEARCH_SLACK` seconds. When that runs out, or the models can't be used at all (no
  llama-server, the index built with another model, a lock another run is holding), the answer is
  a BM25 search and `notes` says why. A worker still running when the limit hits finishes in the
  background, and the next search goes straight to BM25 until it's done, so calls never pile up.
  A server the gateway starts is tied to the gateway's process (index/procs.py tie_to_me, one
  Windows job object per server), so it dies with the gateway however the gateway ends, unless the
  gateway has let go of it while another run still uses it (index/server.py release() unties it).
- Every hit goes through gateway/access.py first, so a note excluded, moved or deleted since the
  last index build never comes back. A hit whose file changed since the build (its sha256
  differs from the index row's) gets its title, date and snippet from the file as it is now, from the
  same turn; when that turn is gone the hit is dropped. Dropped hits are never counted or
  mentioned, so the answer can't tell anyone what an off-limits note holds.

Each search is logged to BONSAI_STATE/index/queries.jsonl like the CLI's, but only with the hits
the client gets.
"""
from __future__ import annotations

import dataclasses
import hashlib
import sqlite3
import sys
import threading
from pathlib import Path

from gateway import access
from index import chunking, procs, server
from index import search as index_search
from index.models import ModelError
from index.server import ServerError
from index.settings import Settings

DEFAULT_WAIT = 20.0      # seconds each llama-server gets to start (a cold start takes about 2s each)
SEARCH_SLACK = 15.0      # on top of that, for the search and the rerank themselves
MAX_K = 50
EXTRA = 10               # hits asked for beyond k, to cover any the access check drops

_busy = threading.Lock()   # held while a model search runs; a timed-out one keeps it until it's done


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _bm25(s: Settings, query: str, k: int, log) -> index_search.Result:
    return index_search.search(s, query, "bm25", k, log=log, log_queries=False)


def _rows(s: Settings, ids: list[int]) -> dict[int, tuple]:
    """{chunk id: (note sha256, turn, part)} from the index, opened read-only."""
    if not ids:
        return {}
    db = sqlite3.connect(f"{s.db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(ids))
        return {r[0]: r[1:] for r in db.execute(
            f"SELECT c.id, n.sha256, c.turn, c.part FROM chunks c JOIN notes n ON n.path = c.path "
            f"WHERE c.id IN ({marks})", ids)}
    finally:
        db.close()


def _current(s: Settings, hit: index_search.Hit, row: tuple | None, ts: list[str]) -> index_search.Hit | None:
    """The hit as the client may see it, or None. Only notes access.load() serves pass, and a note
    whose content changed since the build (its sha256 differs from the row's: a re-hash of the
    text just read, so an edit that keeps the size and mtime is caught too) is re-read for its
    title, date and snippet."""
    try:
        rel, text, _ = access.load(s.vault, hit.path)
    except access.Refused:
        return None
    if row is None:
        return None
    sha, turn, part = row
    path = Path(s.vault) / rel
    if hashlib.sha256(text.encode("utf-8")).hexdigest() == sha:    # load() decoded strictly, so same bytes
        return hit
    kind = "chat" if _under(rel, chunking.CHATS_DIR) else "memory"
    nc = chunking.note_chunks(Path(s.vault), path, kind, text)
    if nc is None:
        return None
    same = [c for c in nc.chunks if c.turn == turn]
    chunk = next((c for c in same if c.part == part), same[0] if same else None)
    if chunk is None:
        return None
    return dataclasses.replace(hit, path=rel, title=nc.title, date=chunk.date,
                               snippet=index_search.snippet(chunk.text, ts))


def _under(rel: str, folder) -> bool:
    return rel.lower().startswith(Path(folder).as_posix().lower() + "/")


def run(s: Settings, query: str, k: int = 10, mode: str = "hybrid", wait: float = DEFAULT_WAIT,
        log=_stderr) -> dict:
    query = str(query or "").strip()
    if not query:
        raise ValueError("the query is empty")
    if mode not in index_search.MODES:
        raise ValueError(f"mode must be one of {', '.join(index_search.MODES)}")
    k = max(1, min(int(k), MAX_K))
    if not s.db_path.exists():
        raise FileNotFoundError("there's no search index on this computer yet: run `python -m index build`")
    ask = k + EXTRA
    notes: list[str] = []

    if mode == "bm25":
        res = _bm25(s, query, ask, log)
    elif not _busy.acquire(blocking=False):
        notes.append("an earlier search is still waiting on the models: keyword search only")
        res = _bm25(s, query, ask, log)
    else:
        box: dict = {}

        def work():
            try:
                box["res"] = index_search.search(s, query, mode, ask, log=log, wait=wait, tie=True,
                                                 log_queries=False)
            except BaseException as e:          # handed to the caller below
                box["err"] = e
            finally:
                _busy.release()

        t = threading.Thread(target=work, name="bonsai-search", daemon=True)
        t.start()
        t.join(wait + SEARCH_SLACK)
        if t.is_alive():
            notes.append(f"the search models didn't answer within {wait + SEARCH_SLACK:.0f}s: keyword search only")
            res = _bm25(s, query, ask, log)
        elif "err" in box:
            err = box["err"]
            if not isinstance(err, (ModelError, ServerError, OSError)) or isinstance(err, FileNotFoundError):
                raise err
            notes.append(f"the search models couldn't be used ({err}): keyword search only")
            res = _bm25(s, query, ask, log)
        else:
            res = box["res"]

    rows = _rows(s, [h.chunk_id for h in res.hits])
    ts = index_search.terms(query)
    hits = [h for h in (_current(s, h, rows.get(h.chunk_id), ts) for h in res.hits) if h is not None][:k]
    shown = dataclasses.replace(res, hits=hits, notes=notes + list(res.notes))
    index_search.log_query(s, shown, mode)
    return {
        "query": query,
        "mode_asked": mode,
        "mode_used": shown.mode,
        "score_kind": shown.score_kind,
        "notes": shown.notes,
        "hits": [{f: getattr(h, f) for f in ("path", "title", "date", "turn", "source", "snippet", "score")}
                 for h in hits],
    }


def shutdown(s: Settings | None) -> None:
    """Release this process's hold on the llama servers (the gateway is exiting). A server this
    process started stops when nobody else is using it, even mid-start. One another run is still
    using (an index build, say) is untied by release() as this process leaves it, so this exit
    doesn't take it along. When a release fails, that server's tie stays: the job object is then
    the only cleanup left."""
    if s is None:
        return
    me = procs.me()
    for role in server.ROLES:
        try:
            server.release(s, role, me)
        except Exception as e:   # noqa: BLE001  exiting anyway; say why and carry on
            _stderr(f"gateway: couldn't release the {role} server ({type(e).__name__}: {e})")
