"""`python -m index search "query" [--k 10] [--mode hybrid|vector|bm25]`.

- **bm25:** SQLite FTS5 over the chunk text. The question is cut into words, each one quoted and
  joined with OR, so punctuation, `C++` or a stray AND can't break the query. When the question has
  Chinese, Japanese or Korean in it, the trigram table is searched too, and any CJK word under 3
  characters (which neither table can match) is found with a plain substring scan instead.
- **vector:** the question gets the Qwen3 query instruction, is embedded, and sqlite-vec returns
  the nearest chunks by cosine distance.
- **hybrid (default):** the top 50 of each list are fused with reciprocal rank fusion (k=60), the
  windows of one turn are folded together, and the top 30 (at most 2 from one note) go to the
  reranker, which sets the final order.

Each list keeps at most 3 chunks from one note before it's cut to 50, because a chat with 100
turns on one topic would otherwise fill a list and leave 3 notes after folding. Without a reranker (none configured, or it fails the relevant-or-junk check) the RRF order
  stands, and the output says so.

Results are folded to one per note (its best chunk, with the turn number), so one long chat can't
fill the list. Every search is logged to BONSAI_STATE/index/queries.jsonl.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field

from index import store
from index.chunking import CJK_RX, has_cjk
from index.models import Embedder, LlamaEmbedder, LlamaReranker, ModelError, Reranker, reranker_works
from index.server import Servers
from index.settings import Settings

MODES = ("hybrid", "vector", "bm25")
LIST_DEPTH = 50        # how deep each list goes into the fusion
FETCH_DEPTH = 300      # how many chunks each list reads before the per-note cap
PER_NOTE = 3           # at most this many chunks of one note in a list, so one long chat can't fill it
RRF_K = 60
RERANK_TOP = 30
RERANK_PER_NOTE = 2
STOPWORDS = set("""a an and are as at be but by can did do does for from had has have how i i'd i'm i've
if in is it it's me my of on or so that the their them then there these they this to was we were what
when where which who why will with would you your about any did ever""".split())


@dataclass
class Hit:
    path: str
    title: str
    date: str
    turn: int | None
    source: str
    snippet: str
    score: float
    chunk_id: int = 0


@dataclass
class Result:
    query: str
    mode: str
    hits: list[Hit]
    score_kind: str                     # rerank | rrf | bm25 | cosine
    notes: list[str] = field(default_factory=list)   # fallbacks and anything else worth saying
    ms: int = 0


# --- the query ------------------------------------------------------------------------------

def terms(query: str) -> list[str]:
    words = re.findall(r"[\w][\w'+#.-]*", query.lower())
    out = []
    for w in words:
        w = w.strip("'.-")
        if w and w not in STOPWORDS and w not in out:
            out.append(w)
    return out


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def fts_query(ts: list[str]) -> str:
    return " OR ".join(_quote(t) for t in ts)


def cjk_terms(ts: list[str]) -> tuple[list[str], list[str]]:
    """(CJK runs of 3 or more characters for the trigram table, shorter ones for the substring scan)."""
    runs = []
    for t in ts:
        runs += re.findall(f"{CJK_RX.pattern}+", t)
    return [r for r in runs if len(r) >= 3], [r for r in runs if len(r) < 3]


# --- the lists ------------------------------------------------------------------------------

def cap_per_note(db: sqlite3.Connection, lst: list[tuple[int, float]], per_note: int = PER_NOTE,
                 depth: int = LIST_DEPTH) -> list[tuple[int, float]]:
    """Keep at most `per_note` chunks from any one note, then the top `depth`."""
    if not lst:
        return lst
    marks = ",".join("?" * len(lst))
    path = {r[0]: r[1] for r in db.execute(f"SELECT id, path FROM chunks WHERE id IN ({marks})", [c for c, _ in lst])}
    seen: dict[str, int] = {}
    out = []
    for cid, score in lst:
        p = path.get(cid)
        if p is None or seen.get(p, 0) >= per_note:
            continue
        seen[p] = seen.get(p, 0) + 1
        out.append((cid, score))
        if len(out) >= depth:
            break
    return out


def bm25_list(db: sqlite3.Connection, table: str, q: str, depth: int = FETCH_DEPTH,
              notes: list[str] | None = None) -> list[tuple[int, float]]:
    """An FTS5 list. A query FTS5 refuses gives an empty list, and the error goes into `notes`
    (so the output and queries.jsonl show it) instead of vanishing."""
    if not q:
        return []
    try:
        rows = db.execute(f"SELECT rowid, bm25({table}) AS s FROM {table} WHERE {table} MATCH ? "
                          f"ORDER BY s LIMIT ?", (q, depth)).fetchall()
    except sqlite3.OperationalError as e:
        if notes is not None:
            notes.append(f"{table} query failed ({e}): that keyword list is empty")
        return []
    return [(r[0], -r[1]) for r in rows]


def substring_list(db: sqlite3.Connection, term: str, depth: int = FETCH_DEPTH) -> list[tuple[int, float]]:
    rows = db.execute(
        "SELECT id, (length(text) - length(replace(text, ?, ''))) / ? AS n FROM chunks "
        "WHERE instr(text, ?) > 0 ORDER BY n DESC, date DESC LIMIT ?",
        (term, len(term), term, depth)).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def keyword_lists(db: sqlite3.Connection, query: str, notes: list[str] | None = None) -> list[list[tuple[int, float]]]:
    ts = terms(query)
    lists = [bm25_list(db, "chunks_fts", fts_query(ts), notes=notes)]
    if has_cjk(query):
        long_runs, short_runs = cjk_terms(ts)
        lists.append(bm25_list(db, "chunks_tri", fts_query(long_runs), notes=notes))
        for t in short_runs:
            lists.append(substring_list(db, t))
    return [cap_per_note(db, l) for l in lists if l]


def vector_list(db: sqlite3.Connection, qvec: list[float], depth: int = FETCH_DEPTH) -> list[tuple[int, float]]:
    rows = db.execute("SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
                      "ORDER BY distance", (store.pack(qvec), depth)).fetchall()
    return cap_per_note(db, [(r[0], 1.0 - r[1]) for r in rows])


def rrf(lists: list[list[tuple[int, float]]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for lst in lists:
        for rank, (cid, _) in enumerate(lst, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda t: -t[1])


# --- folding and snippets -------------------------------------------------------------------

def _rows(db: sqlite3.Connection, ids: list[int]) -> dict[int, sqlite3.Row]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {r["id"]: r for r in db.execute(f"SELECT * FROM chunks WHERE id IN ({marks})", ids)}


def fold(ranked: list[tuple[int, float]], rows: dict[int, sqlite3.Row], key) -> list[tuple[int, float]]:
    """Keep the first (best) chunk for each key, in order."""
    seen, out = set(), []
    for cid, score in ranked:
        r = rows.get(cid)
        if r is None:
            continue
        kv = key(r)
        if kv in seen:
            continue
        seen.add(kv)
        out.append((cid, score))
    return out


def snippet(text: str, ts: list[str], width: int = 260) -> str:
    body = text.split("\n", 1)[1] if "\n" in text else text
    flat = " ".join(body.split())
    low = flat.lower()
    pos = min((p for p in (low.find(t) for t in ts) if p >= 0), default=-1)
    if pos < 0:
        return flat[:width] + ("..." if len(flat) > width else "")
    start = max(0, pos - 80)
    out = flat[start:start + width]
    return ("..." if start else "") + out + ("..." if start + width < len(flat) else "")


# --- search ---------------------------------------------------------------------------------

def run(db: sqlite3.Connection, query: str, mode: str = "hybrid", k: int = 10,
        embedder: Embedder | None = None, reranker: Reranker | None = None) -> Result:
    """The search itself, with the models passed in (None means not available)."""
    t0 = time.time()
    notes: list[str] = []
    ts = terms(query)
    if mode in ("vector", "hybrid") and embedder is None:
        if mode == "vector":
            raise ModelError("vector search needs the embedding server, and it isn't available")
        notes.append("embedding server not available: keyword search only")
        mode_used = "bm25"
    else:
        mode_used = mode

    vec: list[tuple[int, float]] = []
    if mode_used in ("vector", "hybrid"):
        try:
            vec = vector_list(db, embedder.embed_query(query))
        except ModelError as e:
            if mode_used == "vector":
                raise
            notes.append(f"embedding the query failed ({e}): keyword search only")
            mode_used = "bm25"
    kw = keyword_lists(db, query, notes) if mode_used in ("bm25", "hybrid") else []

    if mode_used == "vector":
        ranked, kind = vec, "cosine"
    elif mode_used == "bm25":
        ranked, kind = (kw[0], "bm25") if len(kw) == 1 else (rrf(kw), "rrf")
    else:
        ranked, kind = rrf(kw + [vec]), "rrf"

    rows = _rows(db, [cid for cid, _ in ranked])
    ranked = fold(ranked, rows, lambda r: (r["path"], r["turn"]))   # windows of one turn

    if mode_used == "hybrid":
        if reranker is None:
            notes.append("no reranker: RRF order")
        else:
            top = cap_per_note(db, ranked, RERANK_PER_NOTE, RERANK_TOP)
            try:
                scores = [float(x) for x in reranker.rerank(query, [rows[cid]["text"] for cid, _ in top])]
                if len(scores) != len(top):
                    raise ModelError(f"{len(scores)} scores for {len(top)} documents")
                ranked = sorted(zip([c for c, _ in top], scores), key=lambda t: -t[1])
                kind = "rerank"
            except (ModelError, KeyError, TypeError, ValueError) as e:
                notes.append(f"reranker failed ({e}): RRF order")

    ranked = fold(ranked, rows, lambda r: r["path"])[:k]
    hits = []
    for cid, score in ranked:
        r = rows[cid]
        hits.append(Hit(r["path"], r["title"], r["date"], r["turn"], r["source"],
                        snippet(r["text"], ts), round(float(score), 4), cid))
    return Result(query, mode_used, hits, kind, notes, int((time.time() - t0) * 1000))


def log_query(s: Settings, res: Result, asked_mode: str) -> None:
    s.index_dir.mkdir(parents=True, exist_ok=True)
    line = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "query": res.query,
            "mode": asked_mode, "mode_used": res.mode, "score": res.score_kind, "notes": res.notes,
            "ms": res.ms, "top": [{"path": h.path, "turn": h.turn, "score": h.score} for h in res.hits]}
    with open(s.query_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def search(s: Settings, query: str, mode: str = "hybrid", k: int = 10, rerank: bool = True,
           keep_server: bool = False, log=print) -> Result:
    """Open the index, start the servers this mode needs (stopping them after unless they were
    already running or keep_server is set), search, and log the query."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not s.db_path.exists():
        raise FileNotFoundError(f"no index at {s.db_path}: run `python -m index build` first")
    db = store.connect(s.db_path)
    if mode in ("vector", "hybrid"):
        have = store.get_meta(db, "embed_model")
        if have and have != s.embed_model_id:
            db.close()
            raise ModelError(f"the index was built with {have}, but the configured model is {s.embed_model_id}; "
                             "vectors from two models can't be compared. Use --mode bm25, or run "
                             "`python -m index build --rebuild`")
    roles = []
    if mode in ("vector", "hybrid"):
        roles.append("embed")
    if mode == "hybrid" and rerank and s.rerank_model is not None:
        roles.append("rerank")
    try:
        with Servers(s, roles, keep=keep_server, log=log) as sv:
            embedder = LlamaEmbedder(s.embed_url) if sv.up.get("embed") else None
            reranker = None
            extra = []
            if "rerank" in roles:
                if sv.up.get("rerank"):
                    rr = LlamaReranker(s.rerank_url)
                    try:
                        works, good, junk = reranker_works(rr)
                        if works:
                            reranker = rr
                        else:
                            extra.append(f"reranker failed the relevant-or-junk check ({good:.3g} vs {junk:.3g}): RRF order")
                    except (ModelError, KeyError, TypeError, ValueError) as e:
                        extra.append(f"reranker check failed ({e}): RRF order")
                else:
                    extra.append("reranker server not available: RRF order")
            elif mode == "hybrid":
                extra.append("reranking off" if not rerank else "no reranker model configured")
            res = run(db, query, mode, k, embedder, reranker)
            res.notes = extra + [n for n in res.notes if not (extra and n.startswith("no reranker"))]
        log_query(s, res, mode)
        return res
    finally:
        db.close()


def as_json(res: Result) -> str:
    return json.dumps(asdict(res), ensure_ascii=False, indent=2)
