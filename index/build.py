"""`python -m index build`: bring the index in step with the vault.

1. List the notes the index covers (index/chunking.py), with `4 Archive/` left out.
2. A note whose mtime and size match its row is skipped without being read. A changed mtime with
   the same sha256 only updates the row. Anything else is re-chunked.
3. Rows for notes that are gone or now excluded (deleted, moved to an excluded folder, or with
   frontmatter that keeps them out) are deleted and committed before any server starts. When the
   exclusion rules or the chunker changed since the last build, every note is re-checked.
4. Re-chunked notes reuse the vector of any chunk whose text didn't change, and only the rest go
   to the embedding server, in batches. The database commits after every batch, so a stopped
   build loses at most one batch.

The vectors record which model made them (`meta.embed_model`). A build with a different model
refuses to run until `--rebuild` starts the index from scratch, so two models' vectors never mix.
The vault is only read.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from index import chunking, store
from index.models import Embedder, LlamaEmbedder
from index.server import Servers
from index.settings import Settings

BATCH = 32          # chunks per embedding request


class BuildError(RuntimeError):
    pass


@dataclass
class Report:
    scanned: int = 0
    unchanged: int = 0
    touched: int = 0
    indexed: int = 0
    skipped: int = 0
    deleted: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    shortened: int = 0
    seconds: float = 0.0
    totals: dict = field(default_factory=dict)

    def lines(self) -> list[str]:
        return [
            f"notes scanned {self.scanned}: {self.indexed} indexed, {self.unchanged + self.touched} unchanged, "
            f"{self.skipped} skipped, {self.deleted} removed",
            f"chunks embedded {self.chunks_embedded}, reused {self.chunks_reused}, shortened to fit {self.shortened}",
            f"index now: {self.totals}",
            f"time {self.seconds:.1f}s",
        ]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plan(s: Settings, db) -> tuple[list[tuple[Path, str, str, float, int, str]], Report]:
    """(notes to re-chunk as (path, kind, text, mtime, size, sha256), report so far).

    Every exclusion is applied here, before any server work, and committed: rows for notes that
    are gone or now on EXCLUDED_PATHS (not in the scan), and for notes whose frontmatter now
    keeps them out (`excluded:`, a `type` on EXCLUDED_TYPES, or frontmatter that won't parse).
    So a build that later fails to start the embedding server, or fails mid-embed, has already
    removed them. When the rules changed since the last build, every note is re-read for this,
    even one whose file didn't change. When the chunker changed, every note is re-chunked."""
    rep = Report()
    rows = store.note_rows(db)
    recheck = bool(rows) and store.get_meta(db, "rules") != chunking.rules_id()
    rechunk = bool(rows) and store.get_meta(db, "chunker") != chunking.CHUNKER_VERSION
    seen: set[str] = set()
    todo = []
    for path, kind in chunking.note_files(s.vault):
        rep.scanned += 1
        rel = path.relative_to(s.vault).as_posix()
        seen.add(rel)
        st = path.stat()
        row = rows.get(rel)
        same_file = row is not None and row["mtime"] == st.st_mtime and row["size"] == st.st_size
        if same_file and not recheck and not rechunk:
            rep.unchanged += 1
            continue
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        if chunking.skip_reason(text):
            if row is not None:
                store.delete_note(db, rel)
                rep.deleted += 1
            rep.skipped += 1
            continue
        sha = _sha(data)
        if row is not None and row["sha256"] == sha and not rechunk:
            if same_file:
                rep.unchanged += 1
            else:
                store.touch_note(db, rel, st.st_mtime, st.st_size)
                rep.touched += 1
            continue
        todo.append((path, kind, text, st.st_mtime, st.st_size, sha))
    for rel in rows:
        if rel not in seen:
            store.delete_note(db, rel)
            rep.deleted += 1
    store.set_meta(db, "rules", chunking.rules_id())
    db.commit()
    return todo, rep


def _index(s: Settings, db, todo, rep: Report, embedder: Embedder, log) -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    pending: list[tuple[dict, list, list]] = []     # (note row, chunks, vectors or None)
    waiting = 0

    def flush():
        nonlocal pending, waiting
        need = [(i, j) for i, (_, chunks, vecs) in enumerate(pending) for j, v in enumerate(vecs) if v is None]
        for k in range(0, len(need), BATCH):
            part = need[k:k + BATCH]
            got = embedder.embed([pending[i][1][j].text for i, j in part])
            rep.shortened += getattr(embedder, "shortened", 0)
            for (i, j), v in zip(part, got):
                pending[i][2][j] = store.pack(v)
            rep.chunks_embedded += len(part)
        for note, chunks, vecs in pending:
            store.write_note(db, note, chunks, vecs)
        db.commit()
        pending, waiting = [], 0

    started = last_log = time.time()
    for n, (path, kind, text, mtime, size, sha) in enumerate(todo, 1):
        rel = path.relative_to(s.vault).as_posix()
        nc = chunking.note_chunks(s.vault, path, kind, text)
        if nc is None:                       # plan() already skips these; kept as a backstop
            store.delete_note(db, rel)
            db.commit()
            rep.skipped += 1
            continue
        old = store.old_vectors(db, rel)
        vecs = [old.get(c.text_hash) for c in nc.chunks]
        rep.chunks_reused += sum(v is not None for v in vecs)
        note = {"path": rel, "kind": nc.kind, "title": nc.title, "date": nc.date, "source": nc.source,
                "visibility": nc.visibility, "mtime": mtime, "size": size, "sha256": sha, "indexed_at": now}
        pending.append((note, nc.chunks, vecs))
        waiting += sum(v is None for v in vecs)
        rep.indexed += 1
        if waiting >= BATCH * 4:
            flush()
            if time.time() - last_log > 30:
                last_log = time.time()
                log(f"index: {n}/{len(todo)} notes, {rep.chunks_embedded} chunks embedded, "
                    f"{last_log - started:.0f}s")
    flush()


def build(s: Settings, embedder: Embedder | None = None, rebuild: bool = False, keep_server: bool = False,
          log=print) -> Report:
    t0 = time.time()
    if rebuild and s.db_path.exists():
        for suffix in ("", "-wal", "-shm"):
            Path(str(s.db_path) + suffix).unlink(missing_ok=True)
    db = store.connect(s.db_path)
    try:
        todo, rep = plan(s, db)          # exclusions are applied and committed first, whatever happens next
        have = store.get_meta(db, "embed_model")
        if have and have != s.embed_model_id:
            raise BuildError(f"the index was built with {have}, not {s.embed_model_id}; "
                             "run `python -m index build --rebuild` to start it again")
        if todo:
            store.set_meta(db, "embed_model", s.embed_model_id)
            db.commit()
            if embedder is not None:
                _index(s, db, todo, rep, embedder, log)
            else:
                with Servers(s, ["embed"], keep=keep_server, log=log) as sv:
                    if not sv.up["embed"]:
                        raise BuildError("the embedding server didn't start (see BONSAI_STATE/index/llama-embed.log)")
                    _index(s, db, todo, rep, LlamaEmbedder(s.embed_url), log)
        store.set_meta(db, "chunker", chunking.CHUNKER_VERSION)   # only once every note is chunked this way
        rep.totals = store.counts(db)
        rep.seconds = time.time() - t0
        store.set_meta(db, "last_build", dt.datetime.now().isoformat(timespec="seconds"))
        db.commit()
        return rep
    finally:
        db.close()
