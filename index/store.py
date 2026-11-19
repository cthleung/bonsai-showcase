"""The index file: one SQLite database with FTS5, a trigram FTS5 table and sqlite-vec vectors.

Tables:
    meta        key/value: the embedding model and dims the vectors came from
    notes       one row per indexed note: path, mtime, size, sha256, title, date, kind, source
    chunks      one row per chunk: path, turn, part, date, source, title, visibility, text, text hash
    chunks_fts  FTS5 over chunks.text (unicode61 with diacritics removed): BM25 for most text
    chunks_tri  FTS5 over chunks.text with the trigram tokenizer: Chinese, Korean and partial words
    chunks_vec  sqlite-vec vec0, float[1024] with cosine distance, keyed by chunk id

The FTS tables are ordinary FTS5 tables holding their own copy of the text, keyed by the chunk id
as rowid and kept in step by hand (inserted and deleted with the chunk). Vectors for a chunk whose text didn't change are reused
when a note is re-indexed, so a chat that gains one turn only embeds that turn.
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from index.chunking import Chunk
from index.settings import EMBED_DIMS

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS notes (
    path TEXT PRIMARY KEY, kind TEXT, title TEXT, date TEXT, source TEXT, visibility TEXT,
    mtime REAL, size INTEGER, sha256 TEXT, chunks INTEGER, indexed_at TEXT);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, path TEXT, kind TEXT, source TEXT, title TEXT, date TEXT,
    turn INTEGER, part INTEGER, visibility TEXT, text TEXT, text_hash TEXT);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(text_hash);
CREATE INDEX IF NOT EXISTS chunks_date ON chunks(date);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_tri USING fts5(text, tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY, embedding float[{EMBED_DIMS}] distance_metric=cosine);
"""


def pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def connect(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def get_meta(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))


def note_rows(db: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    return {r["path"]: r for r in db.execute("SELECT * FROM notes")}


def old_vectors(db: sqlite3.Connection, path: str) -> dict[str, bytes]:
    """{text hash: packed vector} for a note's current chunks, so unchanged chunks keep their vector."""
    rows = db.execute(
        "SELECT c.text_hash AS h, v.embedding AS e FROM chunks c JOIN chunks_vec v ON v.chunk_id = c.id "
        "WHERE c.path = ?", (path,))
    return {r["h"]: r["e"] for r in rows}


def delete_note(db: sqlite3.Connection, path: str) -> None:
    ids = [r["id"] for r in db.execute("SELECT id FROM chunks WHERE path=?", (path,))]
    for i in ids:
        db.execute("DELETE FROM chunks_fts WHERE rowid=?", (i,))
        db.execute("DELETE FROM chunks_tri WHERE rowid=?", (i,))
        db.execute("DELETE FROM chunks_vec WHERE chunk_id=?", (i,))
    db.execute("DELETE FROM chunks WHERE path=?", (path,))
    db.execute("DELETE FROM notes WHERE path=?", (path,))


def write_note(db: sqlite3.Connection, note: dict, chunks: list[Chunk], vectors: list[bytes]) -> None:
    """Replace one note's rows. `vectors[i]` is the packed vector for `chunks[i]`."""
    assert len(chunks) == len(vectors)
    delete_note(db, note["path"])
    db.execute(
        "INSERT INTO notes(path, kind, title, date, source, visibility, mtime, size, sha256, chunks, indexed_at) "
        "VALUES(:path, :kind, :title, :date, :source, :visibility, :mtime, :size, :sha256, :chunks, :indexed_at)",
        {**note, "chunks": len(chunks)})
    for c, v in zip(chunks, vectors):
        cur = db.execute(
            "INSERT INTO chunks(path, kind, source, title, date, turn, part, visibility, text, text_hash) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (c.path, c.kind, c.source, c.title, c.date, c.turn, c.part, c.visibility, c.text, c.text_hash))
        cid = cur.lastrowid
        db.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?, ?)", (cid, c.text))
        db.execute("INSERT INTO chunks_tri(rowid, text) VALUES(?, ?)", (cid, c.text))
        db.execute("INSERT INTO chunks_vec(chunk_id, embedding) VALUES(?, ?)", (cid, v))


def touch_note(db: sqlite3.Connection, path: str, mtime: float, size: int) -> None:
    db.execute("UPDATE notes SET mtime=?, size=? WHERE path=?", (mtime, size, path))


def counts(db: sqlite3.Connection) -> dict[str, int]:
    out = {"notes": db.execute("SELECT count(*) FROM notes").fetchone()[0],
           "chunks": db.execute("SELECT count(*) FROM chunks").fetchone()[0],
           "vectors": db.execute("SELECT count(*) FROM chunks_vec").fetchone()[0]}
    for r in db.execute("SELECT kind, count(*) AS n FROM notes GROUP BY kind"):
        out[f"notes_{r['kind']}"] = r["n"]
    return out
