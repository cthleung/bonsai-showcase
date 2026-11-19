"""python -m index <command>

    build [--rebuild] [--keep-server]
        Chunk the kept chats and memory notes and embed what changed since the last build.
        --rebuild deletes the index first (needed after changing the embedding model).
        --keep-server leaves the embedding server running afterwards.
    search "query" [--k 10] [--mode hybrid|vector|bm25] [--no-rerank] [--keep-server] [--json]
        Hybrid by default: BM25 plus vectors fused with RRF, then reranked. bm25 needs no server.
    serve [--stop | --status]
        Start both llama-server processes and leave them running (or stop the ones this package
        started, or show what's up). Searches are quicker with them already up.
    status
        Index counts, the model the vectors came from, and the last build time.

The index is BONSAI_STATE/index/index.sqlite. The vault is only read.
"""
from __future__ import annotations

import argparse
import json
import sys

from index import server, settings, store


def _print_result(res) -> None:
    print(f'"{res.query}"  mode {res.mode}, score {res.score_kind}, {res.ms} ms')
    for n in res.notes:
        print(f"  note: {n}")
    if not res.hits:
        print("  no results")
    for i, h in enumerate(res.hits, 1):
        turn = f" turn {h.turn}" if h.turn is not None else ""
        print(f"{i:>2}. {h.score:<8} {h.date}  {h.title}{turn}")
        print(f"    {h.path}")
        print(f"    {h.snippet}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m index", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rebuild", action="store_true")
    b.add_argument("--keep-server", action="store_true")
    q = sub.add_parser("search")
    q.add_argument("query")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("--mode", choices=("hybrid", "vector", "bm25"), default="hybrid")
    q.add_argument("--no-rerank", action="store_true")
    q.add_argument("--keep-server", action="store_true")
    q.add_argument("--json", action="store_true")
    sv = sub.add_parser("serve")
    sv.add_argument("--stop", action="store_true")
    sv.add_argument("--status", action="store_true")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    s = settings.load()

    if args.cmd == "build":
        from index.build import BuildError, build
        try:
            rep = build(s, rebuild=args.rebuild, keep_server=args.keep_server)
        except BuildError as e:
            print(f"index: {e}", file=sys.stderr)
            return 1
        print("\n".join(rep.lines()))
        return 0

    if args.cmd == "search":
        from index.models import ModelError
        from index.search import as_json, search
        try:
            res = search(s, args.query, args.mode, args.k, rerank=not args.no_rerank,
                         keep_server=args.keep_server)
        except (FileNotFoundError, ModelError) as e:
            print(f"index: {e}", file=sys.stderr)
            return 1
        print(as_json(res) if args.json else "", end="")
        if not args.json:
            _print_result(res)
        return 0

    if args.cmd == "serve":
        if args.status:
            print(json.dumps(server.status(s), indent=2))
            return 0
        if args.stop:
            for role in ("rerank", "embed"):
                print(f"{role}: {'stopped' if server.stop(s, role) else 'not ours or not running'}")
            return 0
        ok = server.start(s, "embed")
        print(f"embed: {'up' if ok else 'failed'} on {s.embed_url}")
        if s.rerank_model is not None:
            ok2 = server.start(s, "rerank")
            print(f"rerank: {'up' if ok2 else 'failed'} on {s.rerank_url}")
            ok = ok and ok2
        return 0 if ok else 1

    if args.cmd == "status":
        if not s.db_path.exists():
            print(f"no index at {s.db_path}")
            return 1
        db = store.connect(s.db_path)
        print(json.dumps({**store.counts(db), "embed_model": store.get_meta(db, "embed_model"),
                          "last_build": store.get_meta(db, "last_build"), "path": str(s.db_path)}, indent=2))
        db.close()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
