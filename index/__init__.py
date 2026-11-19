"""Bonsai's semantic index: hybrid search over the kept chats and memory notes.

    python -m index build                 # chunk the vault, embed what changed, store in BONSAI_STATE/index
    python -m index search "NAS setup"    # BM25 plus vectors, fused with RRF, then reranked

The index lives in BONSAI_STATE/index/index.sqlite, never in the vault, and can be deleted and
rebuilt at any time. The vault is only ever read.
"""
