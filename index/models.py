"""Clients for the two llama-server endpoints, plus the Matryoshka cut.

    emb = LlamaEmbedder("http://127.0.0.1:8091")
    vecs = emb.embed(["chunk text", ...])          # each cut to 1024 dims and re-normalized
    qvec = emb.embed_query("NAS setup")             # gets the query instruction first
    rr = LlamaReranker("http://127.0.0.1:8092")
    scores = rr.rerank("NAS setup", ["doc 1", "doc 2"])   # one relevance score per doc, in order

Anything with the same methods works in their place (the tests use small fakes).
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Protocol

from index.chunking import est_tokens
from index.localhttp import urlopen
from index.settings import EMBED_DIMS, QUERY_INSTRUCT

RERANK_DOC_TOKENS = 600      # about this many tokens of each candidate go to the reranker


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[float]: ...


class ModelError(RuntimeError):
    pass


def cut(vec: list[float], dims: int = EMBED_DIMS) -> list[float]:
    """Keep the first `dims` values and scale back to length 1 (Qwen3-Embedding is trained for this)."""
    head = vec[:dims]
    if len(head) < dims:
        raise ModelError(f"embedding has {len(vec)} dims, the index needs {dims}")
    norm = math.sqrt(sum(x * x for x in head)) or 1.0
    return [x / norm for x in head]


def trim_tokens(text: str, tokens: int) -> str:
    """The start of `text`, about `tokens` long by the chunker's estimate (CJK counts as 1 a char)."""
    if est_tokens(text) <= tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if est_tokens(text[:mid]) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def query_text(query: str) -> str:
    return f"Instruct: {QUERY_INSTRUCT}\nQuery: {query}"


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise ModelError(f"{url}: HTTP {e.code}: {detail}") from None
    except (urllib.error.URLError, OSError) as e:
        raise ModelError(f"{url}: {e}") from None
    except ValueError as e:
        raise ModelError(f"{url}: reply isn't JSON ({e})") from None
    if not isinstance(data, dict):
        raise ModelError(f"{url}: reply isn't a JSON object")
    return data


class LlamaEmbedder:
    def __init__(self, url: str, dims: int = EMBED_DIMS, timeout: float = 300.0):
        self.url, self.dims, self.timeout = url.rstrip("/"), dims, timeout
        self.shortened = 0       # inputs cut shorter in the last embed() call to fit the server

    def _raw(self, texts: list[str]) -> list[list[float]]:
        data = _post(f"{self.url}/v1/embeddings", {"input": texts}, self.timeout)
        try:
            rows = sorted(data["data"], key=lambda d: int(d["index"]))
            vecs = [[float(x) for x in r["embedding"]] for r in rows]
        except (KeyError, TypeError, ValueError) as e:
            raise ModelError(f"{self.url}: odd embeddings reply ({e!r})") from None
        if len(vecs) != len(texts):
            raise ModelError(f"{self.url}: {len(vecs)} embeddings for {len(texts)} inputs")
        return vecs

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.shortened = 0
        try:
            return [cut(v, self.dims) for v in self._raw(texts)]
        except ModelError as e:
            if "HTTP 4" not in str(e):
                raise
        # The batch was refused (usually one input over the slot size, "exceeds the available
        # context size"): embed one at a time, cutting the end off any input that still fails.
        out = []
        for t in texts:
            while True:
                try:
                    out.append(cut(self._raw([t])[0], self.dims))
                    break
                except ModelError as e:
                    if "HTTP 4" not in str(e) or len(t) < 200:
                        raise
                    t = t[: int(len(t) * 0.7)]
                    self.shortened += 1
        return out

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query_text(query)])[0]


class LlamaReranker:
    def __init__(self, url: str, timeout: float = 120.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        payload = {"query": query, "documents": [trim_tokens(d, RERANK_DOC_TOKENS) for d in docs],
                   "top_n": len(docs)}
        data = _post(f"{self.url}/v1/rerank", payload, self.timeout)
        scores: list[float | None] = [None] * len(docs)
        try:
            for r in data["results"]:
                i = int(r["index"])
                if not 0 <= i < len(docs):
                    raise ValueError(f"index {i} out of range")
                scores[i] = float(r["relevance_score"])
        except (KeyError, TypeError, ValueError) as e:
            raise ModelError(f"{self.url}: odd rerank reply ({e!r})") from None
        if any(x is None or x != x for x in scores):        # a missing doc, or NaN
            raise ModelError(f"{self.url}: rerank reply doesn't score every document")
        return [float(x) for x in scores]


def reranker_works(rr: Reranker) -> tuple[bool, float, float]:
    """The broken-GGUF check from the research note: a relevant doc must score well above a junk one.
    Returns (works, relevant score, junk score)."""
    good, junk = rr.rerank("how do I set up RAID on my NAS",
                           ["To set up RAID on a Synology NAS, open Storage Manager, create a storage pool "
                            "and pick SHR or RAID 5 for the drives.",
                            "A simple recipe for banana bread with walnuts and brown sugar."])
    return good > junk + 0.1 and good > 0.05, good, junk
