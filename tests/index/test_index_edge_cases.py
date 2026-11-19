"""Index edge cases: exclusions, model changes, failures that fall back, and CJK search."""
import dataclasses

import pytest

from common.notes import render
from index import build, chunking, search, store
from index.models import LlamaReranker, ModelError
from tests.index.conftest import CHAT2_REL, CHAT_REL, FakeEmbedder, FakeReranker, write, write_chat

PROFILE_NOTE = {"name": "x", "description": "a profile line", "type": "user", "source": "agent",
                "date": "2026-11-19", "visibility": "private"}
NOTE_REL = "2 Areas/Profile/reflection.md"


def _indexed(settings) -> set[str]:
    db = store.connect(settings.db_path)
    try:
        return {r["path"] for r in db.execute("SELECT path FROM notes")}
    finally:
        db.close()


class RaisingEmbedder(FakeEmbedder):
    def embed(self, texts):
        raise ModelError("the embedding server went away")

    def embed_query(self, query):
        raise ModelError("the embedding server went away")


# --- 1. frontmatter exclusions apply without a file change, and before any server work ------

def test_a_new_rule_removes_an_unchanged_note(settings, monkeypatch):
    write(settings.vault, NOTE_REL, render({**PROFILE_NOTE, "type": "diary"}, "a profile line\n"))
    monkeypatch.setattr(chunking, "EXCLUDED_TYPES", ())       # an index built before the rule existed
    build.build(settings, FakeEmbedder())
    assert NOTE_REL in _indexed(settings)
    monkeypatch.undo()                                         # the rule arrives; the file never changed
    rep = build.build(settings, FakeEmbedder())
    assert NOTE_REL not in _indexed(settings) and rep.deleted == 1
    again = build.build(settings, FakeEmbedder())
    assert again.indexed == 0 and again.deleted == 0 and NOTE_REL not in _indexed(settings)


def test_exclusion_is_committed_even_when_embedding_fails(settings):
    build.build(settings, FakeEmbedder())
    write_chat(settings.vault, excluded="junk")                         # newly excluded
    write(settings.vault, NOTE_REL, render(PROFILE_NOTE, "a new note that needs embedding\n"))
    with pytest.raises(ModelError):
        build.build(settings, RaisingEmbedder())
    assert CHAT_REL not in _indexed(settings)


def test_exclusion_is_committed_even_when_the_server_wont_start(settings):
    build.build(settings, FakeEmbedder())
    write_chat(settings.vault, excluded="junk")
    write(settings.vault, NOTE_REL, render(PROFILE_NOTE, "a new note that needs embedding\n"))
    with pytest.raises(build.BuildError, match="didn't start"):
        build.build(settings, None, log=lambda m: None)           # settings point at no llama-server
    assert CHAT_REL not in _indexed(settings)


def test_exclusion_is_committed_even_when_the_model_changed(settings):
    build.build(settings, FakeEmbedder())
    write_chat(settings.vault, excluded="junk")
    other = dataclasses.replace(settings, embed_model=settings.state / "other.gguf")
    with pytest.raises(build.BuildError):
        build.build(other, FakeEmbedder())
    assert CHAT_REL not in _indexed(settings)


def test_a_chunker_change_rechunks_everything_and_reuses_vectors(settings, monkeypatch):
    build.build(settings, FakeEmbedder())
    monkeypatch.setattr(chunking, "CHUNKER_VERSION", "test-2")
    emb = FakeEmbedder()
    rep = build.build(settings, emb)
    assert rep.indexed == rep.scanned and emb.embedded == 0 and rep.chunks_reused == 6


def test_type_as_a_list_counts_as_diary():
    assert chunking.excluded_meta({"type": ["diary"]})
    assert chunking.excluded_meta({"type": ["note", "Diary"]})
    assert not chunking.excluded_meta({"type": ["user"]})
    assert not chunking.excluded_meta({"type": None})


# --- 2. search refuses vectors from another model -------------------------------------------

def test_search_refuses_vectors_from_another_model_but_bm25_works(settings):
    build.build(settings, FakeEmbedder())
    other = dataclasses.replace(settings, embed_model=settings.state / "other.gguf")
    for mode in ("vector", "hybrid"):
        with pytest.raises(ModelError, match="built with"):
            search.search(other, "synology nas", mode=mode, log=lambda m: None)
    assert search.search(other, "synology nas", mode="bm25").hits[0].path == CHAT_REL


# --- 4. model failures fall back and are logged ---------------------------------------------

@pytest.fixture
def db(settings):
    build.build(settings, FakeEmbedder())
    conn = store.connect(settings.db_path)
    yield conn
    conn.close()


def test_embed_failure_in_hybrid_falls_back_to_bm25(db):
    res = search.run(db, "synology nas", mode="hybrid", embedder=RaisingEmbedder())
    assert res.mode == "bm25" and res.hits[0].path == CHAT_REL
    assert any("embedding the query failed" in n for n in res.notes)


def test_embed_failure_in_vector_mode_is_an_error(db):
    with pytest.raises(ModelError):
        search.run(db, "synology nas", mode="vector", embedder=RaisingEmbedder())


class OddReranker:
    def __init__(self, reply):
        self.reply = reply

    def rerank(self, query, docs):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.mark.parametrize("reply", [ModelError("down"), KeyError("results"), ["x", "y"], [0.5]])
def test_rerank_failure_keeps_rrf_order(db, reply):
    plain = search.run(db, "synology nas", mode="hybrid", embedder=FakeEmbedder())
    res = search.run(db, "synology nas", mode="hybrid", embedder=FakeEmbedder(), reranker=OddReranker(reply))
    assert res.score_kind == "rrf" and [h.path for h in res.hits] == [h.path for h in plain.hits]
    assert any("reranker failed" in n for n in res.notes)


def test_malformed_rerank_replies_raise_model_error(monkeypatch):
    rr = LlamaReranker("http://127.0.0.1:1")
    for reply in ({}, {"results": [{"index": 5, "relevance_score": 1}]}, {"results": [{"index": 0}]},
                  {"results": [{"index": 0, "relevance_score": 0.5}]}):
        monkeypatch.setattr("index.models._post", lambda url, payload, timeout, r=reply: r)
        with pytest.raises(ModelError):
            rr.rerank("q", ["a", "b"])


def test_reranker_check_errors_are_caught_in_search(settings, monkeypatch):
    build.build(settings, FakeEmbedder())
    monkeypatch.setattr(search, "reranker_works", lambda rr: (_ for _ in ()).throw(ModelError("odd reply")))
    monkeypatch.setattr(search, "Servers", _FakeServers)
    monkeypatch.setattr(search, "LlamaEmbedder", lambda url: FakeEmbedder())
    s = dataclasses.replace(settings, rerank_model=settings.state / "r.gguf")
    res = search.search(s, "synology nas", mode="hybrid")
    assert res.score_kind == "rrf" and any("reranker check failed" in n for n in res.notes)


class _FakeServers:
    def __init__(self, s, roles, **kw):
        self.up = {r: True for r in roles}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def test_fts_errors_are_logged_not_swallowed(db, monkeypatch):
    notes = []
    assert search.bm25_list(db, "chunks_fts", '"unterminated', notes=notes) == []
    assert notes and "chunks_fts query failed" in notes[0]


def test_cjk_search_still_works_after_the_changes(db):
    assert [h.path for h in search.run(db, "餃子", mode="bm25").hits] == [CHAT2_REL]


# --- 5. localhost traffic never goes through a proxy ----------------------------------------

def test_the_local_opener_ignores_proxy_settings(monkeypatch):
    import urllib.request
    from index import localhttp
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9")
    # build_opener drops the default ProxyHandler (the one that reads HTTP_PROXY and the system
    # proxy) because an empty one was given, and the empty one adds no proxy method.
    assert not any(isinstance(h, urllib.request.ProxyHandler) for h in localhttp.OPENER.handlers)
    default = urllib.request.build_opener()
    assert any(isinstance(h, urllib.request.ProxyHandler) for h in default.handlers)
