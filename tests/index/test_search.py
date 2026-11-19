import json
import math

import pytest

from index import build, search, server, store
from index.models import ModelError, cut, query_text, reranker_works, trim_tokens
from tests.index.conftest import CHAT2_REL, CHAT_REL, MEMORY_REL, FakeReranker


@pytest.fixture
def db(settings, fake_embedder):
    build.build(settings, fake_embedder)
    conn = store.connect(settings.db_path)
    yield conn
    conn.close()


def _paths(res):
    return [h.path for h in res.hits]


def test_bm25_finds_the_nas_chat_and_folds_turns_to_one_hit(db):
    res = search.run(db, "synology nas", mode="bm25")
    assert _paths(res) == [CHAT_REL]
    assert res.hits[0].turn == 1 and res.score_kind == "bm25"
    assert "synology" in res.hits[0].snippet.lower()


def test_odd_punctuation_never_breaks_the_fts_query(db):
    for q in ['C++ AND "raid', "what's NOT (nas)*", "-", "OR OR", "nas: raid; near/2"]:
        search.run(db, q, mode="bm25")


def test_short_cjk_word_is_found_by_substring_and_long_one_by_trigram(db):
    assert _paths(search.run(db, "餃子", mode="bm25")) == [CHAT2_REL]
    assert _paths(search.run(db, "小籠包", mode="bm25")) == [CHAT2_REL]
    assert search.cjk_terms(["小籠包", "餃子", "nas"]) == (["小籠包"], ["餃子"])


def test_romanized_words_match_by_keyword(db):
    assert _paths(search.run(db, "arigatou domo", mode="bm25"))[0] == CHAT2_REL


def test_vector_mode_uses_the_embedder(db, fake_embedder):
    res = search.run(db, "watering the tomatoes in the garden", mode="vector", embedder=fake_embedder)
    assert res.hits[0].path == MEMORY_REL and res.score_kind == "cosine"


def test_vector_mode_without_an_embedder_is_an_error(db):
    with pytest.raises(ModelError):
        search.run(db, "nas", mode="vector")


def test_hybrid_without_embedder_falls_back_to_keywords_and_says_so(db):
    res = search.run(db, "synology nas", mode="hybrid")
    assert res.mode == "bm25" and any("keyword search only" in n for n in res.notes)


def test_hybrid_without_reranker_keeps_rrf_order(db, fake_embedder):
    res = search.run(db, "backing up photos from the nas", mode="hybrid", embedder=fake_embedder)
    assert res.score_kind == "rrf" and res.hits[0].path == CHAT_REL and res.hits[0].turn == 2
    assert len(set(_paths(res))) == len(res.hits)
    assert any("RRF" in n for n in res.notes)


def test_reranker_sets_the_final_order(db, fake_embedder):
    normal = search.run(db, "nas raid synology", mode="hybrid", embedder=fake_embedder, reranker=FakeReranker())
    flipped = search.run(db, "nas raid synology", mode="hybrid", embedder=fake_embedder,
                         reranker=FakeReranker(reverse=True))
    assert normal.score_kind == "rerank" and normal.hits[0].path == CHAT_REL
    assert flipped.hits[-1].path == CHAT_REL


def test_one_long_chat_cannot_crowd_out_other_notes(settings):
    from common.notes import render
    from tests.index.conftest import FakeEmbedder, chat_body, chat_meta, write
    turns = [("2025-05-05", f"nas question {i}", f"nas raid synology answer {i}") for i in range(120)]
    write(settings.vault, "3 Resources/chats/chatgpt/2025/big nas.md",
          render(chat_meta("big nas", "chatgpt", "big1", "2025-05-05"), chat_body("big nas", "ChatGPT", turns)))
    emb = FakeEmbedder()
    build.build(settings, emb)
    db = store.connect(settings.db_path)
    try:
        for mode in ("bm25", "vector", "hybrid"):
            res = search.run(db, "nas raid synology", mode=mode, embedder=emb, reranker=FakeReranker())
            assert CHAT_REL in _paths(res), mode
    finally:
        db.close()


def test_reranker_check_catches_a_broken_gguf():
    assert reranker_works(FakeReranker())[0]
    assert not reranker_works(FakeReranker(broken=True))[0]


def test_search_logs_every_query(settings, db):
    search.search(settings, "synology nas", mode="bm25")
    search.search(settings, "nothing matches zzqx", mode="bm25", k=3)
    lines = [json.loads(l) for l in settings.query_log.read_text(encoding="utf-8").splitlines()]
    assert [l["query"] for l in lines] == ["synology nas", "nothing matches zzqx"]
    assert lines[0]["mode"] == "bm25" and lines[0]["top"][0]["path"] == CHAT_REL
    assert lines[1]["top"] == [] and "ms" in lines[1] and "ts" in lines[1]


def test_search_without_an_index_says_to_build(settings):
    with pytest.raises(FileNotFoundError, match="index build"):
        search.search(settings, "x", mode="bm25")


def test_hybrid_search_with_no_server_falls_back_and_logs_it(settings, db):
    res = search.search(settings, "synology nas", mode="hybrid")   # llama-server path doesn't exist
    assert res.mode == "bm25" and res.hits[0].path == CHAT_REL
    line = json.loads(settings.query_log.read_text(encoding="utf-8").splitlines()[-1])
    assert line["mode"] == "hybrid" and line["mode_used"] == "bm25"


def test_cut_keeps_the_first_1024_and_renormalizes():
    v = [3.0, 4.0] + [0.0] * 1022 + [100.0] * 1536
    out = cut(v)
    assert len(out) == 1024 and math.isclose(out[0], 0.6) and math.isclose(sum(x * x for x in out), 1.0)
    with pytest.raises(ModelError):
        cut([1.0] * 512)


def test_query_gets_the_instruction_and_trim_respects_cjk():
    assert query_text("nas").startswith("Instruct: ") and query_text("nas").endswith("\nQuery: nas")
    assert 590 <= len(trim_tokens("信" * 5000, 600)) <= 600
    assert trim_tokens("short", 600) == "short"


def test_servers_bind_to_localhost_only(settings):
    for role in ("embed", "rerank"):
        cmd = server.command(settings, role)
        assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert "--embedding" in server.command(settings, "embed")
    assert "--reranking" in server.command(settings, "rerank")
