"""gateway/searching.py: the BM25 fallback, the time limit, and hits checked against the rules."""
import dataclasses
import threading
import time
from pathlib import Path

import pytest

from gateway import searching
from index import search as index_search
from index.models import ModelError
from index.server import ServerError
from tests.gateway.conftest import CHAT_REL, SECRET
from tests.index.conftest import write_chat

FAKE_SERVER = Path(__file__).resolve().parents[1] / "index" / "fake_llama_server.py"
QUIET = lambda *a, **k: None   # noqa: E731


def test_bm25_needs_no_models(built):
    out = searching.run(built, "synology nas", mode="bm25", log=QUIET)
    assert out["mode_used"] == "bm25" and [h["path"] for h in out["hits"]] == [CHAT_REL]
    hit = out["hits"][0]
    assert set(hit) == {"path", "title", "date", "turn", "source", "snippet", "score"}
    assert hit["turn"] == 1 and hit["title"] == "home server plans" and "synology" in hit["snippet"].lower()


def test_hybrid_without_a_llama_server_falls_back_to_bm25_and_says_so(built):
    t0 = time.time()
    out = searching.run(built, "synology nas", mode="hybrid", log=QUIET)
    assert time.time() - t0 < 5
    assert out["mode_asked"] == "hybrid" and out["mode_used"] == "bm25"
    assert any("keyword search only" in n for n in out["notes"])
    assert [h["path"] for h in out["hits"]] == [CHAT_REL]


def test_a_server_that_never_gets_healthy_is_given_up_on(built, monkeypatch):
    """The fake llama-server answers /health with 503 forever: after `wait` it's stopped, and the
    answer is BM25."""
    s = dataclasses.replace(built, llama_server=FAKE_SERVER)
    s.embed_model.write_bytes(b"fake")
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    t0 = time.time()
    out = searching.run(s, "synology nas", mode="hybrid", wait=1.5, log=QUIET)
    assert time.time() - t0 < 1.5 + searching.SEARCH_SLACK
    assert out["mode_used"] == "bm25" and any("keyword search only" in n for n in out["notes"])
    assert [h["path"] for h in out["hits"]] == [CHAT_REL]


def test_a_search_that_hangs_is_cut_off_at_the_time_limit(built, monkeypatch):
    release = threading.Event()
    real = index_search.search

    def slow(s, query, mode="hybrid", k=10, **kw):
        if mode != "bm25":
            release.wait(30)
        return real(s, query, mode, k, **kw)

    monkeypatch.setattr(index_search, "search", slow)
    monkeypatch.setattr(searching, "SEARCH_SLACK", 0.0)
    try:
        t0 = time.time()
        out = searching.run(built, "synology nas", mode="hybrid", wait=0.5, log=QUIET)
        assert time.time() - t0 < 5
        assert out["mode_used"] == "bm25" and any("didn't answer within" in n for n in out["notes"])
        # the first search is still stuck: the next one doesn't queue behind it
        t0 = time.time()
        out2 = searching.run(built, "synology nas", mode="hybrid", wait=0.5, log=QUIET)
        assert time.time() - t0 < 2
        assert any("still waiting on the models" in n for n in out2["notes"])
    finally:
        release.set()
    deadline = time.time() + 10
    while searching._busy.locked() and time.time() < deadline:
        time.sleep(0.05)
    assert not searching._busy.locked()


@pytest.mark.parametrize("err", [ModelError("the index was built with other.gguf@1024"),
                                 ServerError("another run holds servers.lock"),
                                 ConnectionRefusedError("refused")])
def test_model_failures_fall_back_to_bm25(built, monkeypatch, err):
    real = index_search.search

    def failing(s, query, mode="hybrid", k=10, **kw):
        if mode != "bm25":
            raise err
        return real(s, query, mode, k, **kw)

    monkeypatch.setattr(index_search, "search", failing)
    out = searching.run(built, "synology nas", mode="vector", log=QUIET)
    assert out["mode_used"] == "bm25" and any(str(err) in n for n in out["notes"])


def test_hits_for_notes_excluded_since_the_build_are_dropped(built):
    """The index is stale: the chat was moved out after the build. The hit never comes back."""
    write_chat(built.vault, excluded="person")
    out = searching.run(built, "synology nas", mode="bm25", log=QUIET)
    assert out["hits"] == [] and out["notes"] == []      # dropped without a word, so no oracle


def test_hits_for_notes_moved_into_an_excluded_folder_are_dropped(built):
    src = built.vault / CHAT_REL
    dest = built.vault / "0 Inbox" / "win" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dest)
    out = searching.run(built, "synology nas", mode="bm25", log=QUIET)
    assert out["hits"] == []


def test_the_off_limits_notes_never_reach_the_index_or_the_results(built):
    for mode in ("bm25", "hybrid"):
        assert searching.run(built, SECRET, mode=mode, log=QUIET)["hits"] == []


def test_k_is_clamped_and_bad_arguments_are_errors(built):
    assert len(searching.run(built, "nas", k=0, mode="bm25", log=QUIET)["hits"]) <= 1
    searching.run(built, "nas", k=10_000, mode="bm25", log=QUIET)
    with pytest.raises(ValueError):
        searching.run(built, "  ", log=QUIET)
    with pytest.raises(ValueError):
        searching.run(built, "nas", mode="grep", log=QUIET)


def test_no_index_is_a_clear_error(settings):
    with pytest.raises(FileNotFoundError, match="index build"):
        searching.run(settings, "nas", log=QUIET)


def test_searches_are_logged_like_the_cli(built):
    searching.run(built, "synology nas", mode="bm25", log=QUIET)
    assert "synology nas" in built.query_log.read_text(encoding="utf-8")
