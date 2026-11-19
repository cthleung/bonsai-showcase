import dataclasses
import os
import shutil

import pytest

from index import build, store
from tests.index.conftest import CHAT_REL, FakeEmbedder, write_chat


def _counts(settings):
    db = store.connect(settings.db_path)
    try:
        return store.counts(db)
    finally:
        db.close()


def test_build_indexes_chats_and_memory_with_a_vector_per_chunk(settings, fake_embedder):
    rep = build.build(settings, fake_embedder)
    c = _counts(settings)
    assert c["notes"] == 4 and c["notes_chat"] == 2 and c["notes_memory"] == 2
    assert c["chunks"] == 6 == c["vectors"] == rep.chunks_embedded
    assert settings.db_path.is_file() and not (settings.vault / "index.sqlite").exists()


def test_second_build_embeds_nothing(settings, fake_embedder):
    build.build(settings, fake_embedder)
    again = FakeEmbedder()
    rep = build.build(settings, again)
    assert again.embedded == 0 and rep.unchanged == 4 and rep.indexed == 0


def test_new_turn_embeds_only_that_turn(settings, fake_embedder):
    build.build(settings, fake_embedder)
    write_chat(settings.vault, extra_turn=True)
    again = FakeEmbedder()
    rep = build.build(settings, again)
    assert rep.indexed == 1 and again.embedded == 1 and rep.chunks_reused == 2
    assert "which ups" in again.calls[0][0]
    c = _counts(settings)
    assert c["chunks"] == c["vectors"] == 7


def test_mtime_change_alone_reads_but_does_not_embed(settings, fake_embedder):
    build.build(settings, fake_embedder)
    p = settings.vault / CHAT_REL
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 100))
    again = FakeEmbedder()
    rep = build.build(settings, again)
    assert again.embedded == 0 and rep.touched == 1


def test_deleted_and_archived_notes_leave_the_index(settings, fake_embedder):
    build.build(settings, fake_embedder)
    src = settings.vault / CHAT_REL
    dest = settings.vault / "4 Archive/chats-excluded/junk" / src.name
    shutil.move(src, dest)
    (settings.vault / "2 Areas/Profile/hometown.md").unlink()
    rep = build.build(settings, FakeEmbedder())
    assert rep.deleted == 2
    c = _counts(settings)
    assert c["notes"] == 2 and c["chunks"] == c["vectors"] == 3
    db = store.connect(settings.db_path)
    assert db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 3
    assert db.execute("SELECT count(*) FROM chunks_tri").fetchone()[0] == 3
    db.close()


def test_chat_marked_excluded_in_place_leaves_the_index(settings, fake_embedder):
    build.build(settings, fake_embedder)
    write_chat(settings.vault, excluded="junk")
    rep = build.build(settings, FakeEmbedder())
    assert rep.skipped == 1
    assert _counts(settings)["notes_chat"] == 1


def test_a_different_model_refuses_until_rebuild(settings, fake_embedder):
    build.build(settings, fake_embedder)
    other = dataclasses.replace(settings, embed_model=settings.state / "other-model.gguf")
    write_chat(other.vault, extra_turn=True)
    with pytest.raises(build.BuildError, match="--rebuild"):
        build.build(other, FakeEmbedder())
    rep = build.build(other, FakeEmbedder(), rebuild=True)
    assert rep.chunks_embedded == 7
    db = store.connect(other.db_path)
    assert store.get_meta(db, "embed_model") == "other-model.gguf@1024"
    db.close()


def test_build_never_writes_to_the_vault(settings, fake_embedder):
    before = {p: p.stat().st_mtime for p in settings.vault.rglob("*") if p.is_file()}
    build.build(settings, fake_embedder)
    after = {p: p.stat().st_mtime for p in settings.vault.rglob("*") if p.is_file()}
    assert before == after
