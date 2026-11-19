"""Tests for common/: config, scrub, notes. All values are made up; fake keys are built at run time
so no key-shaped string sits in git."""
import os
from pathlib import Path

import pytest

from common import config, notes, scrub

FAKE = "x9Q2" * 8   # 32 characters, key-like


# --- config ---------------------------------------------------------------------------------

def write_env(tmp_path, text):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return p


def test_config_reads_env_file(tmp_path, monkeypatch):
    for k in ("BONSAI_VAULT", "BONSAI_STATE", "BONSAI_MACHINE", "BONSAI_REPO_VISIBILITY", "BONSAI_SPECSTORY_DIRS"):
        monkeypatch.delenv(k, raising=False)
    env = write_env(tmp_path, '# comment\nBONSAI_VAULT="/home/someone/My Vault"\n'
                              "BONSAI_STATE=/home/someone/bonsai-state\nBONSAI_MACHINE=WIN\n")
    cfg = config.load(env)
    assert cfg.vault == Path("/home/someone/My Vault")
    assert cfg.state == Path("/home/someone/bonsai-state")
    assert cfg.machine == "win" and cfg.inbox_name == "win"
    assert cfg.specstory_dirs == () and cfg.repo_visibilities == {}     # no defaults: both come from .env


def test_config_environment_wins(tmp_path, monkeypatch):
    env = write_env(tmp_path, "BONSAI_VAULT=/a\nBONSAI_STATE=/b\nBONSAI_MACHINE=win\n")
    monkeypatch.setenv("BONSAI_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("BONSAI_MACHINE", "mac")
    cfg = config.load(env)
    assert cfg.vault == tmp_path / "vault" and cfg.machine == "mac"


def test_config_errors(tmp_path, monkeypatch):
    for k in ("BONSAI_VAULT", "BONSAI_STATE", "BONSAI_MACHINE"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(config.ConfigError):
        config.load(write_env(tmp_path, "BONSAI_STATE=/b\n"))
    with pytest.raises(config.ConfigError):
        config.load(write_env(tmp_path, "BONSAI_VAULT=/a\nBONSAI_STATE=/b\nBONSAI_MACHINE=linux\n"))


def test_config_require(tmp_path, monkeypatch):
    monkeypatch.setenv("BONSAI_VAULT", str(tmp_path / "nope"))
    monkeypatch.setenv("BONSAI_STATE", str(tmp_path / "state"))
    with pytest.raises(config.ConfigError):
        config.load(tmp_path / "missing.env").require()
    (tmp_path / "nope").mkdir()
    config.load(tmp_path / "missing.env").require()
    assert (tmp_path / "state").is_dir()


# --- scrub ----------------------------------------------------------------------------------

@pytest.mark.parametrize("key,name", [
    ("sk-" + "ant-api03-" + FAKE, "ANTHROPIC_KEY"),
    ("sk-" + "proj-" + FAKE, "OPENAI_KEY"),
    ("sk-" + FAKE, "OPENAI_KEY"),
    ("tvly-" + FAKE, "TAVILY_API_KEY"),
    ("hf_" + FAKE, "HF_TOKEN"),
    ("sk-" + "tinyfish-" + FAKE, "TINYFISH_API_KEY"),
    ("tf_" + FAKE, "TINYFISH_API_KEY"),
    ("https://discord.com/api/" + "webhooks/123456789/" + FAKE, "DISCORD_WEBHOOK_URL"),
    ("ghp_" + "A1b2C3d4E5" * 4, "GITHUB_TOKEN"),
    ("github_pat_" + "A1b2C3d4E5_" * 5, "GITHUB_TOKEN"),
    ("eyJ" + "hbGciOiJIUzI1" + ".eyJ" + "zdWIiOiIxMjM0" + "." + "dozjgNryP4J3jVm", "JWT"),
    ("eyJ" + "hbGciOiJkaXIiLCJlbmMi" + ".." + "a1B2c3D4e5F6" + "." + "x" * 40 + "." + "T4g5H6j7K8l9", "JWE"),
    ("AKIA" + "IOSFODNN7EXAMPLE", "AWS_ACCESS_KEY"),
    ("AIza" + "Sy" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6q", "GOOGLE_API_KEY"),
])
def test_scrub_shapes(key, name):
    text, hits = scrub.scrub(f"my key is {key} ok", secrets={})
    assert key not in text
    assert text == f"my key is [REDACTED:{name}] ok"
    assert hits == {name: 1}


def test_scrub_exa_keeps_label():
    uuid = "0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b"
    text, hits = scrub.scrub(f"EXA_API_KEY={uuid}", secrets={})
    assert text == "EXA_API_KEY=[REDACTED:EXA_API_KEY]" and hits == {"EXA_API_KEY": 1}


def test_scrub_leaves_ordinary_text():
    text = "the sk-rope is fine, task-runner, hf_x, see https://example.com/a-b-c"
    assert scrub.scrub(text, secrets={}) == (text, {})


def test_load_secrets_by_value_skips_paths(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"BONSAI_VAULT=D:/Synced Files/Some One/My Vault\nBONSAI_STATE=~/bonsai-state\n"
                   f"BONSAI_MACHINE=win\nPLAIN_LONG_PATH=C:/a/really/long/path/without/spaces\n"
                   f"WEIRD_TOKEN={FAKE}\nSHORT=abc\n", encoding="utf-8")
    settings = tmp_path / "settings.local.json"
    settings.write_text('{"env": {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1", "OTHER_SECRET": "' + FAKE[::-1] + '"}}',
                        encoding="utf-8")
    secrets = scrub.load_secrets([env, settings])
    assert secrets == {FAKE: "WEIRD_TOKEN", FAKE[::-1]: "OTHER_SECRET"}
    text, hits = scrub.scrub(f"raw: ~/bonsai-state/raw/a.json token {FAKE}", secrets)
    assert text == "raw: ~/bonsai-state/raw/a.json token [REDACTED:WEIRD_TOKEN]"
    assert hits == {"WEIRD_TOKEN": 1}


def test_scrub_obj_nested():
    key = "tvly-" + FAKE
    out = scrub.scrub_obj({"a": [key, 3, {"b": key}], "n": None}, secrets={})
    assert out == {"a": ["[REDACTED:TAVILY_API_KEY]", 3, {"b": "[REDACTED:TAVILY_API_KEY]"}], "n": None}


# --- notes ----------------------------------------------------------------------------------

CHAT_META = {
    "title": "best ramen in sapporo",
    "source": "google-ai-mode",
    "chat_id": "AbCdEfGh1234567",
    "url": "https://share.google/aimode/AbCdEfGh1234567",
    "created": "2026-11-10",
    "updated": "2026-11-12",
    "captured": "2026-11-12",
    "model": "unknown",
    "project": "[[Food]]",
    "message_count": 6,
    "capture": "cdp-share-link",
    "raw": "~/bonsai-state/raw/google-ai-mode/AbCdEfGh1234567.json",
    "visibility": "private",
    "tags": ["chat"],
}


def test_note_round_trip(tmp_path):
    p = tmp_path / "n.md"
    body = "## 1. You (2026-11-10)\n\nwhere to eat\n\n### AI Mode\n\nTry somewhere.\n"
    assert notes.write_note(p, CHAT_META, body) is True
    text = p.read_text(encoding="utf-8")
    assert text.startswith("---\ntitle: best ramen in sapporo\n")
    assert "project: '[[Food]]'" in text or 'project: "[[Food]]"' in text
    assert "tags: [chat]\n" in text
    assert "created: 2026-11-10\n" in text
    meta, back = notes.read_note(p)
    assert meta == CHAT_META and list(meta) == list(CHAT_META)
    assert isinstance(meta["created"], str)
    assert back == body


def test_note_dates_and_hand_written_frontmatter(tmp_path):
    import datetime
    p = tmp_path / "n.md"
    p.write_bytes('\ufeff---\r\nname: a\r\ndate: 2026-10-07\r\nproject: "[[Food]]"\r\n---\r\nbody\r\n'.encode("utf-8"))
    meta, body = notes.read_note(p)
    assert meta == {"name": "a", "date": "2026-10-07", "project": "[[Food]]"} and body == "body\n"
    notes.write_note(p, {"date": datetime.date(2026, 11, 17)}, "x")
    assert notes.read_note(p) == ({"date": "2026-11-17"}, "x\n")


def test_scalar_only_frontmatter_is_block_style():
    text = notes.render({"name": "a", "type": "feedback", "visibility": "private"}, "x")
    assert text == "---\nname: a\ntype: feedback\nvisibility: private\n---\nx\n"
    text = notes.render({"projects": ["[[Bonsai]]"], "nested": [{"a": 1}], "tags": ["task", "x"]}, "")
    assert "projects: ['[[Bonsai]]']\n" in text and "tags: [task, x]\n" in text
    assert "nested:\n- a: 1\n" in text
    assert notes.parse(text)[0] == {"projects": ["[[Bonsai]]"], "nested": [{"a": 1}], "tags": ["task", "x"]}


def test_note_without_frontmatter(tmp_path):
    assert notes.parse("# just a note\n") == ({}, "# just a note\n")
    p = tmp_path / "plain.md"
    notes.write_note(p, {}, "# hi")
    assert p.read_text(encoding="utf-8") == "# hi\n"


def test_write_note_scrubs_and_skips_unchanged(tmp_path):
    p = tmp_path / "sub" / "n.md"
    key = "sk-" + "ant-" + FAKE
    assert notes.write_note(p, {"title": f"key {key}"}, f"pasted {key}\n")
    text = p.read_text(encoding="utf-8")
    assert key not in text and text.count("[REDACTED:ANTHROPIC_KEY]") == 2
    mtime = p.stat().st_mtime_ns
    assert notes.write_note(p, {"title": f"key {key}"}, f"pasted {key}\n") is False
    assert p.stat().st_mtime_ns == mtime
    assert not list(p.parent.glob(".*.tmp"))


def test_write_text_atomic_retries_locked_file(tmp_path, monkeypatch):
    p = tmp_path / "n.md"
    calls = {"n": 0}
    real = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked by the sync client")
        real(src, dst)

    monkeypatch.setattr(notes.os, "replace", flaky)
    notes.write_text_atomic(p, "hello", delay=0)
    assert p.read_text(encoding="utf-8") == "hello" and calls["n"] == 3

    monkeypatch.setattr(notes.os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError("stuck")))
    with pytest.raises(PermissionError):
        notes.write_text_atomic(tmp_path / "m.md", "x", retries=2, delay=0)
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("raw,want", [
    ("best ramen in sapporo", "best ramen in sapporo"),
    ('what is "C:\\x"? a/b | c*', "what is C x a b c"),
    ("#tag ^block [[link]]", "tag block link"),
    ("ends with dots...   ", "ends with dots"),
    ("CON", "_CON"),
    ("nul", "_nul"),
    ("aux.notes", "_aux.notes"),
    ("com1 ", "_com1"),
    ("lpt9.", "_lpt9"),
    ("console", "console"),
    ("tab\there\x7fdel", "tab here del"),
    ("???", "untitled"),
])
def test_safe_filename(raw, want):
    assert notes.safe_filename(raw) == want


@pytest.mark.parametrize("part,ok", [
    ("Bonsai", True), ("AI workflow", True), ("note-1.md", True),
    ("", False), (".", False), ("..", False), ("con", False), ("NUL.md", False), ("com1", False),
    ("trailing.", False), ("trailing ", False), (" leading", False),
    ("a|b", False), ("a\nb", False), ("a\x00b", False), ("a:b", False), ("[[x]]", False), ("a#b", False),
    ("a\x85b", False), ("a b", False), ("a b", False), ("a\x1cb", False),
])
def test_is_safe_part(part, ok):
    # One check for file and folder names that dream and capture can share.
    assert notes.is_safe_part(part) is ok


def test_safe_filename_caps_length_on_a_word():
    name = notes.safe_filename("word " * 50, max_len=23)
    assert name == "word word word word" and len(name) <= 23


def test_chat_note_path_and_find(tmp_path):
    # Round brackets, so `[[<stem>]]` doesn't end at the first `]]`.
    p = notes.chat_note_path(tmp_path, "google-ai-mode", "2026-11-10", "best ramen: sapporo?", "AbCdEfGh1234567")
    assert p == tmp_path / "3 Resources" / "chats" / "google-ai-mode" / "2026" / "2026-11-10 best ramen sapporo (AbCdEfGh).md"
    assert "]" not in p.name and "[" not in p.name
    assert notes.find_chat_note(tmp_path, "google-ai-mode", "AbCdEfGh1234567") is None
    notes.write_note(p, CHAT_META, "body")
    # a different chat sharing the first 8 characters isn't a match
    other = p.with_name("2026-11-11 other (AbCdEfGh).md")
    notes.write_note(other, {**CHAT_META, "chat_id": "AbCdEfGhZZZ"}, "body")
    assert notes.find_chat_note(tmp_path, "google-ai-mode", "AbCdEfGh1234567") == p
    assert notes.find_chat_note(tmp_path, "chatgpt", "AbCdEfGh1234567") is None


def test_find_chat_note_still_finds_an_old_bracket_name(tmp_path):
    # A note captured before the rename must not get a second copy under the new name.
    old = tmp_path / "3 Resources/chats/google-ai-mode/2026/2026-11-10 best ramen [AbCdEfGh].md"
    notes.write_note(old, CHAT_META, "body")
    assert notes.find_chat_note(tmp_path, "google-ai-mode", "AbCdEfGh1234567") == old


@pytest.mark.parametrize("chat_id,want", [
    ("AbCdEfGh1234567", "AbCdEfGh"),
    ("Ab/Cd\\Ef:Gh*12", "AbCdEfGh"),           # path and Windows-illegal characters dropped
    ("../../evil..x", "evilx"),                # can't climb out of the folder
    ("a b\nc]]d)e", "abcde"),                  # spaces, newlines and link brackets dropped
    ("ok_id-9", "ok_id-9"),
    ("!!!", "noid"),
])
def test_chat_id_is_cleaned_in_file_names(chat_id, want):
    # chat_id goes into the name only as [A-Za-z0-9_-].
    assert notes.chat_id8(chat_id) == want
    name = notes.chat_note_name("2026-11-10", "t", chat_id)
    assert name == f"2026-11-10 t ({want}).md"
    assert notes.is_safe_part(name)


@pytest.mark.parametrize("source,created", [
    ("../evil", "2026-11-10"), ("Google AI", "2026-11-10"), ("con\n", "2026-11-10"),
    ("chatgpt", "2026/11/10"), ("chatgpt", "../../x"),
])
def test_chat_note_path_rejects_bad_folder_parts(tmp_path, source, created):
    with pytest.raises(ValueError):
        notes.chat_note_path(tmp_path, source, created, "t", "AbCdEfGh")


def test_config_repo_visibility(tmp_path, monkeypatch):
    monkeypatch.delenv("BONSAI_REPO_VISIBILITY", raising=False)
    cfg = config.load(write_env(tmp_path, "BONSAI_VAULT=/a\nBONSAI_STATE=/b\n"
                                          "BONSAI_REPO_VISIBILITY=Novel=creative; work = career ;bad;x=everyone\n"))
    assert cfg.repo_visibilities == {"novel": "creative", "work": "career"}
    assert len(cfg.specstory_warnings) == 2 and all("BONSAI_REPO_VISIBILITY" in w for w in cfg.specstory_warnings)
    monkeypatch.setenv("BONSAI_REPO_VISIBILITY", "notes=public")      # the environment wins
    assert config.load(write_env(tmp_path, "BONSAI_VAULT=/a\nBONSAI_STATE=/b\nBONSAI_REPO_VISIBILITY=x=private\n")
                       ).repo_visibilities == {"notes": "public"}
