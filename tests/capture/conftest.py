import pytest

from common import config


@pytest.fixture(autouse=True)
def no_real_sessions(tmp_path, monkeypatch):
    """No capture test can read the real agent sessions: SpecStory history and Codex transcripts
    point at folders that don't exist."""
    missing = tmp_path / "no-sessions-here"
    monkeypatch.setenv("BONSAI_SPECSTORY_DIRS", str(missing / "specstory" / "history"))
    monkeypatch.setenv("BONSAI_CODEX_SESSIONS", str(missing / "sessions"))
    for key in ("BONSAI_SPECSTORY_AGENTS", "BONSAI_REPO_VISIBILITY", "CODEX_HOME"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A config pointing at a temp vault and state folder. Never the real vault."""
    vault, state = tmp_path / "vault", tmp_path / "state"
    vault.mkdir()
    monkeypatch.setenv("BONSAI_VAULT", str(vault))
    monkeypatch.setenv("BONSAI_STATE", str(state))
    monkeypatch.setenv("BONSAI_MACHINE", "win")
    return config.load().require()
