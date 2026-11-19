"""The command line: the guard, the run lock, and which commands exist."""
import pytest

from capture import __main__ as cli


def test_guard_logs_an_error_by_its_kind_only(cfg):
    def job():
        raise ValueError("a line quoting a note: my secret plans")
    assert cli.guarded(cfg, "codex", job) == 1
    text = (cfg.state / "capture.log").read_text(encoding="utf-8")
    assert "codex: stopped (ValueError)" in text and "secret plans" not in text


def test_guard_counts_skipped_and_ok(cfg):
    assert cli.guarded(cfg, "codex", lambda: cli.SKIPPED) == 2
    assert cli.guarded(cfg, "codex", lambda: None) == 0


def test_run_lock_is_exclusive(cfg):
    with cli.run_lock(cfg) as first:
        with cli.run_lock(cfg) as second:
            assert first and not second
    with cli.run_lock(cfg) as again:
        assert again


def test_hourly_with_nothing_to_read_is_skipped(cfg):
    assert cli.main(["hourly"]) == 2


def test_hourly_holds_off_while_another_run_has_the_lock(cfg):
    with cli.run_lock(cfg):
        assert cli.main(["hourly"]) == 2
    assert "another capture run holds the lock" in (cfg.state / "capture.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("cmd", ["chatgpt", "claude", "daily", "login", "aimode-link"])
def test_only_the_local_sources_are_offered(cfg, cmd):
    with pytest.raises(SystemExit):
        cli.main([cmd])


def test_status_prints_the_cursors(cfg, capsys):
    from capture.pull import Cursor
    Cursor(cfg, "chatgpt").mark("a", 1.0)
    assert cli.main(["status"]) == 0
    assert "cursor chatgpt: 1 chats" in capsys.readouterr().out
