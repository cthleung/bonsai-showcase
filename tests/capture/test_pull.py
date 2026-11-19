"""What the capture sources share: API times, the cursor, and error kinds."""
import datetime as dt

from capture import pull
from capture.pull import Cursor


def test_epoch_and_local_date():
    assert pull.epoch(1700000000) == 1700000000.0
    assert pull.epoch("2026-11-12T12:00:00Z") == dt.datetime(2026, 11, 12, 12, tzinfo=dt.timezone.utc).timestamp()
    assert pull.epoch("2026-11-12T12:00:00.123456+00:00") is not None
    assert pull.epoch("nope") is None and pull.epoch(None) is None
    assert pull.local_date(dt.datetime(2026, 11, 12, 12).timestamp()) == "2026-11-12"


def test_cursor_round_trip(cfg):
    c = Cursor(cfg, "chatgpt")
    assert c.first_run
    c.mark("a", 100.0)
    c.mark("b", 50.0)
    c2 = Cursor(cfg, "chatgpt")
    assert c2.watermark == 100.0 and c2.chats == {"a": 100.0, "b": 50.0} and not c2.first_run
    assert not c2.is_new("a", 100.0) and c2.is_new("a", 101.0) and c2.is_new("z", 1.0)


def test_a_chat_is_given_up_on_after_max_tries(cfg):
    c = Cursor(cfg, "chatgpt")
    c.mark("new", 200.0)
    assert c.fail("old", 100.0) == (1, False) and c.floor() == 100.0
    c.hold("old", 100.0)
    assert c.failures["old"]["tries"] == 1
    assert c.fail("old", 100.0) == (2, False)
    assert c.fail("old", 100.0) == (pull.MAX_TRIES, True)
    assert c.failures == {} and "old" in c.gave_up and "old" in c.chats and c.floor() == 200.0


def test_error_kind_never_quotes_the_text():
    class E(RuntimeError):
        status = 429
    assert pull.error_kind(E("a body with a secret in it")) == "E HTTP 429"
    assert pull.error_kind(ValueError("x")) == "ValueError"
