"""The capture log: one scrubbed line per event."""
from capture import status


def test_log_is_scrubbed(cfg):
    fake = "sk-ant-" + "y" * 30
    status.log(cfg, f"oops {fake}")
    text = (cfg.state / "capture.log").read_text(encoding="utf-8")
    assert fake not in text and "oops" in text
