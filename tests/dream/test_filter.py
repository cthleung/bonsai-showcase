"""dream/filter.py against made-up chat notes and a fake model. No real claude call, no real vault."""
import datetime as _dt
import re

import pytest

from capture.chatnote import Chat, Turn, write_chat
from common.notes import render
from dream import filter as filt
from dream import guards, propose
from tests.dream.conftest import write

DAY = "2026-11-19"
NOW = _dt.datetime(2026, 11, 19, 10, 0)
QUIET = lambda m: None

# title -> (score, whose, reason)
SCORES = {
    "lora rank sweep": (8, "user", "their own training project"),
    "my editor setup": (6, "user", "how they like their tools"),
    "capital of peru": (1, "user", "a trivia lookup"),
    "riley bike chain": (0, "other", "riley's chat on the shared laptop"),
}


@pytest.fixture(autouse=True)
def no_real_claude(monkeypatch):
    def refuse(*a, **k):
        pytest.fail("a test tried to run the real claude CLI")
    monkeypatch.setattr(propose, "subprocess_runner", refuse)


class Model:
    def __init__(self, scores=None, cost=0.01):
        self.scores = SCORES if scores is None else scores
        self.cost, self.calls = cost, []

    def __call__(self, args, stdin_text, cwd):
        self.calls.append((args, stdin_text))
        items = []
        for key, title in re.findall(r"^## (C\d+)\n\ntitle: (.*)$", stdin_text, re.M):
            if title in self.scores:
                s, w, r = self.scores[title]
                items.append({"key": key, "score": s, "whose": w, "reason": r})
        return {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": self.cost,
                "modelUsage": {filt.FILTER_MODEL: {}}, "structured_output": {"chats": items}}


def chat(cfg, cid, title, prompts, created="2025-07-10"):
    turns = [Turn(p, f"the answer to {p}", created) for p in prompts]
    path, _ = write_chat(cfg, Chat("chatgpt", cid, title, "https://x.invalid", created, created, turns, "http-api"),
                         {}, today=created)
    return path


@pytest.fixture
def chats(cfg):
    chat(cfg, "c-lora", "lora rank sweep", ["I always start my sweeps at rank 8 because it trains fast"])
    chat(cfg, "c-edit", "my editor setup", ["I use vim keys everywhere, set that up for me"], created="2025-08-02")
    chat(cfg, "c-peru", "capital of peru", ["what is the capital of peru"], created="2025-08-03")
    chat(cfg, "c-bike", "riley bike chain", ["riley here, my chain slips on every hill"], created="2025-09-01")
    return cfg


def test_scores_go_to_a_sidecar_not_the_notes(chats, cfg):
    before = {p: p.read_text(encoding="utf-8") for p in (cfg.vault / "3 Resources/chats").rglob("*.md")}
    model = Model()
    assert filt.run(cfg, cap_usd=1.0, runner=model, now=NOW) == 0
    after = {p: p.read_text(encoding="utf-8") for p in (cfg.vault / "3 Resources/chats").rglob("*.md")}
    assert before == after
    scores = filt.load_scores(filt.profile_state(cfg.state, cfg.vault, cfg.vault))
    by_title = {re.sub(r"^\d{4}-\d{2}-\d{2} | \(.*\)\.md$", "", r.split("/")[-1]): e for r, e in scores.items()}
    assert by_title["lora rank sweep"]["score"] == 8
    assert by_title["riley bike chain"]["whose"] == "other"
    assert all(e["month"] for e in scores.values())


def test_the_call_uses_the_cheap_model_and_its_own_ledger_line(chats, cfg):
    model = Model(cost=0.02)
    filt.run(cfg, cap_usd=1.0, runner=model, now=NOW)
    args = model.calls[0][0]
    assert args[args.index("--model") + 1] == filt.FILTER_MODEL
    assert args[args.index("--tools") + 1] == ""
    assert float(args[args.index("--max-budget-usd") + 1]) <= filt.MAX_CALL_USD
    spend = guards.load_json(guards.dream_root(cfg.state) / "spend.json", {})
    assert spend == {f"{DAY} profile": 0.02}


def test_only_the_users_turns_reach_the_model(chats, cfg):
    model = Model()
    filt.run(cfg, cap_usd=1.0, runner=model, now=NOW)
    prompt = model.calls[0][1]
    assert "I always start my sweeps" in prompt
    assert "the answer to" not in prompt


def test_a_rerun_scores_only_what_changed(chats, cfg):
    filt.run(cfg, cap_usd=1.0, runner=Model(), now=NOW)
    model = Model()
    assert filt.run(cfg, cap_usd=1.0, runner=model, now=NOW) == 0
    asked = "".join(c[1] for c in model.calls)
    assert "lora rank sweep" not in asked and "my editor setup" not in asked
    assert "ramen in sapporo" in asked       # the fixture chat the fake never answers waits for the next run


def test_excluded_and_archived_chats_are_never_scored(chats, cfg):
    p = chat(cfg, "c-x", "secret health thing", ["about my knee surgery recovery plan"], created="2025-08-05")
    meta_text = p.read_text(encoding="utf-8").replace("tags: [chat]", "excluded: junk\ntags: [chat]")
    p.write_text(meta_text, encoding="utf-8")
    write(cfg.vault, "4 Archive/chats-excluded/sam/2025-08-01 x (abc).md", "made up\n")
    (cfg.state / filt.EXCLUDE_FILE).write_text("# topics\nvim keys\n", encoding="utf-8")
    model = Model()
    filt.run(cfg, cap_usd=1.0, runner=model, now=NOW)
    prompt = "".join(c[1] for c in model.calls)
    assert "knee surgery" not in prompt
    assert "vim keys" not in prompt          # the excluded topic never reaches the model
    assert "lora rank sweep" in prompt


def test_people_notes_go_in_the_prompt(chats, cfg):
    write(cfg.vault, "2 Areas/People/Riley.md", render({"name": "riley", "source": "author"},
                                                      "A made-up friend who borrows the laptop.\n"))
    model = Model()
    filt.run(cfg, cap_usd=1.0, runner=model, now=NOW)
    assert "A made-up friend who borrows the laptop." in model.calls[0][1]


def test_bad_answers_are_dropped(chats, cfg):
    class Bad(Model):
        def __call__(self, args, stdin_text, cwd):
            r = super().__call__(args, stdin_text, cwd)
            r["structured_output"]["chats"] += [{"key": "C99", "score": 5, "whose": "user", "reason": "x"},
                                                {"key": "C1", "score": 3, "whose": "user", "reason": "repeat"}]
            r["structured_output"]["chats"][0]["score"] = 11
            return r
    filt.run(cfg, cap_usd=1.0, runner=Bad(), now=NOW)
    scores = filt.load_scores(filt.profile_state(cfg.state, cfg.vault, cfg.vault))
    assert all(0 <= e["score"] <= 10 for e in scores.values())
    assert len(scores) == 3      # the out-of-range one waits for the next run


def test_no_cap_means_no_call(chats, cfg):
    model = Model()
    assert filt.run(cfg, runner=model, now=NOW) == 2
    assert model.calls == []


def test_stop_file_in_the_real_vault_stops_a_copy_run(chats, cfg, tmp_path):
    copy = tmp_path / "copy"
    import shutil
    shutil.copytree(cfg.vault, copy)
    write(cfg.vault, "0 Inbox/STOP dream.md", "stop\n")
    model = Model()
    assert filt.run(cfg, vault=copy, cap_usd=1.0, runner=model, now=NOW) == 2
    assert model.calls == []


def test_a_held_dream_lock_doesnt_block_the_filter(chats, cfg):
    with guards.Lock(cfg.state, NOW):
        assert filt.run(cfg, cap_usd=1.0, runner=Model(), now=NOW) == 0


def test_pick_months_takes_the_best_run_and_ties_go_recent():
    counts = {"2025-01": {"about": 5}, "2025-02": {"about": 5}, "2025-03": {"about": 0},
              "2025-06": {"about": 4}, "2025-07": {"about": 3}, "2025-08": {"about": 3}}
    assert filt.pick_months(counts) == ["2025-06", "2025-07", "2025-08"]   # 10 each, the later run wins
    counts["2025-02"]["about"] = 6
    assert filt.pick_months(counts) == ["2025-01", "2025-02", "2025-03"]
    assert filt.pick_months({"2025-12": {"about": 1}, "2026-01": {"about": 1}}) == ["2025-12", "2026-01"]
    assert filt.months_between("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]


def test_about_user_needs_the_user_and_the_score():
    assert filt.about_user({"whose": "user", "score": filt.PROFILE_MIN})
    assert not filt.about_user({"whose": "user", "score": filt.PROFILE_MIN - 1})
    assert not filt.about_user({"whose": "other", "score": 10})


def test_each_run_is_recorded_with_its_cost(chats, cfg):
    filt.run(cfg, cap_usd=1.0, runner=Model(cost=0.03), now=NOW)
    runs = guards.load_json(filt.profile_state(cfg.state, cfg.vault, cfg.vault) / filt.FILTER_RUNS, [])
    assert len(runs) == 1 and runs[0]["cost"] == 0.03 and runs[0]["scored"] == 4
