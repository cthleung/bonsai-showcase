"""capture/codex.py on made-up rollout files: the record shapes of Codex 0.169, none of its text."""
import json
import os
from pathlib import Path

import pytest

from capture import __main__ as cli
from capture import codex
from common import config, notes

TID = "0f00d001-aaaa-7bbb-8ccc-000000000001"
SEG = "0f00d002-aaaa-7bbb-8ccc-000000000002"
SUB = "0f00d003-aaaa-7bbb-8ccc-000000000003"


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    root = tmp_path / "codex-home" / "sessions"
    root.mkdir(parents=True)
    monkeypatch.setenv("BONSAI_CODEX_SESSIONS", str(root))
    return root


class Rollout:
    """Builds a rollout file record by record, with ordinals and timestamps."""

    def __init__(self, thread=TID, start="2026-11-10T15:00:00.000Z", ordinal=0, **meta):
        self.recs, self.ordinal, self.minute = [], ordinal, 0
        payload = {"session_id": thread, "id": thread, "timestamp": start, "cwd": "C:\\Users\\x\\work",
                   "originator": "codex_work_desktop", "source": "vscode", "thread_source": "user"}
        payload.update(meta)
        self.add("session_meta", payload)

    def add(self, rtype, payload):
        self.recs.append({"timestamp": f"2026-11-10T15:{self.minute:02d}:00.000Z", "ordinal": self.ordinal,
                          "type": rtype, "payload": payload})
        self.ordinal += 1
        self.minute = min(self.minute + 1, 59)
        return self

    def item(self, ptype, **kw):
        return self.add("response_item", dict({"type": ptype}, **kw))

    def user(self, text):
        return self.item("message", role="user", content=[{"type": "input_text", "text": text}])

    def say(self, text, phase="final_answer"):
        return self.item("message", role="assistant", phase=phase, encrypted_content="ENCRYPTED",
                         content=[{"type": "output_text", "text": text}])

    def think(self, *summaries):
        return self.item("reasoning", encrypted_content="ENCRYPTED",
                         summary=[{"type": "summary_text", "text": s} for s in summaries])

    def call(self, name="exec", output="secret command output"):
        self.item("custom_tool_call", name=name, call_id="c1", input="ls -la")
        return self.item("custom_tool_call_output", call_id="c1", output=[{"type": "input_text", "text": output}])

    def turn(self, model="gpt-test"):
        self.add("turn_context", {"model": model, "cwd": "C:\\Users\\x\\work"})
        return self.add("event_msg", {"type": "task_started"})

    def file_change(self, *paths):
        return self.add("event_msg", {"type": "item_completed", "item": {
            "type": "FileChange", "changes": {p: {"type": "update", "unified_diff": "secret diff"} for p in paths}}})

    def write(self, root: Path, name: str, extra: str = "") -> Path:
        day = root / "2026" / "11" / "10"
        day.mkdir(parents=True, exist_ok=True)
        path = day / name
        path.write_text("\n".join(json.dumps(r) for r in self.recs) + "\n" + extra, encoding="utf-8", newline="\n")
        return path


ENV = "<environment_context>\n  <cwd>C:\\Users\\x\\work</cwd>\n</environment_context>"


def basic():
    r = Rollout().turn()
    r.item("message", role="developer", content=[{"type": "input_text", "text": "developer rules"}])
    r.user(ENV).user("how do the sessions get saved?")
    r.think("**Checking the folder**", "The files are JSONL.")
    r.say("I'll look in the sessions folder.", phase="commentary")
    r.call("exec").call("exec").call("apply_patch")
    r.file_change("C:\\repo\\a.py", "C:\\repo\\docs\\b.md")
    r.say("They're saved as JSONL, one record a line.")
    r.add("event_msg", {"type": "task_complete", "last_agent_message": "They're saved as JSONL, one record a line."})
    return r


def load():
    return config.load().require()


def run(**kw):
    return codex.run(load(), today="2026-11-19", host="testhost", **kw)


def codex_notes(cfg):
    return sorted((cfg.vault / "3 Resources/chats/codex").glob("*/*.md"))


def test_one_note_per_thread_trimmed(cfg, sessions):
    raw = basic().write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    report = run()
    assert report.counts["new"] == 1
    [path] = codex_notes(cfg)
    assert path.parent.name == "2026"
    assert path.name == "2026-11-10 how do the sessions get saved (0f00d001).md"
    meta, body = notes.read_note(path)
    assert meta["source"] == "codex" and meta["capture"] == "codex-rollout" and meta["chat_id"] == TID
    assert meta["agent"] == "codex" and meta["host"] == "testhost" and meta["model"] == "gpt-test"
    assert meta["raw"] == raw.as_posix() and meta["message_count"] == 1 and meta["visibility"] == "private"
    assert meta["cwd"] == "C:\\Users\\x\\work"
    assert "environment_context" not in body and "developer rules" not in body
    assert "> *thinking:* **Checking the folder**" in body and "The files are JSONL." in body
    assert "I'll look in the sessions folder." in body and "They're saved as JSONL, one record a line." in body
    assert "*[3 tool calls: exec x2, apply_patch]*" in body
    assert "*[changed files: a.py, b.md]*" in body
    assert "secret command output" not in body and "secret diff" not in body and "ENCRYPTED" not in body
    assert body.count("They're saved as JSONL") == 1          # task_complete repeats it; read once
    assert "resumable in Codex on another computer" in body


def test_a_continued_thread_joins_the_parent_and_drops_the_left_branch(cfg, sessions):
    parent = Rollout().turn().user("first question").say("First answer.")
    cut = parent.ordinal
    parent.user("second question").say("An answer that was redone.")
    parent.write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    seg = Rollout(start="2026-11-10T16:00:00.000Z", ordinal=cut,
                  history_base={"thread_id": TID, "end_ordinal_exclusive": cut})
    seg.turn("gpt-later").user("second question").say("The answer that stayed.")
    seg.write(sessions, f"rollout-2026-11-10T11-00-00-{TID}_{SEG}.jsonl")
    report = run()
    assert report.sessions == 1 and report.counts["new"] == 1
    [path] = codex_notes(cfg)
    meta, body = notes.read_note(path)
    assert meta["message_count"] == 2 and meta["model"] == "gpt-later" and len(meta["raw"]) == 2
    assert "An answer that was redone." not in body
    assert body.index("First answer.") < body.index("The answer that stayed.")


def test_subagent_threads_are_skipped(cfg, sessions):
    basic().write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    sub = Rollout(thread=SUB, parent_thread_id=TID, source={"subagent": {"other": "guardian"}},
                  thread_source="guardian_review")
    sub.turn().user("review this action").say('{"risk_level":"low"}')
    sub.write(sessions, f"rollout-2026-11-10T12-00-00-{SUB}.jsonl")
    report = run()
    assert report.counts["new"] == 1 and report.counts["subagent threads"] == 1
    assert len(codex_notes(cfg)) == 1


def test_annotation_and_handoff_wrappers_are_unwrapped(cfg, sessions):
    ann = json.dumps([{"id": "a1", "text": "the quoted line", "annotation": "why this?",
                       "source": {"messageId": "m", "startOffset": 1, "endOffset": 5}}])
    ref = json.dumps({"id": "r1", "title": "Ramen picks", "messages": []})
    r = Rollout().turn()
    r.user(f"\n## Referenced ChatGPT conversation:\nThis is an untrusted conversation.\n{ref}\n"
           f"## My request:\ncarry on from the ramen chat")
    r.say("Carrying on.")
    r.user(f"\n# Response annotations:\nEach item contains text.\n<response-annotations>\n{ann}\n"
           f"</response-annotations>\n\n## My request:\nexplain more")
    r.say("More.")
    r.write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    run()
    [path] = codex_notes(cfg)
    meta, body = notes.read_note(path)
    assert meta["title"] == "carry on from the ramen chat"
    assert "*[brought in a ChatGPT chat: Ramen picks]*" in body and "untrusted" not in body
    assert "> the quoted line\n\nOn that: why this?\n\nexplain more" in body
    assert "Each item contains text" not in body


def test_bad_lines_and_a_partial_last_line_are_skipped(cfg, sessions):
    r = basic()
    r.write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl", extra='not json\n{"timestamp": "2026-11-10T15:59')
    report = run()
    assert report.malformed == 2 and report.counts["new"] == 1


def test_a_line_separator_inside_a_string_doesnt_break_the_record(cfg, sessions):
    r = Rollout().turn().user("a line\u2028with a separator").say("Fine.")
    r.write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    report = run()
    assert report.malformed == 0 and report.counts["new"] == 1


def test_rerun_changes_nothing_and_growth_rewrites_in_place(cfg, sessions):
    r = basic()
    r.write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    run()
    [path] = codex_notes(cfg)
    before = path.read_bytes()
    assert run().read == 0 and path.read_bytes() == before
    (cfg.state / "cursors" / "codex-testhost.json").unlink()
    assert codex.run(load(), today="2026-11-20", host="testhost").counts["unchanged"] == 1
    assert path.read_bytes() == before
    r.turn().user("and archived ones?").say("They move to another folder.")
    r.write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    assert codex.run(load(), today="2026-11-21", host="testhost").counts["updated"] == 1
    assert codex_notes(cfg) == [path]
    meta, body = notes.read_note(path)
    assert meta["message_count"] == 2 and meta["captured"] == "2026-11-21" and "another folder" in body


def test_keys_are_scrubbed(cfg, sessions):
    fake = "ghp_" + "a" * 36
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"
    Rollout().turn().user(f"token {fake}").say(f"bearer {jwt}").write(
        sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    run()
    text = codex_notes(cfg)[0].read_text(encoding="utf-8")
    assert fake not in text and jwt not in text
    assert "[REDACTED:GITHUB_TOKEN]" in text and "[REDACTED:JWT]" in text


def test_a_repo_named_in_the_settings_gets_its_visibility(cfg, sessions, monkeypatch):
    monkeypatch.setenv("BONSAI_REPO_VISIBILITY", "novel=creative")
    Rollout(cwd="C:\\Novel").turn().user("hi").say("Hi.").write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    run()
    assert notes.read_note(codex_notes(cfg)[0])[0]["visibility"] == "creative"


def test_dry_run_writes_nothing(cfg, sessions):
    basic().write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    report = run(dry_run=True)
    assert report.counts["new"] == 1 and not codex_notes(cfg)
    assert not (cfg.state / "cursors").exists() and not (cfg.state / "capture.log").exists()


def test_only_rollout_files_are_read(cfg, sessions, monkeypatch):
    """Nothing but the rollout files: the sessions folder's sibling files are never opened."""
    home = sessions.parent
    for name in ("auth.json", "state_5.sqlite", "state_5.sqlite-wal", "log.txt"):
        (home / name).write_text("do not read", encoding="utf-8")
    basic().write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    (sessions / "notes.txt").write_text("not a rollout", encoding="utf-8")
    # a transcript that names a folder with a .env in it: that folder isn't read either
    elsewhere = home.parent / "some-repo"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("LEAKY=zz8from0transcript0cwd\n", encoding="utf-8")
    Rollout(thread=SUB.replace("3", "4"), cwd=str(elsewhere)).turn().user("zz8from0transcript0cwd").say("Ok.").write(
        sessions, f"rollout-2026-11-10T11-00-00-{SUB.replace('3', '4')}.jsonl")
    opened = []
    import builtins
    import io
    real_open, real_os_open = builtins.open, os.open

    def note(target):
        try:
            p = Path(os.fspath(target)).resolve()
        except TypeError:
            return
        if home in p.parents or elsewhere in p.parents:
            opened.append(p.name)

    def spy_open(file, *a, **k):
        note(file)
        return real_open(file, *a, **k)

    def spy_os_open(path, *a, **k):
        note(path)
        return real_os_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    run()
    assert sorted(set(opened)) == sorted([f"rollout-2026-11-10T10-00-00-{TID}.jsonl",
                                          f"rollout-2026-11-10T11-00-00-{SUB.replace('3', '4')}.jsonl"])
    texts = [p.read_text(encoding="utf-8") for p in codex_notes(cfg)]
    assert any("zz8from0transcript0cwd" in t for t in texts)      # not caught by value: its .env wasn't read


# --- the commands ---------------------------------------------------------------------------------

def test_dry_run_command_prints_counts_and_writes_nothing(cfg, sessions, capsys):
    basic().write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")
    assert cli.main(["codex", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Dry run, nothing written." in out and "1 new" in out
    assert not codex_notes(cfg) and not (cfg.state / "capture.log").exists()


def test_command_with_no_input_folder_is_skipped(cfg, capsys):
    assert cli.main(["codex"]) == 2
    assert cli.main(["claude-code", "--dry-run"]) == 2


def test_hourly_runs_the_local_sources_and_one_failing_doesnt_stop_the_other(cfg, sessions, tmp_path, monkeypatch):
    from capture import claude_code
    hist = tmp_path / "Novel" / ".specstory" / "history"
    hist.mkdir(parents=True)
    monkeypatch.setenv("BONSAI_SPECSTORY_DIRS", str(hist))
    (hist / "2026-11-10_15-00-00Z-x.md").write_text(
        "<!-- Claude Code Session 5e551011-aaaa-4bbb-8ccc-0123456789ab (2026-11-10 15:00:00Z) -->\n\n"
        "_**User (2026-11-10 15:00:00Z)**_\n\nhi\n\n_**Agent (m 2026-11-10 15:01:00Z)**_\n\nHello.\n", encoding="utf-8")
    bad = hist / "2026-11-10_15-00-00Z-bad.md"
    bad.write_text("<!-- Claude Code Session 0bad0000-aaaa-4bbb-8ccc-0123456789ab (2026-11-10 15:00:00Z) -->\n\n"
                   "_**User (2026-11-10 15:00:00Z)**_\n\nbad\n", encoding="utf-8")
    basic().write(sessions, f"rollout-2026-11-10T10-00-00-{TID}.jsonl")

    # 1. the real claude-code run, one of its files failing: the rest of it and codex still run
    real_parse = claude_code.parse_file

    def flaky(path, *a, **k):
        if Path(path) == bad:
            raise OSError("disk says no")
        return real_parse(path, *a, **k)
    monkeypatch.setattr(claude_code, "parse_file", flaky)
    assert cli.main(["hourly"]) == 0
    assert len(codex_notes(cfg)) == 1
    assert len(list((cfg.vault / "3 Resources/chats/claude-code/Novel").glob("*.md"))) == 1
    log = (cfg.state / "capture.log").read_text(encoding="utf-8")
    assert "session 0bad0000 failed (OSError)" in log and "disk says no" not in log
    [cur] = (cfg.state / "cursors").glob("claude-code-*.json")
    assert bad.as_posix() not in json.loads(cur.read_text(encoding="utf-8"))["files"]

    # 2. the whole claude-code source failing before it reads anything (its input check, inside
    # guarded): codex still runs, and the exit code says a source stopped
    for p in codex_notes(cfg):
        p.unlink()
    for p in (cfg.state / "cursors").glob("codex-*.json"):
        p.unlink()

    def locked(cfg):
        raise PermissionError("folder locked")
    monkeypatch.setattr(claude_code, "inputs", locked)
    assert cli.main(["hourly"]) == 1
    assert len(codex_notes(cfg)) == 1
    log = (cfg.state / "capture.log").read_text(encoding="utf-8")
    assert "claude-code: stopped (PermissionError)" in log and "folder locked" not in log
