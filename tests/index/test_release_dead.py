"""index.server.release_dead: the servers a killed run left behind. No real process is
started or stopped: procs.alive and procs.kill are fakes."""
from __future__ import annotations

import json

import pytest

from index import server


def entries(s, data):
    s.index_dir.mkdir(parents=True, exist_ok=True)
    s.servers_file.write_text(json.dumps(data), encoding="utf-8")


def left(s) -> dict:
    return json.loads(s.servers_file.read_text(encoding="utf-8"))


def test_stops_only_auto_stop_servers_the_dead_run_used(settings, monkeypatch):
    dead, live = [111, "dead"], [222, "live"]
    alive = {(10, "e"), (20, "r"), (222, "live")}
    killed = []
    monkeypatch.setattr(server.procs, "alive", lambda pid, token: (pid, token) in alive)
    monkeypatch.setattr(server.procs, "kill", lambda pid, token: killed.append(pid) or True)
    entries(settings, {
        "embed": {"pid": 10, "token": "e", "auto_stop": True, "users": [dead]},
        "rerank": {"pid": 20, "token": "r", "auto_stop": True, "users": [dead, live]},
    })
    assert server.release_dead(settings, dead) == ["embed"]
    assert killed == [10]
    assert list(left(settings)) == ["rerank"]
    assert left(settings)["rerank"]["users"] == [live]          # still in use, so it stays


def test_leaves_a_serve_server_and_other_runs_servers(settings, monkeypatch):
    dead = [111, "dead"]
    monkeypatch.setattr(server.procs, "alive", lambda pid, token: pid in (10, 20))
    monkeypatch.setattr(server.procs, "kill", lambda pid, token: pytest.fail("nothing to stop"))
    entries(settings, {
        "embed": {"pid": 10, "token": "e", "auto_stop": False, "users": [dead]},       # from `index serve`
        "rerank": {"pid": 20, "token": "r", "auto_stop": True, "users": [[333, "x"]]},  # not the dead run's
    })
    assert server.release_dead(settings, dead) == []
    assert set(left(settings)) == {"embed", "rerank"}


def test_drops_entries_whose_server_is_gone(settings, monkeypatch):
    monkeypatch.setattr(server.procs, "alive", lambda pid, token: False)
    monkeypatch.setattr(server.procs, "kill", lambda pid, token: pytest.fail("nothing to stop"))
    entries(settings, {"embed": {"pid": 10, "token": "e", "auto_stop": True, "users": [[111, "dead"]]}})
    assert server.release_dead(settings, [111, "dead"]) == []
    assert left(settings) == {}
